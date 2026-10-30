"""Bounded CPU preparation/JPEG work around one caller-owned inference stage."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from itertools import islice

from stash_ai.runtime.cancellation import Cancellation, use
from stash_ai.runtime.cleanup import cleanup
from stash_ai.runtime import shutdown


class OwnedFuture(Future):
    """The reservation follows the result until its consumer takes it."""
    def __init__(self,stage):
        super().__init__();self.stage=stage

    def result(self,timeout=None):
        try:return super().result(timeout)
        finally:
            if self.done():self.stage.retire(self)

    def cancel(self):
        # The executor owns cancellation and the lifetime of copied images.
        return False


class CpuStage:
    def __init__(self, workers, limit):
        self.workers, self.limit = workers, limit
        self.executor = None
        self.cancel = Cancellation()
        self.pending = {}
        self.images = {}
        self.bytes = self.peak_bytes = 0
        self.lock = threading.Lock()

    def _acquire_image(self, image):
        key=id(image)
        with self.lock:
            if key not in self.images:
                size=image.width*image.height*3*8
                self.images[key]=[image,image.copy(),0,size]
                self.bytes+=size
                self.peak_bytes=max(self.peak_bytes,self.bytes)
            entry=self.images[key];entry[2]+=1
            return key,entry[1]

    def _release_image(self, key):
        with self.lock:
            entry=self.images[key];entry[2]-=1
            if not entry[2]:
                del self.images[key];self.bytes-=entry[3];entry[1].close()

    def _call(self, function, image, value, key):
        try:
            with use(self.cancel):
                return function(image, value)
        finally:
            self._release_image(key)

    def submit(self, function, image, value, *, extra=0):
        """Submit owned CPU pixels; inference and database work stay on caller."""
        size = image.width*image.height*3*8 + extra + 4096
        self.cancel.check()
        if self.workers <= 1 or size > self.limit:
            future = Future()
            future.set_result(function(image,value))
            return future
        if self.executor is None:
            shutdown.require_idle()
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='recognition-pixels')
        if not self.can_submit(image,extra=extra):
            raise BufferError('Consume recognition CPU results before submitting another job.')
        key,owned = self._acquire_image(image)
        future=OwnedFuture(self)
        with self.lock:
            self.pending[future]=extra+4096
            self.bytes+=extra+4096
            self.peak_bytes=max(self.peak_bytes,self.bytes)
        try:
            job = self.executor.submit(self._call,function,owned,value,key)
        except BaseException:
            self._release_image(key)
            self.retire(future)
            raise
        def completed(done):
            if done.cancelled():
                self._release_image(key)
                Future.cancel(future);self.retire(future)
            else:
                try:value=done.result()
                except BaseException as error:future.set_exception(error)
                else:future.set_result(value)
        job.add_done_callback(completed)
        return future

    def can_submit(self,image,*,extra=0):
        with self.lock:
            image_size=0 if id(image) in self.images else image.width*image.height*3*8
            return self.bytes+image_size+extra+4096<=self.limit

    def retire(self,future):
        with self.lock:self.bytes-=self.pending.pop(future,0)

    def ordered(self, function, image, values, *, extra=0):
        size = image.width*image.height*3*8 + extra + 4096
        capacity = max(1,min(self.workers*2,self.limit//size))
        pending = deque()
        try:
            for value in values:
                while pending and (len(pending)>=capacity or not self.can_submit(image,extra=extra)):
                    yield pending.popleft().result()
                # Nested preparation may share a stage with an outer iterator.
                # Its results stay reserved; use the existing scalar fallback.
                if not self.can_submit(image,extra=extra):
                    yield function(image,value)
                    continue
                pending.append(self.submit(function,image,value,extra=extra))
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                try:future.result()
                except BaseException:pass

    def close(self):
        self.cancel.cancel()
        if self.executor is not None:
            shutdown.call(lambda: self.executor.shutdown(wait=True, cancel_futures=True), self)
        with self.lock:self.pending.clear();self.bytes=0


def process(image, faces, backend, visibility, config, cue, *, assess, rejection, crop_bytes, geometry_contract, stage=None, defer_embeddings=False, defer_encoding=False):
    """Yield complete face records; caller checkpoints only after the whole cue."""
    from .models import normalized_embedding
    from .pose_policy import measurement_required
    from .visibility import VisibilityError, VisibilityGeometryError, cached
    from stash_ai.runtime.cleanup import closing
    settings = config.data['execution']
    owned_stage = stage is None
    stage = stage or CpuStage(settings['scan_cpu_workers'], settings['scan_queue_bytes']//2)
    prepare_embedding = getattr(backend, 'prepare_embedding', None)
    embed_prepared = getattr(backend, 'embed_prepared_many', None)
    prepared_supported = callable(prepare_embedding) and callable(embed_prepared)
    prepared_visibility = callable(getattr(visibility,'measure_prepared_many',None))

    def prepare(pixels, pair):
        index, face = pair
        quality = assess(pixels, face)
        quality['geometry_contract'] = geometry_contract(face, cue)
        reason = ('face_detection_score' if face.score <= config.data['detection']['score_threshold']
                  else rejection(quality, config.data['detection']))
        visibility_input = None
        tensor = None  # Alignment runs only after the visibility admission gate.
        return dict(index=index, face=face, quality=quality, reason=reason, tensor=tensor,
                    visibility_input=visibility_input, vector=None, crop=None)

    def visibility_pixels(pixels, entry):
        from .visibility import prepare as prepare_visibility
        try:
            entry['visibility_input'] = prepare_visibility(pixels,entry['face'])
        except VisibilityGeometryError as error:
            entry['visibility_input'] = error
        return entry

    def align(pixels, entry):
        entry['tensor'] = prepare_embedding(pixels, entry['face'])
        return entry

    def encode(pixels, entry):
        if entry['reason'] is None:
            details = {}
            entry['crop'], bounds = crop_bytes(pixels, entry['face'], limit=config.data['storage']['max_crop_bytes'], details=details)
            entry['quality'].update({key: details.pop(key) for key in tuple(details) if key.startswith('evidence_')})
            entry['quality'].update(crop_bounds=bounds, crop_encoding=details)
            import hashlib
            entry['quality']['evidence_crop_sha256'] = hashlib.sha256(entry['crop']).hexdigest()
        return entry

    def encode_jobs(entries):
        pending=[];consumed=0
        extra=config.data['storage']['max_crop_bytes']
        for entry in entries:
            while consumed<len(pending) and not stage.can_submit(image,extra=extra):
                pending[consumed].result();consumed+=1
            if stage.can_submit(image,extra=extra):
                pending.append(stage.submit(encode,image,entry,extra=extra))
            else:
                # An outer preparation iterator owns the remaining reservations.
                # This entry is consumed directly into the caller's cue batch.
                future=Future();future.set_result(encode(image,entry));pending.append(future)
                consumed=len(pending)
        return pending

    try:
        with closing(stage.ordered(prepare, image, enumerate(faces), extra=4*256*256*4 if prepared_visibility else 0), name='close scan preparation') as stream:
            batch_size = max(1, min(32, max(settings['scan_cpu_workers']*2, settings['recognizer_batch_size']),
                stage.limit//(3*112*112*4 + config.data['storage']['max_crop_bytes'] + 4096)))
            while batch := list(islice(stream, batch_size)):
                visible = [entry for entry in batch if entry['reason'] is None]
                if visible and measurement_required(config):
                    if visibility is None:
                        raise VisibilityError('Recognition visibility gate requires a visibility model.')
                    supports = getattr(visibility,'supports_prepared',None)
                    use_prepared = prepared_visibility and (not callable(supports) or supports())
                    if use_prepared:
                        with closing(stage.ordered(visibility_pixels,image,visible,extra=4*256*256*4),
                                     name='close visibility preparation') as prepared:
                            visible = list(prepared)
                    for offset in range(0,len(visible),8):
                        chunk = visible[offset:offset+8]
                        many = getattr(visibility,'measure_many',None)
                        if use_prepared:
                            measurements = visibility.measure_prepared_many([e['visibility_input'] for e in chunk])
                        elif callable(many):
                            measurements = many([(image,e['face']) for e in chunk])
                        else:
                            measurements = []
                            for entry in chunk:
                                try:
                                    measurements.append(visibility.measure(image,entry['face']))
                                except VisibilityGeometryError as error:
                                    measurements.append(error)
                        if len(measurements) != len(chunk):
                            raise VisibilityError('Recognition visibility returned an incompatible batch.')
                        for entry, measurement in zip(chunk,measurements):
                            entry['visibility_input'] = None
                            if isinstance(measurement,VisibilityGeometryError):
                                entry['reason'] = 'face_visibility_unmeasurable'
                                continue
                            if isinstance(measurement,BaseException):
                                raise measurement
                            if not isinstance(measurement,dict) or not cached(measurement):
                                raise VisibilityError('Recognition visibility gate received an invalid measurement.')
                            entry['quality'].update(measurement)
                            if measurement['visible_face_fraction'] < config.data['detection']['min_visible_face_fraction']:
                                entry['reason'] = 'face_occluded'
                accepted = [entry for entry in batch if entry['reason'] is None]
                # JPEG workers can overlap owner-thread inference. Their owned
                # copies stay alive through cancellation and deferred cue commits.
                if prepared_supported:
                    with closing(stage.ordered(align, image, accepted, extra=3*112*112*4),
                                 name='close face alignment preparation') as aligned:
                        accepted = list(aligned)
                    encoded = encode_jobs(batch)
                    step = settings['recognizer_batch_size']
                    if defer_embeddings:
                        for entry in accepted:
                            entry['vector'] = entry['tensor']
                    for offset in range(0, 0 if defer_embeddings else len(accepted), step):
                        chunk = accepted[offset:offset+step]
                        vectors = embed_prepared([e['tensor'] for e in chunk])
                        if len(vectors) != len(chunk):
                            raise ValueError('Face recognizer returned an incompatible output batch.')
                        for entry, value in zip(chunk, vectors):
                            entry['vector'] = normalized_embedding(value, backend.dimensions).astype('<f4').tobytes()
                else:
                    encoded = encode_jobs(batch)
                    for entry in accepted:
                        entry['vector'] = normalized_embedding(backend.embed(image, entry['face']), backend.dimensions).astype('<f4').tobytes()
                for entry, future in zip(batch,encoded):
                    crop = future if defer_encoding else future.result()['crop']
                    yield (entry['index'], entry['face'], entry['quality'], entry['reason'], entry['vector'], crop)
    finally:
        if owned_stage:
            cleanup([('stop recognition pixel workers', stage.close)])


def cue_batches(frames, maximum, limit):
    """Transfer prefetched images, or own one copy of an expiring source crop."""
    from .frame_ownership import CueBatch
    pending, size = CueBatch(), 0
    transferred = getattr(frames, 'transfers_images', False)
    try:
        for index, cue, image in frames:
            required = image.width*image.height*3*8 + 4096
            if transferred:
                owned = image
            else:
                try:owned = image.copy()
                finally:image.close()
            try:
                if pending and (len(pending) >= maximum or size+required > limit):
                    yield pending
                    pending.close()
                    pending, size = CueBatch(), 0
                pending.add(index, cue, owned)
                owned = None
                size += required
                if required > limit:
                    yield pending
                    pending.close()
                    pending, size = CueBatch(), 0
            finally:
                # The next transferred frame can already be in hand when a
                # caller stops while consuming the previous microbatch.
                if owned is not None:owned.close()
        if pending:
            yield pending
    finally:
        pending.close()


def process_cues(batch, backend, visibility, config, *, stage, **callbacks):
    """Cross-cue embedding batches; the coordinator remains the inference owner."""
    from .models import normalized_embedding
    supported = (callable(getattr(backend,'prepare_embedding',None)) and
                 callable(getattr(backend,'embed_prepared_many',None)))
    records, positions, tensors = [], [], []
    step=config.data['execution']['recognizer_batch_size']
    def infer():
        if not tensors:
            return
        vectors=backend.embed_prepared_many(tensors)
        if len(vectors)!=len(tensors):
            raise ValueError('Face recognizer returned an incompatible output batch.')
        for row,value in zip(positions,vectors):
            row[4]=normalized_embedding(value,backend.dimensions).astype('<f4').tobytes()
        positions.clear();tensors.clear()
    def finished():
        for index,cue,values in records:
            for row in values:
                if isinstance(row[5],Future):
                    row[5] = row[5].result()['crop']
            yield index,cue,values
    retained=0
    for index, cue, image in batch:
        faces = sorted(backend.detect(image),key=lambda face:tuple(face.row))
        values=[]
        from stash_ai.runtime.cleanup import closing
        with closing(process(image,faces,backend,visibility,config,cue,
                     stage=stage,defer_embeddings=supported,defer_encoding=True,**callbacks),name='close deferred cue') as prepared:
            for result in prepared:
                row=list(result);values.append(row)
                retained+=(config.data['storage']['max_crop_bytes'] if row[3] is None else 0)+4096
                if supported and row[3] is None:
                    positions.append(row);tensors.append(row[4])
                    if len(tensors)>=step:
                        infer()
        records.append((index,cue,values))
        if retained >= config.data['execution']['scan_queue_bytes']//2:
            infer()
            yield from finished()
            records=[];retained=0
    infer()
    yield from finished()
