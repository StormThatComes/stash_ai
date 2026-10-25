"""Bounded CPU preparation/JPEG work around one caller-owned inference stage."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from itertools import islice

from stash_ai.runtime.cancellation import Cancellation, use
from stash_ai.runtime.cleanup import cleanup
from stash_ai.runtime import shutdown


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
            if key not in self.images:self.images[key]=[image,image.copy(),0]
            entry=self.images[key];entry[2]+=1
            return key,entry[1]

    def _release_image(self, key):
        with self.lock:
            entry=self.images[key];entry[2]-=1
            if not entry[2]:
                del self.images[key];entry[1].close()

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
        while True:
            with self.lock:
                if self.bytes + size <= self.limit and len(self.pending) < self.workers*2:
                    break
                oldest = next(iter(self.pending))
            oldest.result()
            self.cancel.check()
        key,owned = self._acquire_image(image)
        try:
            future = self.executor.submit(self._call,function,owned,value,key)
        except BaseException:
            self._release_image(key)
            raise
        with self.lock:
            self.pending[future] = size
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes,self.bytes)
        def completed(done):
            if done.cancelled():
                self._release_image(key)
            with self.lock:
                self.bytes -= self.pending.pop(done,0)
        future.add_done_callback(completed)
        return future

    def ordered(self, function, image, values, *, extra=0):
        size = image.width*image.height*3*8 + extra + 4096
        capacity = max(1,min(self.workers*2,self.limit//size))
        pending = deque()
        try:
            for value in values:
                if len(pending) >= capacity:
                    yield pending.popleft().result()
                pending.append(self.submit(function,image,value,extra=extra))
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()

    def close(self):
        self.cancel.cancel()
        if self.executor is not None:
            shutdown.call(lambda: self.executor.shutdown(wait=True, cancel_futures=True), self)


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
                    encoded = [stage.submit(encode,image,entry,extra=config.data['storage']['max_crop_bytes']) for entry in batch]
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
                    encoded = [stage.submit(encode,image,entry,extra=config.data['storage']['max_crop_bytes']) for entry in batch]
                    for entry in accepted:
                        entry['vector'] = normalized_embedding(backend.embed(image, entry['face']), backend.dimensions).astype('<f4').tobytes()
                for entry, future in zip(batch,encoded):
                    crop = future if defer_encoding else future.result()['crop']
                    yield (entry['index'], entry['face'], entry['quality'], entry['reason'], entry['vector'], crop)
    finally:
        if owned_stage:
            cleanup([('stop recognition pixel workers', stage.close)])


def cue_batches(frames, maximum, limit):
    """Own frame copies until an entire ordered microbatch has been consumed."""
    pending, size = [], 0
    try:
        for index, cue, image in frames:
            required = image.width*image.height*3*8 + 4096
            if pending and (len(pending) >= maximum or size+required > limit):
                yield pending
                for _, _, owned in pending:
                    owned.close()
                pending, size = [], 0
            try:
                owned = image.copy()
            finally:
                image.close()
            pending.append((index, cue, owned))
            size += required
            if required > limit:
                yield pending
                for _, _, owned in pending:
                    owned.close()
                pending, size = [], 0
        if pending:
            yield pending
    finally:
        for _, _, owned in pending:
            owned.close()


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
