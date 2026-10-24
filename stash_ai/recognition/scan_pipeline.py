"""Bounded CPU preparation/JPEG work around one caller-owned inference stage."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from stash_ai.runtime.cancellation import Cancellation, use
from stash_ai.runtime.cleanup import cleanup
from stash_ai.runtime import shutdown


class CpuStage:
    def __init__(self, workers, limit):
        self.workers, self.limit = workers, limit
        self.executor = None
        self.cancel = Cancellation()

    def _call(self, function, image, value):
        try:
            with use(self.cancel):
                return function(image, value)
        finally:
            image.close()

    def ordered(self, function, image, values, *, extra=0):
        # Every worker owns its copy, including cancellation and timed-out joins.
        size = image.width*image.height*3*8 + extra + 4096
        capacity = min(self.workers*2, self.limit//size)
        if self.workers <= 1 or not capacity:
            for value in values:
                self.cancel.check()
                yield function(image, value)
            return
        if self.executor is None:
            shutdown.require_idle()
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='recognition-pixels')
        pending = deque()
        try:
            for value in values:
                if len(pending) >= capacity:
                    yield pending.popleft().result()
                self.cancel.check()
                owned = image.copy()
                try:
                    future = self.executor.submit(self._call, function, owned, value)
                except BaseException:
                    owned.close()
                    raise
                future.add_done_callback(lambda f, owned=owned: owned.close() if f.cancelled() else None)
                pending.append(future)
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()

    def close(self):
        self.cancel.cancel()
        if self.executor is not None:
            shutdown.call(lambda: self.executor.shutdown(wait=True, cancel_futures=True), self)


def process(image, faces, backend, visibility, config, cue, *, assess, rejection, crop_bytes, geometry_contract, stage=None):
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

    def prepare(pixels, pair):
        index, face = pair
        quality = assess(pixels, face)
        quality['geometry_contract'] = geometry_contract(face, cue)
        reason = ('face_detection_score' if face.score <= config.data['detection']['score_threshold']
                  else rejection(quality, config.data['detection']))
        tensor = None  # Alignment runs only after the visibility admission gate.
        return dict(index=index, face=face, quality=quality, reason=reason, tensor=tensor, vector=None, crop=None)

    def align(pixels, entry):
        entry['tensor'] = prepare_embedding(pixels, entry['face'])
        return entry

    def encode(pixels, entry):
        if entry['reason'] is None:
            details = {}
            entry['crop'], bounds = crop_bytes(pixels, entry['face'], limit=config.data['storage']['max_crop_bytes'], details=details)
            entry['quality'].update(crop_bounds=bounds, crop_encoding=details)
            import hashlib
            entry['quality']['evidence_crop_sha256'] = hashlib.sha256(entry['crop']).hexdigest()
        return entry

    try:
        with closing(stage.ordered(prepare, image, enumerate(faces)), name='close scan preparation') as stream:
            batch_size = max(1, min(32, max(settings['scan_cpu_workers']*2, settings['recognizer_batch_size']),
                stage.limit//(3*112*112*4 + config.data['storage']['max_crop_bytes'] + 4096)))
            while batch := list(islice(stream, batch_size)):
                accepted = []
                for entry in batch:
                    quality, face = entry['quality'], entry['face']
                    if entry['reason'] is None and measurement_required(config):
                        if visibility is None:
                            raise VisibilityError('Recognition visibility gate requires a visibility model.')
                        try:
                            quality.update(visibility.measure(image, face))
                        except VisibilityGeometryError:
                            entry['reason'] = 'face_visibility_unmeasurable'
                        else:
                            if not cached(quality):
                                raise VisibilityError('Recognition visibility gate received an invalid measurement.')
                            if quality['visible_face_fraction'] < config.data['detection']['min_visible_face_fraction']:
                                entry['reason'] = 'face_occluded'
                    if entry['reason'] is None:
                        accepted.append(entry)
                if prepared_supported:
                    with closing(stage.ordered(align, image, accepted, extra=3*112*112*4),
                                 name='close face alignment preparation') as aligned:
                        accepted = list(aligned)
                    step = settings['recognizer_batch_size']
                    for offset in range(0, len(accepted), step):
                        chunk = accepted[offset:offset+step]
                        vectors = embed_prepared([e['tensor'] for e in chunk])
                        if len(vectors) != len(chunk):
                            raise ValueError('Face recognizer returned an incompatible output batch.')
                        for entry, value in zip(chunk, vectors):
                            entry['vector'] = normalized_embedding(value, backend.dimensions).astype('<f4').tobytes()
                else:
                    for entry in accepted:
                        entry['vector'] = normalized_embedding(backend.embed(image, entry['face']), backend.dimensions).astype('<f4').tobytes()
                with closing(stage.ordered(encode, image, batch, extra=config.data['storage']['max_crop_bytes']),
                             name='close scan encoding') as encoded:
                    for entry in encoded:
                        yield (entry['index'], entry['face'], entry['quality'], entry['reason'], entry['vector'], entry['crop'])
    finally:
        if owned_stage:
            cleanup([('stop recognition pixel workers', stage.close)])
