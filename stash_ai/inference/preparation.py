"""Image preprocessing and bounded, ordered CPU batch preparation."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING

from stash_ai.inference import sprites

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def prepare_image(app, image: PILImage, target_size: int) -> Any:
    import torchvision.transforms.functional as TVF
    from PIL import Image

    side = max(image.size)
    with Image.new("RGB", (side, side), (255, 255, 255)) as padded:
        padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        with padded.resize((target_size, target_size), Image.Resampling.BICUBIC) as resized:
            tensor = TVF.pil_to_tensor(resized).float().div_(255.0)
    return TVF.normalize(tensor, mean=app.JOYTAG_CLIP_MEAN, std=app.JOYTAG_CLIP_STD)


@dataclass(frozen=True)
class PreparedBatch:
    """A complete backend input assembled on CPU, before inference starts."""
    inputs: Any
    count: int

    def __len__(self):
        return self.count


def prepared_batches(app, tagger, stash, paths, cues, prefetch=0, workers=1, *,
                     report=None, collate=False, cancellation=None,
                     transform_executor=None, frame_lookahead=False, buffer_usage=None):
    """Bounded ordered batches; one producer owns media and worker image copies."""
    if type(prefetch) is not int or not 0 <= prefetch <= 128:
        raise ValueError('prefetch must be an integer in 0..128.')
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('prepare workers must be an integer in 1..16.')
    iterator = sprites.iter_sprite_frames(stash, paths, cues)
    from stash_ai.runtime.cancellation import Cancellation, use
    cancellation = cancellation if cancellation is not None else Cancellation()
    work = report.work() if report is not None else ''
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prepare-frames") if prefetch else None
    transforms = transform_executor
    if transforms is None and workers > 1:
        transforms = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='transform-frames')
    frames = deque()
    current_future = None
    exhausted = False
    # With prefetch, CPU parallelism is independent of the GPU batch size.
    # This is a bounded frame window in addition to the queued ready batches.
    window = max(tagger.batch_size, workers) if prefetch or frame_lookahead else tagger.batch_size

    def measure(name, count=1):
        return (report.measure(name, model=tagger.settings.source.key, work=work,
                               units=count, unit='frames') if report is not None else nullcontext())

    def transform(image: PILImage):
        try:
            with use(cancellation), measure('image_prepare'):
                return tagger.prepare(image)
        finally:
            image.close()

    def fill_frames():
        nonlocal exhausted
        while not exhausted and len(frames) < window:
            cancellation.check()
            try:
                with measure('sprite_frame_read', 0):
                    frame = next(iterator, None)
                if frame is None:
                    exhausted = True
                    break
                cue, image = frame
                owned = image.copy()
                try:
                    future = transforms.submit(transform, owned)
                except BaseException:
                    owned.close()
                    raise
                frames.append((cue, future))
            except BaseException as error:
                # A lookahead failure belongs after the already read frames.
                # Preserve their complete batches for inference/checkpointing.
                failed = Future()
                failed.set_exception(error)
                frames.append((None, failed))
                exhausted = True

    def prepare_next():
        with use(cancellation):
            return _prepare_next()

    def _prepare_next():
        nonlocal current_future
        prepared, batch_cues = [], []
        if transforms:
            fill_frames()
        for _ in range(tagger.batch_size):
            cancellation.check()
            if transforms:
                if not frames:
                    break
                cue, future = frames.popleft()
                current_future = future
                prepared.append(future.result())
                current_future = None
                if prefetch or frame_lookahead:
                    fill_frames()
            else:
                with measure('sprite_frame_read', 0):
                    frame = next(iterator, None)
                if frame is None:
                    break
                cue, image = frame
                with measure('image_prepare'):
                    prepared.append(tagger.prepare(image))
            batch_cues.append(cue)
        cancellation.check()
        if prepared and collate and hasattr(tagger, 'collate_batch'):
            with measure('batch_collate', len(prepared)):
                prepared = PreparedBatch(tagger.collate_batch(prepared), len(prepared))
        return prepared, batch_cues

    def submit_next():
        future = executor.submit(prepare_next)
        if buffer_usage is not None:
            buffer_usage.submitted(future)
        return future

    try:
        pending = deque(submit_next() for _ in range(prefetch)) if executor else deque()
        while True:
            with measure('inference_input_wait', 0):
                if executor:
                    future = pending.popleft()
                    if buffer_usage is not None:
                        buffer_usage.taken(future)
                    prepared, batch_cues = future.result()
                else:
                    prepared, batch_cues = prepare_next()
            if not prepared:
                break
            if executor:
                pending.append(submit_next())
            yield prepared, batch_cues
    finally:
        cancellation.cancel()
        try:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            if buffer_usage is not None:
                buffer_usage.clear()
        if transforms is not None and transform_executor is None:
            transforms.shutdown(wait=True)
        elif transforms is not None:
            # Borrowed workers outlive the scene. Let every owned image reach
            # transform()'s finally before closing media/the backend.
            wait([future for _, future in frames] + ([current_future] if current_future else []))
        iterator.close()
