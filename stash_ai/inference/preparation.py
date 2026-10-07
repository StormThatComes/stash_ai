"""Image preprocessing and bounded, ordered CPU batch preparation."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
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


def prepared_batches(app, tagger, stash, paths, cues, prefetch=False, workers=1, *, report=None):
    """Current batch plus N queued batches; one producer owns media/PIL."""
    if not isinstance(prefetch, int) or not 0 <= prefetch <= 16:
        raise ValueError('prefetch must be an integer in 0..16.')
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('prepare workers must be an integer in 1..16.')
    iterator = sprites.iter_sprite_frames(stash, paths, cues)
    work = report.work() if report is not None else ''
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prepare-frames") if prefetch else None
    transforms = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='transform-frames') if workers > 1 else None
    def transform(image: PILImage):
        try:
            from contextlib import nullcontext
            timer = (report.measure('image_prepare', model=tagger.settings.source.key, work=work, units=1, unit='frames')
                     if report is not None else nullcontext())
            with timer:
                return tagger.prepare(image)
        finally:
            image.close()
    def prepare_next():
        prepared, batch_cues = [], []
        for _ in range(tagger.batch_size):
            try:
                cue, image = next(iterator)
            except StopIteration:
                break
            if transforms:
                # The source iterator closes its frame on the next step. Each
                # worker owns a copy, and closes it even if preparation fails.
                owned = image.copy()
                prepared.append(transforms.submit(transform, owned))
            else:
                from contextlib import nullcontext
                timer = (report.measure('image_prepare', model=tagger.settings.source.key, work=work, units=1, unit='frames')
                         if report is not None else nullcontext())
                with timer:
                    prepared.append(tagger.prepare(image))
            batch_cues.append(cue)
        if transforms:
            prepared = [future.result() for future in prepared]
        return prepared, batch_cues
    try:
        pending = deque(executor.submit(prepare_next) for _ in range(prefetch)) if executor else deque()
        while True:
            prepared, batch_cues = pending.popleft().result() if executor else prepare_next()
            if not prepared:
                break
            if executor:
                pending.append(executor.submit(prepare_next))
            yield prepared, batch_cues
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        if transforms:
            transforms.shutdown(wait=True)
        iterator.close()
