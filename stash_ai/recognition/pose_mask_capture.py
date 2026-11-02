"""Bounded, frame-local mask diagnostics; inactive during ordinary recognition."""
from contextlib import contextmanager
from contextvars import ContextVar


MAX_BYTES = 16 * 1024 * 1024
_CURRENT = ContextVar('pose_mask_capture', default=None)


class FrameMasks:
    def __init__(self, image):
        self.image = image
        self.people = {}
        self.bytes = 0

    def save(self, index, stage, mask):
        import numpy as np
        size = (mask.size + 7) // 8
        if self.bytes + size > MAX_BYTES:
            raise ValueError('Detection audit masks exceed 16 MiB; use --mask-view off.')
        # Pack before retaining: full-frame boolean arrays otherwise multiply
        # peak RAM by the number of people and stages. No GPU tensors survive.
        packed = np.packbits(mask, axis=None).tobytes()
        self.people.setdefault(index, {})[stage] = (mask.shape, packed)
        self.bytes += len(packed)

    def mask(self, index, stage):
        import numpy as np
        item = self.people.get(index, {}).get(stage)
        if item is None:
            return None
        shape, packed = item
        return np.unpackbits(np.frombuffer(packed, dtype=np.uint8),
                             count=shape[0] * shape[1]).reshape(shape).astype(bool)


@contextmanager
def capture_masks(image, *, enabled=True):
    capture = FrameMasks(image) if enabled else None
    token = _CURRENT.set(capture)
    try:
        yield capture
    finally:
        _CURRENT.reset(token)
        if capture is not None:
            capture.people.clear()
            capture.bytes = 0
            capture.image = None


def mask_observer(image, index):
    capture = _CURRENT.get()
    if capture is None or capture.image is not image:
        return None
    return lambda stage, mask: capture.save(index, stage, mask)
