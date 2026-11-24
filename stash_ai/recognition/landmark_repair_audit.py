"""Opt-in numeric detector retry evidence; never retain source pixels or model tensors."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace


_CURRENT = ContextVar('landmark_repair_audit', default=None)


@contextmanager
def capture():
    reports = []
    token = _CURRENT.set(reports)
    try:
        yield reports
    finally:
        _CURRENT.reset(token)


def geometry(face):
    return dict(box=list(face.box), landmarks=[list(p) for p in face.landmarks], score=face.score)


def begin(images, found, *, rotation, threshold):
    reports = _CURRENT.get()
    if reports is None:
        return None
    from .quality import alignment_measurements
    frames = []
    for index, (pixels, faces) in enumerate(zip(images, found, strict=True)):
        image = SimpleNamespace(width=pixels.shape[1], height=pixels.shape[0])
        rows = [dict(face_index=i, original=geometry(face),
                     original_quality=alignment_measurements(face, image),
                     status='not_requested', attempts=[]) for i, face in enumerate(faces)]
        reports.append(dict(batch_frame_index=index, source_dimensions=[image.width, image.height],
                            rotation_enabled=rotation, alignment_retry_error=threshold, faces=rows))
        frames.append(rows)
    return frames


def finish(frames, found):
    if frames is not None:
        for rows, faces in zip(frames, found, strict=True):
            for row, face in zip(rows, faces, strict=True):
                row['result'] = geometry(face)
