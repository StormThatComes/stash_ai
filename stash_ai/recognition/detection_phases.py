"""Opt-in CPU snapshots of one cue's actual inference stages, for the desktop tuner."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import sys


_CURRENT = ContextVar('detection_phases', default=None)
MAX_BYTES = 64 * 1024**2
MAX_PHASES = 256


@dataclass(frozen=True)
class Person:
    label: str
    bounds: tuple
    points: tuple = ()
    mask: object = None  # (shape, packed bytes), shared between immutable snapshots
    positive: tuple = ()
    negative: tuple = ()
    note: str = ''
    rejected: bool = False
    scores: tuple = ()
    probabilities: tuple = ()
    score_kind: str = 'confidence'
    probability_kind: str = 'unavailable'
    raw_points: tuple = ()
    states: tuple = ()
    point_reasons: tuple = ()


@dataclass(frozen=True)
class Phase:
    title: str
    people: tuple


class DetectionResult(tuple):
    """Keep the existing (image, JSON record) API; history is never serialized."""
    def __new__(cls, image, record, phases):
        value = super().__new__(cls, (image, record))
        value.phases = phases
        return value


class History:
    def __init__(self, image, backend=None):
        self.image = image
        self.backend = backend
        self.phases = []
        self.people = ()
        self.bytes = self.omitted = 0
        self.save('Original cue', ())

    def save(self, title, people):
        # Count even shared tuples/masks again: a conservative bound also covers
        # unusually large sets of negative prompts, not just typical COCO17 rows.
        def size(value):
            return sys.getsizeof(value) + (sum(size(v) for v in value) if isinstance(value, tuple) else 0)
        charge = 256 + size(title) + sum(512 + sum(size(v) for v in vars(p).values()) for p in people)
        if self.omitted or len(self.phases) >= MAX_PHASES or self.bytes + charge > MAX_BYTES:
            self.omitted += 1
            return
        self.people = tuple(people)
        self.phases.append(Phase(title, self.people))
        self.bytes += charge

    def render(self, source, index):
        from .detection_phase_render import render
        return render(source, self.phases[index])


@contextmanager
def capture(image, *, enabled=True, backend=None):
    history = History(image, backend) if enabled else None
    token = _CURRENT.set(history)
    try:
        yield history
    finally:
        _CURRENT.reset(token)
        if history is not None:
            history.image = None  # No inference image or GPU owner survives the run.


def active(image):
    history = _CURRENT.get()
    return history if history is not None and history.image is image else None


def _xy(points):
    return tuple(tuple(float(v) for v in xy) for xy in points)


def _details(person, record):
    note = record.get('reason') or record.get('status', '')
    if note == 'rejected' and not record.get('reason'):
        note = 'validated candidate'
    if 'mask_score' in record:
        note += f" | mask score={record['mask_score']:.3f}"
    if 'mask_variant' in record:
        note += f" | candidate={record['mask_variant']}"
    if 'mask_instance' in record:
        note += f" | instance={record['mask_instance']}"
    for field in ('component_edits', 'bridge_cuts'):
        if record.get(field):
            note += f" | {field}: {len(record[field])}"
    for field in ('restored_first_pass_joints', 'withheld_rival_joints', 'mask_ownership_withheld'):
        if record.get(field):
            note += f" | {field}: {record[field]}"
    return replace(person, positive=_xy(record.get('prompt_points', ())),
                   negative=_xy(record.get('negative_points', ())), note=note,
                   rejected=bool(record.get('mask_conflict') or
                                 record.get('reason') and not record.get('overlap_mask')))


def boxes(image, bounds):
    history = active(image)
    if history is not None:
        history.save('Person boxes (after detector NMS)',
                     [Person(f'D{i}', tuple(b)) for i, b in enumerate(bounds)])


def _quality_mask(image, person, record):
    if not record.get('overlap_mask') or record.get('mask_conflict'):
        return person
    import numpy as np
    from .person_geometry import unpack_mask
    left, top, right, bottom = (int(v) for v in person.bounds)
    pixels = np.zeros((image.height, image.width), dtype=bool)
    pixels[top:bottom, left:right] = unpack_mask(record['overlap_mask'], right-left, bottom-top)
    return replace(person, mask=(pixels.shape, np.packbits(pixels, axis=None).tobytes()))


def pose(image, title, bounds, prediction, *, records=None, indices=None, labels=None, reset=False):
    history = active(image)
    if history is None:
        return
    if history.omitted:
        history.omitted += 1
        return
    import numpy as np
    from .detection_point_details import score_kind, probability_kind
    points, scores = prediction[:2]
    probabilities = prediction[2] if len(prediction) > 2 else {}
    rows = [] if reset else list(history.people)
    slots = list(range(len(bounds))) if indices is None else indices
    for offset, (i, box) in enumerate(zip(slots, bounds)):
        person = rows[i] if i < len(rows) else Person(f'P{i}', tuple(box))
        xy = np.where((scores[offset] > 0)[:, None], points[offset], -1)
        person = replace(person, bounds=tuple(box), points=_xy(xy),
                         label=labels[offset] if labels is not None else person.label,
                         scores=tuple(float(v) for v in scores[offset]),
                         probabilities=tuple((key, tuple(float(v) for v in probabilities[key][offset]))
                                             for key in ('presence', 'visibility', 'flip_agreement') if key in probabilities),
                         score_kind=score_kind(history.backend),
                         probability_kind=probability_kind(history.backend, probabilities),
                         raw_points=_xy(points[offset]), states=(), point_reasons=())
        if records is not None:
            person = _details(person, records[offset])
            person = _quality_mask(image, person, records[offset])
        if i < len(rows):
            rows[i] = person
        else:
            rows.append(person)
    history.save(title, rows)


def notes(image, title, records):
    history = active(image)
    if history is not None:
        history.save(title, [_details(p, r) for p, r in zip(history.people, records)])


def masked_pose(image, jobs, prediction):
    if active(image) is not None:
        pose(image, 'Masked pose pass (before acceptance / joint fusion)',
             [box for _, box, _, _ in jobs], prediction, indices=[i for i, _, _, _ in jobs])


def refined_pose(image, title, prediction, records):
    if active(image) is not None:
        pose(image, title, [r['bounds'] for r in records], prediction, records=records)


def output_people(image, people, *, prediction=None, settings=None):
    history = active(image)
    if history is not None:
        from .detection_point_details import output_person
        rows = [output_person(p, i, raw=prediction[0][i] if prediction is not None else None,
                              settings=settings) for i, p in enumerate(people)]
        # Reuse only the corresponding mask, never another candidate's points/scores.
        for i, row in enumerate(rows):
            if i < len(history.people):
                old = history.people[i]
                if old.bounds == row.bounds and old.raw_points == row.raw_points:
                    row = replace(row, mask=old.mask, label=old.label)
            rows[i] = _quality_mask(image, row, people[i].get('pose_refinement', {}))
        history.save('After output keypoint filters', rows)


def mask(image, title, index, pixels, record, *, positive=None, negative=None, rejected=None):
    history = active(image)
    if history is None:
        return
    if history.omitted:
        history.omitted += 1
        return
    import numpy as np
    rows = list(history.people)
    if index >= len(rows):
        return
    person = _details(rows[index], record)
    packed = None if pixels is None else (pixels.shape, np.packbits(pixels, axis=None).tobytes())
    person = replace(person, mask=packed)
    if positive is not None:
        person = replace(person, positive=_xy(positive), negative=_xy(negative))
    if rejected is not None:
        person = replace(person, rejected=rejected)
    rows[index] = person
    history.save(f'{person.label} | {title}', rows)
