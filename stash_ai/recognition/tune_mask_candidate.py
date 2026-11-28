"""CPU-only inputs to replay shared cleanup for one observed decoder candidate."""
from dataclasses import dataclass
import math

import numpy as np

from .pose_refinement_config import LEGACY_CLEANUP

LOCAL_KEYS = frozenset(LEGACY_CLEANUP) | {
    'min_mask_score', 'min_prompt_coverage', 'max_foreign_coverage', 'max_box_area_ratio', 'max_side_expansion'}
LOCAL_PATHS = frozenset('pose.mask.' + key for key in LOCAL_KEYS)


@dataclass(frozen=True)
class MaskSample:
    title: str
    shape: tuple
    bits: bytes
    score: float
    box: tuple
    prompts: np.ndarray
    own: np.ndarray
    foreign: np.ndarray
    limbs: np.ndarray
    attempt_key: tuple | None = None


def record(history, raw, score, title, box, prompts, own, foreign, limbs, attempt_key):
    arrays = [np.asarray(a).reshape(shape) for a, shape in
              ((prompts, (-1, 2)), (own, (-1, 2)), (foreign, (-1, 2)), (limbs, (-1, 2, 2)))]
    from .detection_phases import MAX_BYTES
    charge = (raw.size + 7)//8 + sum(a.nbytes for a in arrays) + 1536
    if len(history.samples) >= 256 or history.bytes + charge > MAX_BYTES:
        history.samples_omitted += 1
        return
    history.bytes += charge
    history.samples.append(MaskSample(title, raw.shape, np.packbits(raw, axis=None).tobytes(), float(score),
                                      tuple(box), *(a.copy() for a in arrays), attempt_key))


def preview(source, sample, settings, view):
    from .mask_candidates import check_mask
    from .detection_masks import paint_mask

    raw = np.unpackbits(np.frombuffer(sample.bits, np.uint8), count=math.prod(sample.shape)).reshape(
        sample.shape).view(bool)
    stages, details = {}, {}
    _, _, failure = check_mask(raw, sample.score, sample.prompts, sample.foreign, sample.box, settings,
                               source.size, own=sample.own, limbs=sample.limbs, details=details,
                               observe=stages.__setitem__)
    cleaned = stages.get('cleaned')
    canvas = source.convert('RGB')
    if view == 'Raw':
        paint_mask(canvas, raw, (48, 156, 255), opacity=.4)
    elif cleaned is not None:
        paint_mask(canvas, cleaned, (59, 191, 126), opacity=.35)
        if view == 'Removed pixels':
            paint_mask(canvas, raw & ~cleaned, (255, 78, 64), opacity=.8)
    removed = '' if cleaned is None else f' | {int(raw.sum()-cleaned.sum())} pixels removed'
    status = failure or 'passed local mask checks'
    return canvas, f'{sample.title} | {status}{removed}', details, cleaned


def inspect(details, cleaned, x, y):
    labels = details.get('labels')
    if labels is None:
        return 'Cleanup did not run for this candidate.'
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]) or not (label := labels[y, x]):
        return 'Outside the raw mask.'
    area = details['areas'][label]
    gap = details.get('distances')
    gap_text = f" | nearest supported island {gap[labels == label].min():.1f} px" if gap is not None else ''
    if details['keep'][label]:
        state = 'kept by limb support' if label in details['rescued'] else 'kept'
        if not cleaned[y, x]:
            state = 'cut at bridge'
    elif label in details['foreign_only']:
        state = 'foreign island removed'
    elif area < details['minimum'] and label not in details['rescued']:
        state = 'below size floor'
    else:
        state = f"outside supported gap ({details['gap_limit']:.1f} px)"
    return (f"{state} | island {area} px, size floor {details['minimum']} px | "
            f"own points {details['own'][label]}, foreign {details['foreign'][label]}{gap_text}")
