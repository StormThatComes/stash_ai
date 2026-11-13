"""Directional foreground overlap with a conservative per-pair box fallback."""
import numpy as np


def foreground_mask(person):
    """Return a validated crop-local SAM mask and its foreground area."""
    from .person_geometry import unpack_mask
    details = person.get('pose_refinement', {})
    packed = details.get('overlap_mask')
    if (not packed or details.get('status') != 'refined' or details.get('mask_conflict')
            or details.get('bounds') != list(person['bounds'])):
        return None
    left, top, right, bottom = person['bounds']
    mask = unpack_mask(packed, right-left, bottom-top)
    area = int(np.count_nonzero(mask))
    return (mask, area) if area else None


def measure(person, people):
    """Use intersection / this person's area, never IoU or summed overlaps.

    Masks are crop-local and transient. Align only the shared rectangle in
    source coordinates; never allocate a full-frame mask for each comparison.
    Missing, rejected or empty masks retain the old box rule for that pair.
    """
    left, top, right, bottom = person['bounds']
    area = (right-left)*(bottom-top)
    overlap = box_overlap = 0.
    basis = set()
    own = None
    decoded = False
    for other in people:
        if other is person:
            continue
        a, b, c, d = other['bounds']
        x0, y0, x1, y1 = max(left,a), max(top,b), min(right,c), min(bottom,d)
        if x0 >= x1 or y0 >= y1:
            continue
        box_fraction = (x1-x0)*(y1-y0)/area
        box_overlap = max(box_overlap, box_fraction)
        if not decoded:
            own = foreground_mask(person)
            decoded = True
        peer = foreground_mask(other) if own is not None else None
        if own is not None and peer is not None:
            shared = np.count_nonzero(own[0][y0-top:y1-top, x0-left:x1-left] &
                                      peer[0][y0-b:y1-b, x0-a:x1-a])
            fraction = shared / own[1]
            basis.add('mask')
        else:
            fraction = box_fraction
            basis.add('box')
        overlap = max(overlap, fraction)
    return dict(overlap=float(overlap), box_overlap=float(box_overlap),
                overlap_basis='mixed' if len(basis)>1 else next(iter(basis), 'none'))
