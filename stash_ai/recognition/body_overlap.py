"""Directional foreground overlap with a conservative per-pair box fallback."""


def foreground_mask(person):
    """Return a validated crop-local SAM mask and its foreground area."""
    import numpy as np
    from .person_geometry import unpack_mask
    details = person.get('pose_refinement', {})
    packed = details.get('overlap_mask')
    admitted = details.get('mask_valid', details.get('status') in ('refined', 'masked')
                           or details.get('reason') == 'second_pass_core_drift')
    if (not packed or not admitted or details.get('mask_conflict')
            or details.get('bounds') != list(person['bounds'])):
        return None
    left, top, right, bottom = person['bounds']
    mask = unpack_mask(packed, right-left, bottom-top)
    area = int(np.count_nonzero(mask))
    return (mask, area) if area else None


def admission_overlap(quality):
    """Read the gate measurement, including older rows with a single overlap value."""
    return quality.get('admission_overlap', quality.get('overlap'))


def measure(person, people, *, report_above=None):
    """Use intersection / this person's area, never IoU or summed overlaps.

    Masks are crop-local. Align only the shared rectangle in
    source coordinates; never allocate a full-frame mask for each comparison.
    A masked target skips maskless peers for admission. Keep the box fallback
    in the crowding measurement consumed by tracking and identity ranking.
    """
    import numpy as np
    from .mask_usage import uses
    use_masks = uses(person, 'overlap')
    left, top, right, bottom = person['bounds']
    area = (right-left)*(bottom-top)
    overlap = box_overlap = 0.
    basis = set()
    admission_methods = set()
    own = None
    decoded = False
    maximum = None
    skipped_peers = []
    winning_pair = {}
    for index, other in enumerate(people):
        if other is person:
            continue
        a, b, c, d = other['bounds']
        x0, y0, x1, y1 = max(left,a), max(top,b), min(right,c), min(bottom,d)
        if x0 >= x1 or y0 >= y1:
            continue
        box_fraction = (x1-x0)*(y1-y0)/area
        box_overlap = max(box_overlap, box_fraction)
        if not decoded:
            own = foreground_mask(person) if use_masks else None
            decoded = True
        peer = foreground_mask(other) if use_masks and uses(other, 'overlap') else None
        peer_pixels = peer[0][y0-b:y1-b, x0-a:x1-a] if peer is not None else None
        if own is not None and peer_pixels is not None:
            own_pixels = own[0][y0-top:y1-top, x0-left:x1-left]
            fraction = np.count_nonzero(own_pixels & peer_pixels) / own[1]
            method = 'mask'
            basis.add('mask')
            overlap = max(overlap, fraction)
        else:
            # Preserve box crowding for ranking and tracking independently of admission.
            basis.add('box')
            overlap = max(overlap, box_fraction)
            if own is not None:
                skipped_peers.append(index)
                continue
            if peer_pixels is not None:
                fraction = np.count_nonzero(peer_pixels) / area
                method = 'peer_mask_in_box'
            else:
                fraction = box_fraction
                method = 'box'
        admission_methods.add(method)
        if maximum is None or fraction > maximum:
            maximum = float(fraction)
            if report_above is not None and fraction > report_above:
                winning_pair = dict(overlap_peer_index=index, overlap_pair_basis=method,
                                    overlap_pair_box_fraction=box_fraction)
    result = dict(overlap=float(overlap), box_overlap=float(box_overlap),
                  overlap_basis='mixed' if len(basis)>1 else next(iter(basis), 'none'), **winning_pair)
    if skipped_peers or 'peer_mask_in_box' in admission_methods:
        result['admission_overlap'] = maximum
        result['admission_overlap_basis'] = ('mixed' if len(admission_methods) > 1
                                             else next(iter(admission_methods), 'skipped'))
    if skipped_peers:
        result['overlap_skipped_peer_indices'] = skipped_peers
    return result
