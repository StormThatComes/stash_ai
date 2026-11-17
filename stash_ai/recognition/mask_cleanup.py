"""Geometry and pose evidence for cleaning a full-resolution SAM person mask."""
import math

import numpy as np


def inside(mask, points):
    if not len(points):
        return np.empty(0, dtype=bool)
    xy = np.floor(points).astype(np.int64)
    valid = ((xy >= 0).all(axis=1) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0]))
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return result


def _labels_at(labels, points):
    xy = np.asarray(points).reshape(-1, 2)
    xy = np.floor(xy[inside(labels, xy)]).astype(int)
    return labels[xy[:, 1], xy[:, 0]]


def _representative(labels, stats, label):
    # A component always has a pixel on the top edge of its bounding box.
    left, top, width = map(int, stats[label, :3])
    column = np.argmax(labels[top, left:left + width] == label)
    return left + int(column), top


def _owned_islands(mask, labels, keep, prompts, own, foreign, limbs, max_gap, settings):
    import cv2
    support = np.vstack((np.asarray(prompts).reshape(-1, 2), np.asarray(own).reshape(-1, 2)))
    ours = _labels_at(labels, support)
    theirs = _labels_at(labels, foreign)
    path_floor = settings['min_limb_path_coverage']
    foreign_gap = settings['min_limb_foreign_distance_pixels']
    for parent, tip in limbs:
        segment = np.linspace(parent, tip, 9)
        island = _labels_at(labels, [tip])
        if (len(island) and island[0] not in theirs
                and inside(mask, segment[:-1]).mean() >= path_floor
                and (not len(foreign) or np.linalg.norm(foreign - tip, axis=1).min() > foreign_gap)):
            keep[island[0]] = True
    if len(own) and len(ours) and len(theirs):
        foreign_only, counts = np.unique(theirs[~np.isin(theirs, ours)], return_counts=True)
        keep[foreign_only[counts >= settings['min_foreign_island_points']]] = False

    supported = np.zeros_like(keep)
    supported[ours] = True
    supported &= keep
    if max_gap > 0 and supported.any() and np.any(keep & ~supported):
        # Edge-to-edge distance from all surviving, pose-supported components.
        distances = cv2.distanceTransform((~supported[labels]).astype(np.uint8), cv2.DIST_L2, 5)
        nearby = np.zeros_like(keep)
        nearby[labels[distances <= max_gap]] = True
        keep &= nearby


def _bridge_regions(island, radius, max_fraction):
    """Return removable branches. Opening identifies lobes; it never becomes the output mask."""
    import cv2
    width = 2 * radius + 1
    kernel = np.ones((width, width), np.uint8)
    opened = cv2.morphologyEx(island, cv2.MORPH_OPEN, kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Diagonal-only contact is also a narrow bridge between substantial lobes.
    count, lobes, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=4)
    if count < 3:
        return None
    areas = stats[:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas[1:])) + 1
    cap = max_fraction * areas[largest]
    small = areas <= cap
    small[[0, largest]] = False
    if not small.any():
        return None

    # Keep the original outline. Only remove a residual region containing an
    # actual lobe, together with its narrow attachment to a single large core.
    cores = (lobes != 0) & ~small[lobes]
    residual = (island.astype(bool) & ~cores).astype(np.uint8)
    count, regions, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    candidates = np.zeros(count, bool)
    candidates[regions[small[lobes]]] = True
    candidates &= stats[:, cv2.CC_STAT_AREA] <= cap
    # A region joining two retained cores may be a real articulated limb.
    if np.count_nonzero(~small[1:]) > 1 and candidates.any():
        for region in np.flatnonzero(candidates):
            x, y, w, h = stats[region, :4]
            roi = np.s_[max(0, y-1):y+h+1, max(0, x-1):x+w+1]
            border = cv2.dilate((regions[roi] == region).astype(np.uint8), np.ones((3, 3), np.uint8))
            adjacent = lobes[roi][cores[roi] & (border != 0)]
            if len(np.unique(adjacent)) > 1:
                candidates[region] = False
    return regions, stats, candidates


def _prune_bridges(mask, labels, stats, keep, support, radius, fraction, cuts, replay):
    # Analyze component crops. Skip texture-sized islands which cannot contain
    # two opened lobes, and only revisit recorded cuts during bitmap backfill.
    if replay is None:
        targets = np.flatnonzero(keep & (stats[:, 4] >= 2 * (2 * radius + 1) ** 2))
    else:
        targets = np.unique(_labels_at(labels, replay))
    for label in targets:
        x, y, w, h = stats[label, :4]
        roi = np.s_[y:y+h, x:x+w]
        parts = _bridge_regions((labels[roi] == label).astype(np.uint8), radius, fraction)
        if parts is None:
            continue
        regions, bounds, remove = parts
        if replay is None:
            remove[_labels_at(regions, support - (x, y))] = False
        else:
            selected = _labels_at(regions, np.asarray(replay) - (x, y))
            remove &= np.isin(np.arange(len(remove)), selected)
        if remove.any():
            mask[roi] &= ~remove[regions]
            if cuts is not None:
                for region in np.flatnonzero(remove):
                    px, py = _representative(regions, bounds, region)
                    cuts.append((int(x) + px, int(y) + py))


def clean_mask(mask, box, settings, *, prompts, own, foreign, limbs, edits, replay, cuts, replay_cuts):
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    minimum = max(settings['min_island_pixels'], math.ceil(int(areas[1:].sum()) * settings['min_island_fraction']))
    keep = areas >= minimum
    keep[0] = False
    baseline = keep.copy() if edits is not None else None
    if replay is not None:
        for x, y, decision in replay:
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and labels[y, x]:
                keep[labels[y, x]] = decision
    elif count > 2:
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        _owned_islands(mask, labels, keep, prompts, own, foreign, limbs,
                       settings['max_island_gap_fraction'] * diagonal, settings)
    if edits is not None:
        for label in np.flatnonzero(keep != baseline):
            edits.append((*_representative(labels, stats, label), bool(keep[label])))

    cleaned = keep[labels]
    radius = settings['bridge_radius_pixels']
    fraction = settings['max_bridge_lobe_fraction']
    if radius and fraction and (replay_cuts is None or len(replay_cuts)):
        support = np.vstack((np.asarray(prompts).reshape(-1, 2), np.asarray(own).reshape(-1, 2)))
        _prune_bridges(cleaned, labels, stats, keep, support, radius, fraction, cuts, replay_cuts)
    return cleaned
