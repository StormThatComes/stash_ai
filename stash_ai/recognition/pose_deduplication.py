"""Conservative greedy suppression from the first PMPose pass, before SAM."""
from collections import Counter

import numpy as np

from .profiling import pipeline_stage, count


CORE = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12])
REGIONS = {'head': (0, 1, 2, 3, 4), 'shoulders': (5, 6), 'hips': (11, 12)}
MAX_COMPARISONS = 50_000
MAX_AUDIT_KEPT_PAIRS = 128


def _measurements(bounds, prediction, settings):
    points, scores, probabilities = prediction
    n = len(bounds)
    boxes = np.asarray(bounds, dtype=np.float64).reshape(n, 4)
    if (points.shape != (n, 17, 2) or scores.shape != (n, 17)
            or set(probabilities) != {'presence', 'visibility'}
            or any(v.shape != (n, 17) for v in probabilities.values())
            or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2])
            or not np.isfinite(points).all()
            or any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1))
                   for v in (scores, *probabilities.values()))):
        raise ValueError('PMPose deduplication requires finite, valid boxes and COCO17 predictions.')
    valid = ((points[:, :, 0] >= boxes[:, None, 0]) & (points[:, :, 0] < boxes[:, None, 2]) &
             (points[:, :, 1] >= boxes[:, None, 1]) & (points[:, :, 1] < boxes[:, None, 3]) &
             (scores > settings['min_keypoint_score']) &
             (probabilities['presence'] > settings['min_presence']) &
             (probabilities['visibility'] > settings['min_visibility']))
    valid[:, [7, 8, 9, 10, 13, 14, 15, 16]] = False
    # A fixed denominator rewards reliable coverage, not just a few high scores.
    quality = (scores * probabilities['presence'] * probabilities['visibility'] * valid).sum(axis=1) / len(CORE)
    return boxes, valid, quality


def _compare(a, b, boxes, points, valid, settings):
    lo, hi = np.maximum(boxes[a, :2], boxes[b, :2]), np.minimum(boxes[a, 2:], boxes[b, 2:])
    intersection = float(np.maximum(0, hi-lo).prod())
    areas = (boxes[[a, b], 2:] - boxes[[a, b], :2]).prod(axis=1)
    record = dict(detector_indices=[int(a), int(b)],
                  box_iou=intersection / float(areas.sum()-intersection),
                  box_overlap=intersection / float(areas.min()),
                  shared_indices=[], matched_indices=[], matched_regions=[])
    if not intersection:
        return 'no_box_overlap', record
    shared = np.flatnonzero(valid[a] & valid[b])
    record['shared_indices'] = shared.tolist()
    scale = float(np.linalg.norm(boxes[[a, b], 2:] - boxes[[a, b], :2], axis=1).min())
    distances = np.linalg.norm(points[a, shared] - points[b, shared], axis=1) / scale
    near = distances <= settings['max_point_distance']
    matched = shared[near]
    regions = [name for name, indices in REGIONS.items() if np.isin(matched, indices).any()]
    record.update(matched_indices=matched.tolist(), normalized_distances=distances.tolist(),
                  normalization_pixels=scale, match_fraction=float(near.mean()) if len(near) else 0.,
                  matched_regions=regions, max_normalized_distance=float(distances.max()) if len(distances) else None)
    # A separate reliable head or torso is contrary evidence, even when many
    # other points happen to coincide. Never assume upright anatomy.
    for name, indices in REGIONS.items():
        region = np.isin(shared, indices)
        if region.sum() >= 2 and np.median(distances[region]) > settings['max_point_distance']:
            return 'distinct_heads' if name == 'head' else 'distinct_torsos', record
    if np.any(distances > 2 * settings['max_point_distance']):
        return 'conflicting_core', record
    if len(shared) < settings['min_shared_points']:
        return 'insufficient_shared_core', record
    if len(matched) < settings['min_shared_points'] or near.mean() < settings['min_match_fraction']:
        return 'insufficient_core_agreement', record
    if len(regions) < 2:
        return 'single_region_support', record
    for index in (a, b):
        xy = points[index, matched]
        span = float(np.linalg.norm(xy[:, None] - xy[None], axis=2).max()) / scale
        if span < max(.1, 2 * settings['max_point_distance']):
            return 'collapsed_core_support', record
    return 'duplicate_core', record


@pipeline_stage('pmpose_pose_deduplication')
def select(bounds, prediction, settings, *, audit=False):
    boxes, valid, quality = _measurements(bounds, prediction, settings)
    points = prediction[0]
    order = sorted(range(len(boxes)), key=lambda i: (-quality[i], -int(valid[i].sum()), i))
    kept, suppressed, comparisons, reasons = [], [None] * len(boxes), [], Counter()
    total = omitted = retained_records = 0
    for candidate in order:
        for representative in kept:
            total += 1
            if total > MAX_COMPARISONS:
                raise ValueError('PMPose deduplication comparison budget exceeded; reduce pose.max_people_per_frame.')
            reason, comparison = _compare(representative, candidate, boxes, points, valid, settings)
            reasons[reason] += 1
            duplicate = reason == 'duplicate_core'
            # Always keep suppression evidence; cap overlapping pairs kept as
            # separate people. Non-overlapping pairs have an aggregate count.
            if audit and (duplicate or reason != 'no_box_overlap' and retained_records < MAX_AUDIT_KEPT_PAIRS):
                comparison.update(reason=reason, decision='suppress' if duplicate else 'no_suppression',
                                  representative_index=representative if duplicate else None)
                comparisons.append(comparison)
                retained_records += int(not duplicate)
            elif audit and reason != 'no_box_overlap':
                omitted += 1
            if duplicate:
                suppressed[candidate] = representative
                break
        else:
            kept.append(candidate)
    kept.sort()  # Preserve original detector order and stable output colors.
    report = dict(kept_indices=kept, suppressed_by=suppressed)
    if audit:
        output_indices = {original: index for index, original in enumerate(kept)}
        report['audit'] = dict(
            detections=[dict(detector_index=i, bounds=boxes[i].tolist(), quality_score=float(quality[i]),
                             reliable_core_count=int(valid[i].sum()), person_index=output_indices.get(i),
                             suppressed_by=suppressed[i]) for i in range(len(boxes))],
            kept_indices=kept, comparisons=comparisons, comparison_count=total,
            reason_counts=dict(reasons), omitted_kept_pairs=omitted,
            settings=dict(settings), index_note='D indices are candidates after box NMS/score filtering; '
                                               'P indices are surviving people in this cue/trial.')
    count('pmpose_duplicate_people_removed', len(boxes)-len(kept))
    return report


class DeduplicatedPMPose:
    def __init__(self, pose, settings, refined=None):
        self.pose, self.settings, self.refined = pose, settings, refined

    def predict(self, image, bounds):
        return self.predict_many([(image, bounds)])[0]

    def predict_many(self, frames):
        from .pose_deduplication_audit import active, publish
        frames = list(frames)
        predictions = self.pose.predict_many(frames)
        if len(predictions) != len(frames):
            raise ValueError('PMPose returned an invalid deduplication frame count.')
        results = []
        for (image, bounds), prediction in zip(frames, predictions):
            report = select(bounds, prediction, self.settings, audit=active(image))
            keep = np.asarray(report['kept_indices'], dtype=np.int64)
            points, scores, probabilities = prediction
            selected = points[keep], scores[keep], {k: v[keep] for k, v in probabilities.items()}
            selected_bounds = [bounds[i] for i in keep]
            final = self.refined.refine(image, selected_bounds, selected) if self.refined else (*selected, None)
            publish(image, report.get('audit'))
            results.append((*final, report))
        return results
