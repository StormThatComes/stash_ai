"""One pose -> instance mask -> masked pose pass, with bounded box recovery."""
from contextlib import nullcontext
import math

import numpy as np

from .profiling import pipeline_stage

CORE = (0, 1, 2, 3, 4, 5, 6, 11, 12)  # Head/torso, not the suspect wrists/ankles.


def reliable_core(points, scores, probabilities, box, settings):
    indices = np.asarray(CORE)
    xy = points[indices]
    left, top, right, bottom = box
    valid = ((xy[:, 0] >= left) & (xy[:, 0] < right) &
             (xy[:, 1] >= top) & (xy[:, 1] < bottom) &
             (scores[indices] > settings['min_keypoint_score']) &
             (probabilities['presence'][indices] > settings['min_presence']) &
             (probabilities['visibility'][indices] > settings['min_visibility']))
    indices = indices[valid]
    # Stable ordering makes tied predictions independent of batching.
    return indices[np.argsort(-probabilities['visibility'][indices], kind='stable')]


def overlaps(box, other, threshold):
    # Intersection / smaller box detects containment as well as similar boxes.
    intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
        0, min(box[3], other[3]) - max(box[1], other[1]))
    area = min((box[2]-box[0])*(box[3]-box[1]), (other[2]-other[0])*(other[3]-other[1]))
    return intersection > 0 and intersection / area >= threshold


def inside(mask, points):
    if not len(points):
        return np.empty(0, dtype=bool)
    xy = np.floor(points).astype(np.int64)
    valid = ((xy >= 0).all(axis=1) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0]))
    result = np.zeros(len(points), dtype=bool)
    result[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return result


def check_mask(mask, score, prompts, foreign, box, settings, size, *, observe=None):
    """Return an admitted mask/box or an explicit, auditable rejection reason."""
    import cv2
    width, height = size
    if mask.dtype != np.bool_ or mask.shape != (height, width):
        raise ValueError('SAM-pose2seg mask must match the full source frame.')
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('SAM-pose2seg returned an invalid mask score.')
    if observe is not None:
        observe('raw', mask)
    if score < settings['min_mask_score']:
        return None, None, 'low_mask_score'
    if not mask.any():
        return None, None, 'empty_mask'
    # Remove tiny disconnected speckles without throwing away a disconnected
    # but substantial hand/foot. Never force a single connected component.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(4, math.ceil(int(mask.sum()) * .001))
    keep[0] = False
    mask = keep[labels]
    if observe is not None:
        observe('cleaned', mask)
    if not mask.any():
        return None, None, 'empty_mask'
    if inside(mask, prompts).mean() < settings['min_prompt_coverage']:
        return None, None, 'missing_prompt_support'
    if len(foreign) and inside(mask, foreign).mean() > settings['max_foreign_coverage']:
        return None, None, 'foreign_core_in_mask'
    y, x = np.nonzero(mask)
    mask_box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    union = [min(box[0], mask_box[0]), min(box[1], mask_box[1]),
             max(box[2], mask_box[2]), max(box[3], mask_box[3])]
    w, h = box[2] - box[0], box[3] - box[1]
    ratio = (union[2]-union[0]) * (union[3]-union[1]) / (w*h)
    expansion = max((box[0]-union[0])/w, (box[1]-union[1])/h,
                    (union[2]-box[2])/w, (union[3]-box[3])/h)
    if ratio > settings['max_box_area_ratio'] or expansion > settings['max_side_expansion']:
        return None, None, 'excessive_expansion'
    # Expansion only: no shrink around a visible torso when legs are occluded.
    return mask, union if settings['expand_boxes'] else list(box), None


class RefinedPMPose:
    def __init__(self, pose, segmenter, settings):
        self.pose, self.segmenter, self.settings = pose, segmenter, settings

    def close(self):
        from .resident_models import release
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('close raw pose',lambda:release(self.pose)),('close segmenter',lambda:release(self.segmenter))])

    def predict(self, image, bounds):
        return self.predict_many([(image, bounds)])[0]

    def predict_many(self, frames):
        frames = list(frames)
        predictions = self.pose.predict_many(frames)
        if len(predictions) != len(frames):
            raise ValueError('PMPose returned an invalid refinement frame count.')
        return [self.refine(image, bounds, prediction)
                for (image, bounds), prediction in zip(frames, predictions)]

    @pipeline_stage('pmpose_mask_refinement')
    def refine(self, image, bounds, prediction):
        points, scores, probabilities = prediction
        settings = self.settings
        result = (points.copy(), scores.copy(), {k: v.copy() for k, v in probabilities.items()})
        records = [dict(status='skipped', reason='no_reliable_prompts',
                        detector_bounds=list(box), bounds=list(box)) for box in bounds]
        cores = [reliable_core(points[i], scores[i], {k: v[i] for k, v in probabilities.items()}, box, settings)
                 for i, box in enumerate(bounds)]
        jobs = []
        for i, box in enumerate(bounds):
            if settings['mode'] == 'overlap' and not any(
                    overlaps(box, other, settings['overlap_threshold']) for j, other in enumerate(bounds) if i != j):
                records[i]['reason'] = 'no_overlap'
                continue
            foreign = np.concatenate([points[j, indices] for j, indices in enumerate(cores) if j != i], axis=0) if len(bounds)>1 else np.empty((0, 2))
            indices = cores[i]
            if len(foreign) and len(indices):
                # Coincident head/torso estimates do not establish ownership.
                distance = np.linalg.norm(points[i, indices, None] - foreign[None], axis=2).min(axis=1)
                indices = indices[distance > max(2., .02 * math.hypot(box[2]-box[0], box[3]-box[1]))]
            indices = indices[:settings['max_prompt_points']]
            if len(indices) >= settings['min_prompt_points']:
                jobs.append((i, indices, foreign))
        # No reliable prompts / no eligible boxes means no SAM image inference.
        with self.segmenter.frame(image) if jobs else nullcontext():
            pending = []
            for i, indices, foreign in jobs:
                prompts = points[i, indices]
                mask, score = self.segmenter.predict(prompts)
                record = records[i]
                record.update(status='rejected', mask_score=score, prompt_indices=indices.tolist(),
                              prompt_points=prompts.tolist())
                from .pose_mask_capture import mask_observer
                mask, box, reason = check_mask(mask, score, prompts, foreign, bounds[i], settings, image.size,
                                              observe=mask_observer(image, i))
                record['reason'] = reason
                if reason:
                    continue
                record['mask_pixels'] = int(mask.sum())
                pending.append((i, box, mask, indices))
                if len(pending) >= self.pose.batch_size:
                    self._second_pass(image, pending, result, records, points)
                    pending.clear()
            if pending:
                self._second_pass(image, pending, result, records, points)
        return (*result, records)

    @pipeline_stage('pmpose_masked_second_pass')
    def _second_pass(self, image, pending, result, records, original):
        refined = self.pose.predict(image, [box for _, box, _, _ in pending],
                                    masks=[mask for _, _, mask, _ in pending])
        for offset, (i, box, mask, indices) in enumerate(pending):
            new_points, new_scores, probabilities = refined
            keep = ((new_scores[offset, indices] > self.settings['min_keypoint_score']) &
                    (probabilities['presence'][offset, indices] > self.settings['min_presence']) &
                    inside(mask, new_points[offset, indices]))
            old = records[i]['detector_bounds']
            tolerance = max(3., .15 * math.hypot(old[2]-old[0], old[3]-old[1]))
            keep &= np.linalg.norm(new_points[offset, indices] - original[i, indices], axis=1) <= tolerance
            if keep.sum() < self.settings['min_prompt_points']:
                records[i]['reason'] = 'second_pass_core_drift'
                continue
            result[0][i], result[1][i] = new_points[offset], new_scores[offset]
            for key in result[2]:
                result[2][key][i] = probabilities[key][offset]
            records[i].update(status='refined', reason=None, bounds=box,
                              box_expanded=box != records[i]['detector_bounds'])
