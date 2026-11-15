"""One pose -> instance mask -> masked pose pass, with bounded box recovery."""
import base64
import math

import numpy as np

from . import prompt_recovery
from .mask_candidates import check_mask, inside, select_masks, spread_indices
from .profiling import pipeline_stage, pipeline_timing

CORE = (0, 1, 2, 3, 4, 5, 6, 11, 12)  # Head/torso, not the suspect wrists/ankles.


@pipeline_stage('mask_prompt_core')
def reliable_core(points, scores, probabilities, box, settings):
    indices = np.asarray(CORE)
    xy = points[indices]
    left, top, right, bottom = box
    valid = ((xy[:, 0] >= left) & (xy[:, 0] < right) &
             (xy[:, 1] >= top) & (xy[:, 1] < bottom) &
             (scores[indices] > settings['min_keypoint_score']) &
             (probabilities['presence'][indices] > settings['min_presence']) &
             (probabilities['visibility'][indices] > settings['min_visibility']))
    if 'flip_agreement' in probabilities:
        valid &= probabilities['flip_agreement'][indices] >= .5
    indices = indices[valid]
    # Stable ordering makes tied predictions independent of batching.
    return indices[np.argsort(-probabilities['visibility'][indices], kind='stable')]


def overlaps(box, other, threshold):
    # Intersection / smaller box detects containment as well as similar boxes.
    intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
        0, min(box[3], other[3]) - max(box[1], other[1]))
    area = min((box[2]-box[0])*(box[3]-box[1]), (other[2]-other[0])*(other[3]-other[1]))
    return intersection > 0 and intersection / area >= threshold


def _retain_quality_mask(record, mask, bounds):
    left, top, right, bottom = bounds
    packed = np.packbits(mask[top:bottom, left:right], axis=None)
    if packed.any():
        record['overlap_mask'] = base64.b64encode(packed).decode('ascii')


class RefinedPMPose:
    def __init__(self, pose, segmenter, settings, *, frame_batch_size=4, queue_bytes=134217728):
        self.pose, self.segmenter, self.settings = pose, segmenter, settings
        self.frame_batch_size, self.queue_bytes = frame_batch_size, queue_bytes

    def close(self):
        from .resident_models import release
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('close raw pose',lambda:release(self.pose)),('close segmenter',lambda:release(self.segmenter))])

    def predict(self, image, bounds):
        return self.predict_many([(image, bounds)])[0]

    def predict_many(self, frames):
        frames = list(frames)
        with pipeline_timing('pmpose_first_pass'):
            predictions = self.pose.predict_many(frames)
        if len(predictions) != len(frames):
            raise ValueError('PMPose returned an invalid refinement frame count.')
        return self.refine_many(frames, predictions)

    def _plan(self, bounds, prediction, image=None):
        points, scores, probabilities = prediction
        settings = self.settings
        result = (points.copy(), scores.copy(), {k: v.copy() for k, v in probabilities.items()})
        records = [dict(status='skipped', reason='no_reliable_prompts',
                        detector_bounds=list(box), bounds=list(box)) for box in bounds]
        cores = [reliable_core(points[i], scores[i], {k: v[i] for k, v in probabilities.items()}, box, settings)
                 for i, box in enumerate(bounds)]
        recovery = settings.get('recover_prompt_points', False)
        if recovery:
            valid, tests = prompt_recovery.eligible(points, scores, probabilities, bounds, settings)
            faces = prompt_recovery.assigned_faces(prompt_recovery.faces_for(image), bounds, points, valid)
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
            minimum = settings['min_prompt_points']
            if recovery:
                counts = {name: int((~passed[i, CORE]).sum()) for name, passed in tests.items()}
                counts['near_peer'] = len(cores[i]) - len(indices)
                records[i]['prompt_rejections'] = counts
                indices, minimum = prompt_recovery.recover(i, indices, points, valid, scores,
                    probabilities, bounds, faces.get(i), foreign, settings, records[i])
                records[i]['prompt_minimum'] = minimum
                # On contact prioritize distinct torso/head regions over adjacent facial points.
                if len(indices) >= minimum and any(j != i and overlaps(box, b, settings['overlap_threshold'])
                                                   for j, b in enumerate(bounds)):
                    xy = prompt_recovery.prompt_points(points, records)[i]
                    indices = spread_indices(xy, indices, len(indices), initial=1)
                records[i]['support_indices'] = [int(k) for k in cores[i] if k in indices]
            if len(indices) >= minimum:
                indices = spread_indices(points[i], indices, settings['max_prompt_points'],
                                         initial=minimum)
                jobs.append((i, indices, foreign))
        return result, records, cores, jobs

    def _prompts(self, bounds, points, cores, i, indices):
        prompts = points[i, indices]
        neighbours = [j for j, other in enumerate(bounds) if j != i
                      and overlaps(bounds[i], other, self.settings['overlap_threshold'])]
        negatives = []
        limit = self.settings['max_negative_prompt_points_per_neighbor']
        if limit:
            tolerance = max(2., .02 * math.hypot(bounds[i][2]-bounds[i][0], bounds[i][3]-bounds[i][1]))
            for j in neighbours:
                usable = [k for k in cores[j] if np.linalg.norm(prompts - points[j,k], axis=1).min() > tolerance]
                negatives.extend(points[j, spread_indices(points[j], usable, limit)])
        return prompts, neighbours, negatives

    def refine_many(self, frames, predictions):
        from .refinement_batches import refine_many
        return refine_many(self, frames, predictions)

    @pipeline_stage('pmpose_mask_refinement')
    def refine(self, image, bounds, prediction, *, plan=None, masks=None, defer=None):
        points = prediction[0]
        result, records, cores, jobs = plan or self._plan(bounds, prediction, image)
        points = prompt_recovery.prompt_points(points, records)
        second_pass = defer or self._second_pass
        pending = []
        valid, _ = prompt_recovery.eligible(*prediction, bounds, self.settings)
        evidence = [xy[ok] for xy, ok in zip(prediction[0], valid)]
        for candidate in select_masks(self, image, bounds, points, cores, jobs, records, masks, evidence):
            pending.append(candidate)
            if len(pending) >= self.pose.batch_size:
                second_pass(image, pending, result, records, prediction)
                pending.clear()
        if pending:
            second_pass(image, pending, result, records, prediction)
        return (*result, records)

    @pipeline_stage('pmpose_masked_second_pass')
    def _second_pass(self, image, pending, result, records, original):
        refined = self.pose.predict(image, [box for _, box, _, _ in pending],
                                    masks=[mask for _, _, mask, _ in pending])
        self._apply_second_pass(pending, result, records, original, refined)

    def _apply_second_pass(self, pending, result, records, original, refined):
        for offset, (i, box, mask, indices) in enumerate(pending):
            new_points, new_scores, probabilities = refined
            keep = ((new_scores[offset, indices] > self.settings['min_keypoint_score']) &
                    (probabilities['presence'][offset, indices] > self.settings['min_presence']) &
                    (probabilities['visibility'][offset, indices] > self.settings['min_visibility']) &
                    inside(mask, new_points[offset, indices]))
            old = records[i]['detector_bounds']
            tolerance = max(3., .15 * math.hypot(old[2]-old[0], old[3]-old[1]))
            keep &= np.linalg.norm(new_points[offset, indices] - records[i].get('prompt_points', original[0][i, indices]), axis=1) <= tolerance
            if keep.sum() < records[i].get('prompt_minimum', self.settings['min_prompt_points']):
                record = records[i]
                record['reason'] = 'second_pass_core_drift'
                # The first pose and original crop survive drift. For neighbouring
                # boxes, reuse SAM only when its torso ownership was established.
                has_neighbours = 'negative_points' in record
                if not has_neighbours or record['separated']:
                    _retain_quality_mask(record, mask, old)
                continue
            from .limb_ownership import fuse
            joints, confidence, evidence, restored = fuse(i, offset, original, refined, mask, box, self.settings)
            if restored:
                records[i]['restored_first_pass_joints'] = restored
            # Quality needs the validated mask even when encoder masking is off.
            # Both fields share one packed crop. Collection stores the quality mask separately from JSON.
            _retain_quality_mask(records[i], mask, box)
            if records[i].get('overlap_mask') and records[i].get('separated') and self.settings.get('masked_identity_crop'):
                records[i]['encoder_mask'] = records[i]['overlap_mask']
            result[0][i], result[1][i] = joints, confidence
            for key in result[2]:
                result[2][key][i] = evidence[key]
            records[i].update(status='refined', reason=None, bounds=box,
                              box_expanded=box != records[i]['detector_bounds'])
