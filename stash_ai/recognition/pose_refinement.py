"""One pose -> instance mask -> masked pose pass, with bounded box recovery."""
import base64
import math

import numpy as np

from . import prompt_recovery
from .mask_candidates import check_mask, inside, regional_seeds, select_masks, spread_indices
from .profiling import pipeline_stage, pipeline_timing
from . import detection_phases as phases

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


def retain_quality_mask(record, mask, bounds):
    left, top, right, bottom = bounds
    packed = np.packbits(mask[top:bottom, left:right], axis=None)
    if packed.any():
        record['overlap_mask'] = base64.b64encode(packed).decode('ascii')
        record['mask_valid'] = True


class PromptCores(list):
    pass


class RefinedPMPose:
    def __init__(self, pose, segmenter, settings, *, frame_batch_size=4, queue_bytes=134217728, joint_quality=None):
        self.previous_cue = None
        self.pose, self.segmenter, self.settings = pose, segmenter, settings
        self.frame_batch_size, self.queue_bytes = frame_batch_size, queue_bytes
        # Prompt confidence is intentionally stricter than output joint eligibility.
        self.joint_quality = dict(settings, **(joint_quality or {}))

    @property
    def mask_usage(self):
        from .mask_usage import applications
        return applications(self.settings, generic=getattr(self, 'generic_estimator', False))

    @property
    def requires_second_pass(self):
        return self.mask_usage['pose_refinement']

    def close(self):
        from .resident_models import release
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('close raw pose',lambda:release(self.pose)),('close segmenter',lambda:release(self.segmenter))])

    def mask_frame(self, image):
        return self.segmenter.frame(image)

    def predict(self, image, bounds):
        return self.predict_many([(image, bounds)])[0]

    def predict_many(self, frames):
        frames = list(frames)
        from .refinement_budget import begin
        begin(frames,self.settings)
        with pipeline_timing('pmpose_first_pass'):
            predictions = self.pose.predict_many(frames)
        if len(predictions) != len(frames):
            raise ValueError('PMPose returned an invalid refinement frame count.')
        return self.refine_many(frames, predictions)

    def _plan(self, bounds, prediction, image=None):
        phases.pose(image, 'Initial keypoints (before refinement)', bounds, prediction)
        points, scores, probabilities = prediction
        settings = self.settings
        result = (points.copy(), scores.copy(), {k: v.copy() for k, v in probabilities.items()})
        records = [dict(status='skipped', reason='no_reliable_prompts',
                        detector_bounds=list(box), bounds=list(box), mask_usage=self.mask_usage,
                        mask_applied=dict.fromkeys(self.mask_usage, False)) for box in bounds]
        cores = [reliable_core(points[i], scores[i], {k: v[i] for k, v in probabilities.items()}, box, settings)
                 for i, box in enumerate(bounds)]
        cores = PromptCores(cores)
        if settings.get('negative_limb_prompts',False):
            from .limb_ownership import PARENT
            extra = []
            for i,box in enumerate(bounds):
                valid,_ = prompt_recovery.eligible(points,scores,probabilities,bounds,settings)
                extra.append(list(cores[i]) + [k for k,p in PARENT.items()
                    if valid[i,k] and valid[i,p] and min(scores[i,k],scores[i,p]) >= .8])
            cores.negatives = extra
        recovery = settings.get('recover_prompt_points', False)
        from .temporal_prompts import capture, borrow
        previous = self.previous_cue if self.requires_second_pass else None
        context = getattr(self, 'cue_contexts', None) if self.requires_second_pass else None
        signature = getattr(image, '_recognition_cue_signature', None)
        if context is not None and signature:
            scan = getattr(image, '_recognition_scan', None)
            previous = context.get((scan, signature[0]-1))
            self.next_cue = context.get((scan, signature[0]+1))
        self.previous_cue = None
        if recovery:
            valid, tests = prompt_recovery.eligible(points, scores, probabilities, bounds, settings)
            self.previous_cue = capture(image, bounds, points, valid)
            faces = prompt_recovery.assigned_faces(prompt_recovery.faces_for(image), bounds, points, valid)
        jobs = []
        for i, box in enumerate(bounds):
            if settings['mode'] == 'overlap' and not any(
                    overlaps(box, other, settings['overlap_threshold']) for j, other in enumerate(bounds) if i != j):
                records[i]['reason'] = 'no_overlap'
                if recovery:
                    prompt_recovery.record_points(records[i], points[i], tests, i, cores[i], cores[i])
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
                record = records[i]
                record['prompt_rejections'] = counts
                prompt_recovery.record_points(record, points[i], tests, i, cores[i], indices)
                indices, minimum = prompt_recovery.recover(i, indices, points, valid, scores,
                    probabilities, bounds, faces.get(i), foreign, settings, record)
                local = (prompt_recovery.local_candidates(points, scores, tests, valid, i)
                         if previous and self.previous_cue else {})
                indices = borrow(previous, self.previous_cue, i, indices, settings['min_prompt_points'],
                                 record, local=local)
                future = getattr(self,'next_cue',None) if self.requires_second_pass else None
                if future and 'temporal_prompts' not in record:
                    local = prompt_recovery.local_candidates(points,scores,tests,valid,i)
                    indices = borrow(future,self.previous_cue,i,indices,settings['min_prompt_points'],record,local=local)
                if 'temporal_prompts' in records[i]:
                    minimum = settings['min_prompt_points']
                records[i]['prompt_minimum'] = minimum
                record['support_indices'] = [int(k) for k in cores[i] if k in indices]
            if len(indices) >= minimum:
                limit = len(indices) if settings.get('adaptive_decoder_retries') else settings['max_prompt_points']
                if recovery:
                    xy = prompt_recovery.prompt_points(points, records)[i]
                    if any(j != i and overlaps(box, b, settings['overlap_threshold'])
                           for j, b in enumerate(bounds)):
                        indices = spread_indices(xy, indices, limit)
                    else:
                        indices = regional_seeds(xy, indices, record.get('temporal_prompts', ()))[:limit]
                else:
                    indices = spread_indices(points[i], indices, limit, initial=minimum)
                jobs.append((i, indices, foreign))
        if phases.active(image) is not None:
            planned = {i for i, _, _ in jobs}
            phases.notes(image, 'Prompt eligibility / recovery', [
                dict(r, reason=None, status='eligible for mask prompts') if i in planned else r
                for i, r in enumerate(records)])
        return result, records, cores, jobs

    def _prompts(self, bounds, points, cores, i, indices, leaks=None):
        prompts = points[i, indices]
        neighbours = [j for j, other in enumerate(bounds) if j != i
                      and (overlaps(bounds[i], other, self.settings['overlap_threshold']) or (leaks and leaks.get(j)))]
        negatives = []
        limit = self.settings['max_negative_prompt_points_per_neighbor']
        if limit:
            tolerance = max(2., .02 * math.hypot(bounds[i][2]-bounds[i][0], bounds[i][3]-bounds[i][1]))
            for j in neighbours:
                usable = [k for k in getattr(cores,'negatives',cores)[j] if np.linalg.norm(prompts - points[j,k], axis=1).min() > tolerance]
                if leaks and j in leaks:
                    usable.sort(key=lambda k: k not in leaks[j])
                negatives.extend(points[j, spread_indices(points[j], usable, limit)])
        return prompts, neighbours, negatives

    def refine_many(self, frames, predictions):
        if self.requires_second_pass and self.settings.get('bidirectional_prompts',False):
            from .pose_context import contexts
            self.cue_contexts = contexts(self, frames, predictions)
            try:
                from .refinement_batches import refine_many
                return refine_many(self, frames, predictions)
            finally:
                self.cue_contexts = self.next_cue = None
        from .refinement_batches import refine_many
        return refine_many(self, frames, predictions)

    def candidates(self, image, bounds, prediction, plan, masks, decoder=None):
        _, records, cores, jobs = plan
        points = prompt_recovery.prompt_points(prediction[0], records)
        valid, _ = prompt_recovery.eligible(*prediction, bounds, self.settings)
        evidence = [xy[ok] for xy, ok in zip(prediction[0], valid)]
        return select_masks(self, image, bounds, points, cores, jobs, records, masks, evidence, decoder, valid)

    @pipeline_stage('pmpose_mask_refinement')
    def refine(self, image, bounds, prediction, *, plan=None, masks=None, defer=None, selected=None):
        plan = plan or self._plan(bounds, prediction, image)
        result, records = plan[:2]
        second_pass = defer or self._second_pass
        pending = []
        if selected is None:
            selected = self.candidates(image, bounds, prediction, plan, masks)
        reliable = None
        if self.requires_second_pass and len(bounds):
            reliable, _ = prompt_recovery.eligible(*prediction, bounds, self.settings)
        for candidate in selected:
            if getattr(self,'requires_second_pass',True):
                from .refinement_budget import budget
                i,box,mask,_=candidate
                # An isolated complete skeleton already supported by its mask needs no second inference.
                edited = any(records[i].get(key) for key in ('bridge_cuts','component_edits','temporal_prompts'))
                if reliable is not None and not edited and reliable[i].all() and tuple(box) == tuple(bounds[i]) and not any(
                        j != i and overlaps(bounds[i], other, 0) for j,other in enumerate(bounds)):
                    xy = prediction[0][i].astype(int)
                    left,top,right,bottom = bounds[i]
                    contained = np.count_nonzero(mask) == np.count_nonzero(mask[top:bottom,left:right])
                    if (contained and len(set(map(tuple,xy))) == len(xy) and (xy >= 0).all()
                            and (xy[:,0] < mask.shape[1]).all() and (xy[:,1] < mask.shape[0]).all()
                            and mask[xy[:,1],xy[:,0]].all()):
                        self._mask_only(image, [candidate], result, records, prediction)
                        records[i]['second_pass_reason'] = 'complete_isolated_first_pass'
                        continue
                if not budget(image,self.settings).admit('masked_pose',record=records[i]):
                    records[i].update(status='skipped',reason='shared_retry_budget')
                    retain_quality_mask(records[i],mask,records[i]['detector_bounds'])
                    continue
            pending.append(candidate)
            if len(pending) >= self.pose.batch_size:
                second_pass(image, pending, result, records, prediction)
                pending.clear()
        if pending:
            second_pass(image, pending, result, records, prediction)
        if defer is None:
            self.finish_maskless(image,bounds,prediction,result,records)
        from .refinement_budget import budget
        used=budget(image,self.settings)
        for record in records:
            record['retry_budget']=dict(used=dict(used.used),skipped=dict(used.skipped),remaining=used.remaining)
        return (*result, records)

    def finish_maskless(self,image,bounds,prediction,result,records):
        if not self.mask_usage['keypoint_filtering']:
            return
        if self.settings.get('joint_limb_ownership',False):
            from .joint_limbs import settle
            settle(result,bounds,records,self.settings)
        from .limb_ownership import maskless
        for i, record in enumerate(records):
            if not record.get('overlap_mask') and record.get('status') != 'refined':
                removed = maskless(prediction,i,bounds[i],image.size,result,self.joint_quality)
                if removed:
                    record['withheld_rival_joints'] = removed
        phases.refined_pose(image, 'After ownership / maskless fallback', result, records)

    def masked_inputs(self, image, jobs, original):
        if getattr(self.pose,'supports_rival_mask',False):
            from .limb_ownership import conditioning
            return conditioning(image,jobs,original)
        return [mask for _,_,mask,_ in jobs]

    def _second_pass(self, image, pending, result, records, original):
        if not self.requires_second_pass:
            return self._mask_only(image, pending, result, records, original)
        with pipeline_timing('pmpose_masked_second_pass'):
            refined = self.pose.predict(image, [box for _, box, _, _ in pending],
                                        masks=self.masked_inputs(image,pending,original))
        phases.masked_pose(image, pending, refined)
        self._apply_second_pass(pending, result, records, original, refined)
        phases.refined_pose(image, 'After refinement acceptance / joint fusion', result, records)

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
            temporal = {k for k, _ in records[i].get('temporal_prompts', ())}
            used_temporal = [j for j, k in enumerate(indices) if k in temporal]
            if (keep.sum() < records[i].get('prompt_minimum', self.settings['min_prompt_points'])
                    or not keep[used_temporal].all()):
                record = records[i]
                record['reason'] = 'second_pass_core_drift'
                # The first pose and original crop survive drift. For neighbouring
                # boxes, reuse SAM only when its torso ownership was established.
                has_neighbours = 'negative_points' in record
                if not used_temporal and (not has_neighbours or record['separated']):
                    retain_quality_mask(record, mask, old)
                continue
            from .limb_ownership import fuse
            if self.mask_usage['keypoint_filtering']:
                joints, confidence, evidence, restored = fuse(i, offset, original, refined, mask, box, self.joint_quality)
            else:
                joints, confidence = new_points[offset].copy(), new_scores[offset].copy()
                evidence = {k: v[offset].copy() for k, v in probabilities.items()}
                restored = []
            if restored:
                records[i]['restored_first_pass_joints'] = restored
            # Quality needs the validated mask even when encoder masking is off.
            # Both fields share one packed crop. Collection stores the quality mask separately from JSON.
            retain_quality_mask(records[i], mask, box)
            if records[i].get('overlap_mask') and records[i].get('separated') and self.settings.get('masked_identity_crop'):
                records[i]['encoder_mask'] = records[i]['overlap_mask']
            records[i].setdefault('mask_applied', {}).update(
                pose_refinement=True, keypoint_filtering=self.mask_usage['keypoint_filtering'],
                box_expansion=box != records[i]['detector_bounds'],
                body_encoder_input=bool(records[i].get('encoder_mask')))
            result[0][i], result[1][i] = joints, confidence
            for key in result[2]:
                result[2][key][i] = evidence[key]
            records[i].update(status='refined', reason=None, bounds=box,
                              box_expanded=box != records[i]['detector_bounds'])

    def _mask_only(self, image, pending, result, records, original):
        for i, box, mask, indices in pending:
            record = records[i]
            if self.mask_usage['keypoint_filtering']:
                from .limb_ownership import fuse, PARENT
                # A single pass can remove foreign points, but cannot corroborate a
                # disconnected endpoint by comparing the prediction with itself.
                own=(original[0][i:i+1],original[1][i:i+1],{k:v[i:i+1] for k,v in original[2].items()})
                xy,score,prob,_=fuse(i,0,original,own,mask,box,self.joint_quality)
                for k,p in PARENT.items():
                    if score[p]<=self.joint_quality['min_keypoint_score']:
                        xy[k],score[k]=-1,0
                        for value in prob.values():value[k]=0
                removed=np.flatnonzero((original[1][i]>0)&(score==0)).tolist()
                if removed:record['mask_ownership_withheld']=removed
                result[0][i],result[1][i]=xy,score
                for key in result[2]:result[2][key][i]=prob[key]
            retain_quality_mask(record, mask, box)
            if record.get('separated') and self.settings.get('masked_identity_crop'):
                record['encoder_mask'] = record.get('overlap_mask')
            record.setdefault('mask_applied', {}).update(
                keypoint_filtering=self.mask_usage['keypoint_filtering'],
                box_expansion=box != record['detector_bounds'],
                body_encoder_input=bool(record.get('encoder_mask')))
            record.update(status='masked', reason=None, bounds=box,
                          box_expanded=box != record['detector_bounds'])
            if getattr(self, 'generic_estimator', False):
                record.update(point_evidence='confidence_proxy', visibility_measured=False)
        from . import detection_phases as phases
        title = ('After mask ownership filtering (no second pose pass)' if self.mask_usage['keypoint_filtering']
                 else 'Mask retained (keypoints unchanged)')
        phases.refined_pose(image, title, result, records)
