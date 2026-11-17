"""SAM masks for COCO17 estimators without model-specific visibility heads."""
import numpy as np
from .pose_refinement import RefinedPMPose, retain_quality_mask


class PoseMasks(RefinedPMPose):
    """Confidence proxies select prompts; they never become measured visibility."""
    def predict_many(self, frames):
        frames = list(frames)
        values = self.pose.predict_many(frames)
        predictions = [(xy, score, dict(presence=score.copy(), visibility=score.copy()))
                       for xy, score in values]
        # Without a masked second estimator pass, temporal borrowed prompts cannot
        # be independently verified. Current-frame recovery still runs.
        results = []
        for (image, bounds), prediction in zip(frames, predictions):
            self.previous_cue = None
            results.append(self.refine(image, bounds, prediction))
        return results

    def _second_pass(self, image, pending, result, records, original):
        for i, box, mask, indices in pending:
            record = records[i]
            retain_quality_mask(record, mask, box)
            if record.get('separated') and self.settings.get('masked_identity_crop'):
                record['encoder_mask'] = record.get('overlap_mask')
            record.update(status='masked', reason=None, bounds=box,
                          point_evidence='confidence_proxy', visibility_measured=False,
                          box_expanded=box != record['detector_bounds'])


def wrap(pose, config, device):
    from .sam_pose2seg import load
    from .resident_models import current, Stage
    segmenter = (Stage('pose_segmenter', lambda: load(config, device), device)
                 if current() is not None else load(config, device))
    return PoseMasks(pose, segmenter, config.data['sapiensid']['pose']['refinement'])
