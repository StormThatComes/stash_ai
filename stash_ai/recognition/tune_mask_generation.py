"""Replay mask generation on the inference thread from one cue's initial pose."""
from contextlib import contextmanager
from copy import copy, deepcopy

from .tune_mask_candidate import LOCAL_PATHS

PROMPT_PATHS = frozenset('pose.mask.' + key for key in (
    'mode', 'overlap_threshold', 'min_keypoint_score', 'min_presence', 'min_visibility',
    'min_recovery_mask_score', 'adaptive_decoder_retries', 'recover_prompt_points',
    'negative_limb_prompts', 'min_prompt_points', 'max_prompt_points',
    'max_negative_prompt_points_per_neighbor', 'max_decoder_attempts', 'max_mask_combinations'))


def generate_edited(detector, frame, token, base, edits):
    # Config normalization and YAML serialization also belong off the Tk thread.
    from .detection_parameters import apply_edits
    config, _ = apply_edits(base, edits, detection_preview=True)
    return config, detector.regenerate_masks(frame, token, config)


class MaskReplay:
    def __init__(self, frame):
        self.frame = frame
        self.token = object()
        self.owner = None
        self.encoded = None

    @contextmanager
    def bind(self, image):
        image._tuner_mask_replay = self
        try:
            yield
        finally:
            del image._tuner_mask_replay

    def remember_pose(self, owner, image, bounds, prediction):
        # Store the first plan only. A later retry may already contain repaired poses.
        if self.owner is not None or len(bounds) > 256:
            return
        from .prompt_recovery import faces_for
        from .refinement_budget import budget
        self.owner = copy(owner)
        self.bounds = deepcopy(bounds)
        self.prediction = deepcopy(prediction)
        self.faces = faces_for(image)
        self.signature = getattr(image, '_recognition_cue_signature', None)
        self.scan = getattr(image, '_recognition_scan', None)
        self.remaining = budget(image, owner.settings).remaining

    def supports(self, token, frame, config):
        if self.token is not token or frame is not self.frame or self.owner is None:
            return False
        settings = config.data['sapiensid']['pose']['refinement']
        old = self.owner.settings
        allowed = {path.rpartition('.')[2] for path in PROMPT_PATHS | LOCAL_PATHS}
        if any(old.get(key) != settings.get(key) for key in old.keys() | settings.keys() if key not in allowed):
            return False
        # RF-DETR removes proposals below its confidence floor during encoding.
        return settings['backend'] != 'rfdetr_seg_large' or settings['min_mask_score'] == old['min_mask_score']

    def generate(self, config):
        from .detection_phases import capture
        from .prompt_recovery import face_inputs
        from .refinement_budget import budget
        owner = copy(self.owner)
        owner.settings = config.data['sapiensid']['pose']['refinement']
        with self.frame.image.copy() as image, self.bind(image), face_inputs([(image, self.faces)]), capture(
                image, backend=config.data['sapiensid']['pose']['backend'], samples_only=True) as history:
            image._recognition_cue_signature = self.signature
            image._recognition_scan = self.scan
            budget(image, owner.settings).remaining = self.remaining
            plan = owner._plan(self.bounds, self.prediction, image)
            # candidates runs the production retry and multi-person selection loop.
            for _ in owner.candidates(image, self.bounds, self.prediction, plan, None):
                pass
        history.replay_token = self.token
        return history
