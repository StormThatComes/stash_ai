"""Optional hard portrait admission using the current anchor quality rules."""
from .pose_policy import is_anchor,anchor_visibility
from .visibility import cached


def extra_measurements(config):
    if config.data['portraits'].get('anchor_only',False) and anchor_visibility(config.data['clustering'])>0:
        return {'visibility'}
    return set()


def pose_reason(quality,config):
    if not config.data['portraits'].get('anchor_only',False):return None
    pose={'max_anchor_pose_deviation':config.data['clustering']['max_anchor_pose_deviation']}
    return None if is_anchor(quality,pose) else 'portrait_anchor_required'


def reason(quality,config):
    if not config.data['portraits'].get('anchor_only',False):return None
    rejected=pose_reason(quality,config)
    if rejected:return rejected
    if extra_measurements(config) and (not isinstance(quality,dict) or not cached(quality)):
        return 'portrait_anchor_visibility_unavailable'
    return None if is_anchor(quality,config.data['clustering']) else 'portrait_anchor_required'
