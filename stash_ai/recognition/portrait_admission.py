"""Optional hard portrait admission using the current anchor quality rules."""
from .pose_policy import is_anchor,anchor_visibility
from .visibility import cached, valid_fraction


def extra_measurements(config):
    if config.data['portraits'].get('anchor_only',False) and anchor_visibility(config.data['clustering'])>0:
        return {'visibility'}
    return set()


def measurement_quality(quality, config):
    """Detach mask-derived portrait inputs when the stored cutoff differs."""
    from .config import DEFAULT_FACEOCC_LOGIT_THRESHOLD
    record = quality.get('visibility')
    if not isinstance(record, dict):
        return quality
    recorded = record.get('logit_threshold', DEFAULT_FACEOCC_LOGIT_THRESHOLD)
    if recorded == config.data['detection']['faceocc_logit_threshold']:
        return quality
    result = dict(quality)
    result.pop('visibility', None)
    result.pop('visible_face_fraction', None)
    if 'measurement_basis' in result:
        result['measurement_basis'] = {k: v for k, v in result['measurement_basis'].items() if k != 'visibility'}
    if result.get('sharpness_method') == 'faceocc_mask':
        result.update(sharpness=0., sharpness_pixels=0, sharpness_method='unavailable')
    return result


def ranking_quality(quality, config):
    """Apply the current coverage floor to retained VTT evidence without rewriting the scan."""
    record = quality.get('visibility')
    if not isinstance(record, dict) or 'source_fraction' not in record:
        return quality
    coverage = record['source_fraction']
    if valid_fraction(coverage) and coverage > 0 and coverage >= config.data['detection']['min_face_source_fraction']:
        return quality
    result = dict(quality)
    result.pop('visibility', None)
    result.pop('visible_face_fraction', None)
    result['visibility_source_fraction'] = coverage
    if result.get('sharpness_method') == 'faceocc_mask':
        result.update(sharpness=0., sharpness_pixels=0, sharpness_method='unavailable')
    return result


def pose_reason(quality,config):
    if not config.data['portraits'].get('anchor_only',False):return None
    pose={'max_anchor_pose_deviation':config.data['clustering']['max_anchor_pose_deviation']}
    return None if is_anchor(quality,pose) else 'portrait_anchor_required'


def reason(quality,config):
    if not config.data['portraits'].get('anchor_only',False):return None
    rejected=pose_reason(quality,config)
    if rejected:return rejected
    if extra_measurements(config) and (not isinstance(quality,dict) or not cached(quality, config.data['detection']['faceocc_logit_threshold'])):
        return 'portrait_anchor_visibility_unavailable'
    return None if is_anchor(quality,config.data['clustering']) else 'portrait_anchor_required'
