"""Opt-in, orientation-independent PMPose duplicate suppression policy."""


def defaults():
    return dict(enabled=False, max_contact_attempts=4, contact_crop_variants=False, contact_min_iou=.70, contact_max_iou=.94, min_keypoint_score=.5, min_presence=.5, min_visibility=.5,
                min_shared_points=4, max_point_distance=.05, min_match_fraction=.8)


def validate(pose):
    from .config import _number
    settings = pose['deduplication']
    prefix = 'sapiensid.pose.deduplication.'
    if type(settings['enabled']) is not bool:
        raise ValueError(prefix + 'enabled must be boolean.')
    if settings['enabled'] and pose['backend'] not in ('pmpose','vitpose_plus_large','sapiens'):
        raise ValueError(prefix + 'enabled requires a top-down COCO17 pose backend.')
    for key in ('contact_min_iou','contact_max_iou'):
        _number(settings[key],prefix+key,0,1)
    if settings['contact_min_iou']>=settings['contact_max_iou']:
        raise ValueError(prefix+'contact_min_iou must be below contact_max_iou.')
    if type(settings['contact_crop_variants']) is not bool:
        raise ValueError(prefix+'contact_crop_variants must be boolean.')
    _number(settings['max_contact_attempts'], prefix+'max_contact_attempts', 0, 16, integer=True)
    for key in ('min_keypoint_score', 'min_presence', 'min_visibility'):
        _number(settings[key], prefix + key, 0, 1)
    _number(settings['min_shared_points'], prefix + 'min_shared_points', 4, 9, integer=True)
    _number(settings['max_point_distance'], prefix + 'max_point_distance', .001, .1)
    _number(settings['min_match_fraction'], prefix + 'min_match_fraction', .5, 1)


def contract(settings):
    return dict(version='coco17-core-deduplication-v2', **settings)
