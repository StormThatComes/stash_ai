"""Opt-in, orientation-independent PMPose duplicate suppression policy."""


def defaults():
    return dict(enabled=False, min_keypoint_score=.5, min_presence=.5, min_visibility=.5,
                min_shared_points=4, max_point_distance=.05, min_match_fraction=.8)


def validate(pose):
    from .config import _number
    settings = pose['deduplication']
    prefix = 'sapiensid.pose.deduplication.'
    if type(settings['enabled']) is not bool:
        raise ValueError(prefix + 'enabled must be boolean.')
    if settings['enabled'] and pose['backend'] != 'pmpose':
        raise ValueError(prefix + 'enabled requires pose.backend: pmpose.')
    for key in ('min_keypoint_score', 'min_presence', 'min_visibility'):
        _number(settings[key], prefix + key, 0, 1)
    _number(settings['min_shared_points'], prefix + 'min_shared_points', 4, 9, integer=True)
    _number(settings['max_point_distance'], prefix + 'max_point_distance', .001, .1)
    _number(settings['min_match_fraction'], prefix + 'min_match_fraction', .5, 1)


def contract(settings):
    return dict(version='pmpose-core-deduplication-v1', **settings)
