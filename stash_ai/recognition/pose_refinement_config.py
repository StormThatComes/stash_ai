"""Optional SAM-pose2seg refinement settings; importable without ML packages."""


def defaults():
    return dict(enabled=False, directory='', mode='all', overlap_threshold=.2,
                min_keypoint_score=.5, min_presence=.5, min_visibility=.5,
                min_prompt_points=2, max_prompt_points=3, min_mask_score=.5,
                min_prompt_coverage=1., max_foreign_coverage=0.,
                masked_identity_crop=False, expand_boxes=True, max_box_area_ratio=2., max_side_expansion=.5)


def validate(pose):
    from .config import _number
    settings = pose['refinement']
    prefix = 'sapiensid.pose.refinement.'
    for key in ('enabled', 'expand_boxes', 'masked_identity_crop'):
        if type(settings[key]) is not bool:
            raise ValueError(prefix + key + ' must be boolean.')
    if settings['masked_identity_crop'] and not settings['enabled']:
        raise ValueError(prefix + 'masked_identity_crop requires refinement.enabled.')
    if settings['enabled'] and pose['backend'] != 'pmpose':
        raise ValueError(prefix + 'enabled requires pose.backend: pmpose.')
    if settings['mode'] not in ('all', 'overlap'):
        raise ValueError(prefix + 'mode must be all or overlap.')
    for key in ('overlap_threshold', 'min_keypoint_score', 'min_presence', 'min_visibility',
                'min_mask_score', 'min_prompt_coverage', 'max_foreign_coverage'):
        _number(settings[key], prefix + key, 0, 1)
    for key in ('min_prompt_points', 'max_prompt_points'):
        _number(settings[key], prefix + key, 1, 8, integer=True)
    if settings['min_prompt_points'] > settings['max_prompt_points']:
        raise ValueError(prefix + 'min_prompt_points exceeds max_prompt_points.')
    _number(settings['max_box_area_ratio'], prefix + 'max_box_area_ratio', 1, 10)
    _number(settings['max_side_expansion'], prefix + 'max_side_expansion', 0, 2)
    value = settings['directory']
    if not isinstance(value, str) or '\x00' in value or value and not value.strip():
        raise ValueError(prefix + 'directory requires a local path.')


def contract(settings):
    from . import sam_pose2seg_registry as registry
    return dict(version='pmpose-sam-pose2seg-refinement-v2',
                source_revision=registry.SOURCE_REVISION, weights_sha256=registry.SHA256,
                mask_background_alpha=.25,
                **{k: v for k, v in settings.items() if k != 'directory'})
