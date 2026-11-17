"""Optional SAM-pose2seg refinement settings; importable without ML packages."""


# Omitted keys retain the historical mask and its inference fingerprint.
LEGACY_CLEANUP = dict(min_island_pixels=4, min_island_fraction=.001,
                      max_island_gap_fraction=0., bridge_radius_pixels=0,
                      max_bridge_lobe_fraction=.12, min_limb_path_coverage=.75,
                      min_limb_foreign_distance_pixels=3., min_foreign_island_points=2)


def defaults():
    return dict(enabled=False, directory='', mode='all', overlap_threshold=.2,
                min_keypoint_score=.5, min_presence=.5, min_visibility=.5,
                min_recovery_mask_score=None, adaptive_decoder_retries=False, recover_prompt_points=False, min_prompt_points=2, max_prompt_points=3, max_negative_prompt_points_per_neighbor=1, min_mask_score=.5,
                min_prompt_coverage=1., max_foreign_coverage=0., **LEGACY_CLEANUP,
                masked_identity_crop=False, expand_boxes=True, max_box_area_ratio=2., max_side_expansion=.5)


def validate(pose):
    from .config import _number
    settings = pose['refinement']
    prefix = 'sapiensid.pose.refinement.'
    if type(settings['enabled']) is not bool:
        raise ValueError(prefix + 'enabled must be boolean.')
    if not settings['enabled']:
        return
    for key in ('expand_boxes', 'masked_identity_crop', 'recover_prompt_points', 'adaptive_decoder_retries'):
        if type(settings[key]) is not bool:
            raise ValueError(prefix + key + ' must be boolean.')
    if pose['backend'] != 'pmpose':
        raise ValueError(prefix + 'enabled requires sapiensid.pose.backend: pmpose.')
    if settings['mode'] not in ('all', 'overlap'):
        raise ValueError(prefix + 'mode must be all or overlap.')
    for key in ('overlap_threshold', 'min_keypoint_score', 'min_presence', 'min_visibility',
                'min_mask_score', 'min_prompt_coverage', 'max_foreign_coverage'):
        _number(settings[key], prefix + key, 0, 1)
    for key in ('min_island_fraction', 'max_island_gap_fraction', 'max_bridge_lobe_fraction', 'min_limb_path_coverage'):
        _number(settings[key], prefix + key, 0, 1)
    _number(settings['min_island_pixels'], prefix + 'min_island_pixels', 0, 1000000, integer=True)
    _number(settings['min_limb_foreign_distance_pixels'], prefix + 'min_limb_foreign_distance_pixels', 0, 1000000)
    _number(settings['min_foreign_island_points'], prefix + 'min_foreign_island_points', 1, 1000, integer=True)
    _number(settings['bridge_radius_pixels'], prefix + 'bridge_radius_pixels', 0, 8, integer=True)
    if settings['min_recovery_mask_score'] is not None:
        _number(settings['min_recovery_mask_score'], prefix+'min_recovery_mask_score', 0, 1)
    for key in ('min_prompt_points', 'max_prompt_points'):
        _number(settings[key], prefix + key, 1, 8, integer=True)
    _number(settings['max_negative_prompt_points_per_neighbor'], prefix + 'max_negative_prompt_points_per_neighbor', 0, 8, integer=True)
    if settings['min_prompt_points'] > settings['max_prompt_points']:
        raise ValueError(prefix + 'min_prompt_points exceeds max_prompt_points.')
    _number(settings['max_box_area_ratio'], prefix + 'max_box_area_ratio', 1, 10)
    _number(settings['max_side_expansion'], prefix + 'max_side_expansion', 0, 2)
    value = settings['directory']
    if not isinstance(value, str) or '\x00' in value or value and not value.strip():
        raise ValueError(prefix + 'directory requires a local path.')


def contract(settings):
    from . import sam_pose2seg_registry as registry
    active = {k: v for k, v in settings.items() if k != 'directory' and k not in LEGACY_CLEANUP}
    for key in LEGACY_CLEANUP.keys() - {'bridge_radius_pixels', 'max_bridge_lobe_fraction'}:
        if settings.get(key, LEGACY_CLEANUP[key]) != LEGACY_CLEANUP[key]:
            active[key] = settings[key]
    if settings.get('bridge_radius_pixels') and settings.get('max_bridge_lobe_fraction'):
        active.update(bridge_radius_pixels=settings['bridge_radius_pixels'],
                      max_bridge_lobe_fraction=settings['max_bridge_lobe_fraction'])
    return dict(version=('pmpose-sam-pose2seg-refinement-v17' if settings.get('recover_prompt_points')
                         else 'pmpose-sam-pose2seg-refinement-v14'),
                source_revision=registry.SOURCE_REVISION, weights_sha256=registry.SHA256,
                mask_background_alpha=.25,
                **active)
