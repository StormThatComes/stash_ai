"""Optional cache cutoffs, independent from scene/marker publication policy."""


def scene_cache_floor(config, policy):
    required = config.scene_threshold * (policy.scene_support_ratio if policy.enabled else 1.0)
    configured = getattr(config, 'scene_cache_floor', None)
    # Retention may keep more evidence, never discard current publication inputs.
    return required if configured is None else min(required, configured)


def marker_cache_floor(config, policy):
    from stash_ai.temporal import marker_floor
    required = marker_floor(policy, config.marker_threshold)
    configured = getattr(config, 'marker_cache_floor', None)
    return required if configured is None else min(required, configured)
