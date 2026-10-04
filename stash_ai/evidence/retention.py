"""Optional cache cutoffs, independent from scene/marker publication policy."""

from stash_ai.models import ModelSettings


def scene_cache_floor(config: ModelSettings, policy):
    required = config.scene_threshold * (policy.scene_support_ratio if policy.enabled else 1.0)
    configured = config.scene_cache_floor
    # Retention may keep more evidence, never discard current publication inputs.
    return required if configured is None else min(required, configured)


def marker_cache_floor(config: ModelSettings, policy):
    from stash_ai.evidence.temporal.policy import marker_floor
    required = marker_floor(policy, config.marker_threshold)
    configured = config.marker_cache_floor
    return required if configured is None else min(required, configured)
