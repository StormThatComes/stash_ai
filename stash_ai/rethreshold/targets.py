"""Select lightweight saved-result targets and group preflight failures."""

def threshold_targets(store, scenes, settings, kinds):
    """Select requested output kinds using result keys/indexes without decoding payloads."""
    scene_ids = [str(scene) for scene in scenes]
    models = [config.source.key for config in settings]
    requested = frozenset(kinds)
    known = store.result_presence(models, scene_ids) | store.result_index_presence(models, scene_ids)
    targets = []
    for scene in scene_ids:
        for config in settings:
            model = config.source.key
            available = frozenset(kind for kind in requested if (scene, model, kind) in known)
            if available:
                targets.append((scene, config, available))
    selected = len(scene_ids) * len(settings)
    return targets, selected - len(targets), selected


def iter_threshold_groups(store, scenes, settings, kinds, *, chunk=128):
    """Yield bounded scene/model groups without retaining full selection key sets."""
    from itertools import batched
    models = [setting.source.key for setting in settings]
    requested = frozenset(kinds)
    for group in batched(map(str, scenes), chunk):
        known = store.result_presence(models, group) | store.result_index_presence(models, group)
        targets = []
        for scene in group:
            for setting in settings:
                available = frozenset(kind for kind in requested
                                      if (scene, setting.source.key, kind) in known)
                if available:
                    targets.append((scene, setting, available))
        yield targets, len(group)


def threshold_failure_reason(exc):
    """Collapse scene-specific preflight errors into stable operator-facing reasons."""
    from stash_ai.runtime.reasons import LABELS
    code = getattr(exc, 'reason_code', None)
    if code in LABELS:
        return LABELS[code]
    # Adapter for historical exceptions from external callers.
    message = str(exc).strip()
    if ': ' in message:
        prefix, remainder = message.split(': ', 1)
        if '/' in prefix:
            message = remainder
    lower = message.lower()
    rules = (
        ('no scene results', 'missing scene results'),
        ('scene frame evidence is missing', 'missing scene frame evidence'),
        ('scene cache floor', 'scene cache floor too high'),
        ('no cached marker frames', 'missing marker frames'),
        ('imported ranges cannot be rethresholded', 'imported marker ranges'),
        ('marker cache floor', 'marker cache floor too high'),
        ('cache floor', 'marker cache floor too high'),
        ('invalid cached marker threshold', 'invalid cached marker threshold'),
        ('obsolete active inference', 'obsolete inference evidence'),
        ('scene_threshold changed', 'scene threshold requires inference'),
        ('finish the pending merge', 'pending publication'),
    )
    return next((reason for token, reason in rules if token in lower), type(exc).__name__)
