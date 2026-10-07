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


def threshold_failure_reason(exc):
    """Collapse scene-specific preflight errors into stable operator-facing reasons."""
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
