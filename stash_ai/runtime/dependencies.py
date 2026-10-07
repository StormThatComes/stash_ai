"""Setting dependencies: prediction identity and replay/publication are distinct."""

SETTING_DEPENDENCIES = {
    'APPLY_SCENE_TAGS': frozenset(('evidence', 'publication')),
    'CREATE_MARKERS': frozenset(('evidence', 'publication')),
    'MARKER_MIN_HITS': frozenset(('evidence', 'publication', 'marker_snapshot')),
    'MARKER_MERGE_GAP_SECONDS': frozenset(('evidence', 'publication', 'marker_snapshot')),
    'MARKER_MAX_TAGS_PER_FRAME': frozenset(('evidence', 'publication', 'marker_snapshot')),
    'TEMPORAL_POLICY': frozenset(('evidence', 'policy_snapshot')),
    'MARKER_MATCH_TOLERANCE_SECONDS': frozenset(('publication',)),
    'CREATE_MISSING': frozenset(('publication',)),
    'AGGREGATION': frozenset(('prediction',)),
    'TOP_K_FRAMES': frozenset(('prediction',)),
    'JOYTAG_CLIP_MEAN': frozenset(('prediction',)),
    'JOYTAG_CLIP_STD': frozenset(('prediction',)),
    'MAX_FRAMES_PER_SCENE': frozenset(('input',)),
}

# Model weights, ordered vocabulary and effective input manifests are identities,
# not configuration-file/code hashes. Only the two retention ratios affect raw
# observation requirements; other temporal settings replay existing evidence.
MODEL_DEPENDENCIES = {
    'source': 'prediction', 'model_dir': 'assets', 'batch_size': 'prediction',
    'scene_threshold': 'threshold_reuse', 'marker_threshold': 'threshold_reuse',
    'scene_cache_floor': 'retention', 'marker_cache_floor': 'retention',
}


def settings_for(stage):
    return tuple(name for name, stages in SETTING_DEPENDENCIES.items() if stage in stages)


def capture(app, stage):
    return {name: getattr(app, name) for name in settings_for(stage)}
