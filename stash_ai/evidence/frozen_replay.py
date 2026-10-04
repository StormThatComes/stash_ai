"""Select exact published proofs and restore their frozen qualification policy."""
from dataclasses import replace

from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.models import SOURCES


class PublishedRecords:
    """Lazy per-scene signature indexes shared by scene and marker replay."""
    def __init__(self, store, scene_id):
        self.store, self.scene_id = store, scene_id
        self.indexes = {}

    def matching(self, kind, required, *, scene_frames=False, error):
        key = kind, scene_frames
        if key not in self.indexes:
            records = self.store.records(self.scene_id, kind, borrow=True)
            if scene_frames:
                from stash_ai.evidence.cache_views import frame_records
                records = frame_records(self.store.records(self.scene_id, 'scene', borrow=True), records)
            self.indexes[key] = {source: {marker_segment_signature(segment): segment for segment in segments}
                                 for source, segments in records.items()}
        matching = {}
        for source, signatures in required.items():
            available = self.indexes[key].get(source, {})
            if not set(signatures) <= set(available):
                raise RuntimeError(error)
            matching[source] = [available[signature] for signature in signatures]
        return matching


def snapshot_context(runtime, options, kind):
    context = replace(runtime, TEMPORAL_POLICY=TemporalPolicy.from_snapshot(options['policy']))
    if kind == 'markers':
        context.MARKER_THRESHOLDS = options.get('marker_thresholds', {})
        for name, value in options['settings'].items():
            setattr(context, name, value)
    else:
        context.MAX_TAGS = options['max_tags']
        context.SCENE_THRESHOLDS = options.get('scene_thresholds', {})
    return context


def frozen_policies(options, field, selected, current, *, legacy=frozenset()):
    """New cleanup bans revoke proof; never admit votes excluded at publication."""
    required = field in ('core_exclusions', 'scene_exclusions')
    frozen = options[field] if required else options.get(field, {})
    return {source: frozenset(frozen[source] if required else frozen.get(source, [])) | legacy |
            (current[source] if source in selected else frozenset()) for source in SOURCES}
