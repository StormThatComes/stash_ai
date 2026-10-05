"""Select exact published proofs and restore their frozen qualification policy."""
from dataclasses import replace

from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.historical_markers import frozen_policy
from stash_ai.models import SOURCES


class PublishedRecords:
    """Lazy per-scene signature indexes shared by scene and marker replay."""
    def __init__(self, store, scene_id):
        self.store, self.scene_id = store, scene_id
        self.indexes = {}

    def matching(self, kind, required, *, scene_frames=False, error):
        key = kind, scene_frames
        if key not in self.indexes:
            from stash_ai.storage.archive import records as archived_records
            def with_archive(output):
                active = self.store.records(self.scene_id, output, borrow=True)
                archive = archived_records(self.store, self.scene_id, output)
                return {source: [*archive.get(source, []), *active.get(source, [])]
                        for source in archive.keys() | active.keys()}
            records = with_archive(kind)
            if scene_frames:
                from stash_ai.evidence.cache_views import frame_records
                records = frame_records(with_archive('scene'), records)
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
    context = replace(runtime, TEMPORAL_POLICY=frozen_policy(options))
    if kind == 'markers':
        context.MARKER_THRESHOLDS = options['marker_thresholds']
        for name, value in options['settings'].items():
            setattr(context, name, value)
    else:
        context.MAX_TAGS = options['max_tags']
        context.SCENE_THRESHOLDS = options['scene_thresholds']
    return context


def frozen_policies(options, field, selected, current):
    """New cleanup bans revoke proof; never admit votes excluded at publication."""
    frozen = options[field]
    return {source: frozenset(frozen[source]) |
            (current[source] if source in selected else frozenset()) for source in SOURCES}
