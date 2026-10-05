"""Select exact published proofs and restore their frozen qualification policy."""
from dataclasses import replace

from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.temporal.policy import frozen_policy
from stash_ai.models import SOURCES


class PublishedRecords:
    """Lazy per-scene signature indexes shared by scene and marker replay."""
    def __init__(self, store, scene_id):
        self.store, self.scene_id = store, scene_id
        self.indexes = {}

    def matching(self, kind, required, *, error):
        key = kind
        if key not in self.indexes:
            from stash_ai.storage.archive import records as archived_records
            def with_archive(output):
                active = self.store.records(self.scene_id, output, borrow=True)
                archive = archived_records(self.store, self.scene_id, output)
                return {source: [*archive.get(source, []), *([active[source]] if source in active else [])]
                        for source in archive.keys() | active.keys()}
            records = with_archive(kind)
            self.indexes[key] = {source: {marker_segment_signature(segment): segment for segment in segments}
                                 for source, segments in records.items()}
        matching = {}
        for source, signature in required.items():
            available = self.indexes[key].get(source, {})
            if not isinstance(signature, str) or signature not in available:
                raise RuntimeError(error)
            matching[source] = available[signature]
        return matching


def snapshot_context(runtime, options, kind):
    context = replace(runtime, TEMPORAL_POLICY=frozen_policy(options))
    if kind == 'markers':
        for name, value in options['settings'].items():
            setattr(context, name, value)
    else:
        context.MAX_TAGS = options['max_tags']
    return context


def frozen_policies(options, field, selected, current):
    """New cleanup bans revoke proof; never admit votes excluded at publication."""
    frozen = options[field]
    return {source: frozenset(frozen[source]) |
            (current[source] if source in selected else frozenset()) for source in SOURCES}


def frozen_records(options, kind, records):
    from stash_ai.storage.thresholds import overlay
    from stash_ai.storage.segments import StoredSegment
    thresholds = options['scene_thresholds' if kind == 'scene' else 'marker_thresholds']
    return {source: StoredSegment(overlay(dict(record), kind, thresholds.get(marker_segment_signature(record))))
            for source, record in records.items()}
