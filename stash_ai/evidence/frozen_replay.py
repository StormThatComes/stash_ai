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
        from stash_ai.storage.archive import matching_record
        matching = {}
        for source, signature in required.items():
            if source not in SOURCES or not isinstance(signature, str) or not signature:
                raise RuntimeError(error)
            key = kind, source, signature
            if key not in self.indexes:
                active = self.store.model_record(self.scene_id, kind, source, borrow=True)
                if active is not None and marker_segment_signature(active) == signature:
                    record = active
                else:
                    record = matching_record(self.store, self.scene_id, kind, source, signature)
                if record is None:
                    raise RuntimeError(error)
                self.indexes[key] = record
            matching[source] = self.indexes[key]
        return matching


def snapshot_context(runtime, options, kind):
    context = replace(runtime, TEMPORAL_POLICY=frozen_policy(options))
    if kind == 'markers':
        for name, value in options['settings'].items():
            setattr(context, name, value)
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
