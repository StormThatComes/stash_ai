"""Replay scene qualification from sparse frame evidence and exact aggregates."""
from dataclasses import replace

from stash_ai.evidence.identity import SCENE_OVERRIDE


def scene_threshold(segment):
    return segment.get(SCENE_OVERRIDE, segment['meta'].get('scene_threshold'))


def validate_scene(app, segment, threshold, *, has_frames=None):
    from stash_ai.evidence.thresholds import _score
    meta = segment['meta']
    from stash_ai.evidence.formats import validate_current
    validate_current({**segment, **({'has_frames': has_frames} if has_frames is not None else {})})
    needed = threshold * app.TEMPORAL_POLICY.scene_support_ratio
    floor = meta.get('scene_score_floor')
    has_frames = 'frames' in segment if has_frames is None else has_frames
    if (segment['mode'] != 'scene' or meta.get('origin') != 'inference'
            or meta.get('scene_frames_format') != 1 or not has_frames):
        raise ValueError('scene frame evidence is missing; run inference once to populate the cache.')
    if not _score(floor) or floor > needed:
        raise ValueError(f'scene cache floor {floor!r} exceeds required {needed:g}; run inference.')


def project(app, segment):
    """A transient view; original scores, statistics and provenance stay immutable."""
    from stash_ai.domain import Cue
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.storage.segments import StoredSegment
    from stash_ai.evidence.temporal.statistics import SceneAccumulator
    if getattr(segment, '_scene_projected', False):
        return segment
    signature = marker_segment_signature(segment)
    meta = segment['meta']
    from stash_ai.evidence.formats import validate_current
    validate_current(segment)
    threshold = (app.SCENE_THRESHOLDS.get(signature, meta.get('scene_threshold'))
                 if app.SCENE_THRESHOLDS is not None else scene_threshold(segment))
    support_changed = meta.get('scene_support_floor') != threshold * app.TEMPORAL_POLICY.scene_support_ratio
    if threshold == meta.get('scene_threshold') and not support_changed:
        return segment
    validate_scene(app, segment, threshold)
    items = segment['items']
    meta = {**meta, 'scene_threshold': threshold}
    labels = [item['tag'] for item in items]
    indexes = {tag: i for i, tag in enumerate(labels)}
    accumulator = SceneAccumulator(threshold, app.TEMPORAL_POLICY)
    for row in segment['frames']:
        accumulator.add_sparse(((indexes[tag], score) for tag, score in row['tags']),
                               Cue(row['start'], row['end'], '', (0, 0, 1, 1)))
    statistics = accumulator.statistics(labels)
    items = [{**item, 'statistics': statistics.get(item['tag'])} for item in items]
    meta['scene_support_floor'] = threshold * app.TEMPORAL_POLICY.scene_support_ratio
    result = StoredSegment({**segment, 'meta': meta, 'items': items})
    result._evidence_digest, result._scene_projected = signature, True
    return result


def views(app, records):
    from stash_ai.evidence.identity import marker_segment_signature
    if app.SCENE_THRESHOLDS is None:
        app = replace(app, SCENE_THRESHOLDS={
            marker_segment_signature(s): s[SCENE_OVERRIDE]
            for s in records.values() if SCENE_OVERRIDE in s})
    return app, {source: project(app, record) for source, record in records.items()}


def as_frames(segment):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.storage.segments import StoredSegment
    if segment['mode'] != 'scene':
        return segment
    value = StoredSegment(mode='frames', items=segment['frames'], meta={**segment['meta'],
                          'marker_score_floor': segment['meta']['scene_score_floor']})
    value._evidence_digest = marker_segment_signature(segment)
    return value


def frame_records(scenes):
    """Scene qualification and occurrences use their own complete observations."""
    return {source: as_frames(record) for source, record in scenes.items()}
