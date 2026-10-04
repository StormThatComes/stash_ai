"""Replay scene qualification from sparse frame evidence and exact aggregates."""
from types import SimpleNamespace

from stash_ai.evidence.identity import SCENE_OVERRIDE


def scene_threshold(segment):
    return segment.get(SCENE_OVERRIDE, segment['meta'].get('scene_threshold'))


def validate_scene(app, segment, threshold, *, has_frames=None):
    from stash_ai.evidence.thresholds import _score
    meta = segment['meta']
    needed = threshold * (app.TEMPORAL_POLICY.scene_support_ratio if app.TEMPORAL_POLICY.enabled else 1.0)
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
    threshold = (app.SCENE_THRESHOLDS.get(signature, meta.get('scene_threshold'))
                 if hasattr(app, 'SCENE_THRESHOLDS') else scene_threshold(segment))
    # Temporal statistics depend on the support floor as well as the high
    # threshold. Old aggregates without retention metadata keep their original
    # behavior; frame-backed results can also gain statistics after --no-temporal.
    support_changed = (app.TEMPORAL_POLICY.enabled
        and ('scene_support_floor' in meta or meta.get('scene_frames_format') == 1)
        and meta.get('scene_support_floor') != threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    if threshold == meta.get('scene_threshold') and not support_changed:
        return segment
    validate_scene(app, segment, threshold)
    items = segment['items']
    meta = {**meta, 'scene_threshold': threshold}
    if app.TEMPORAL_POLICY.enabled:
        labels = [item['tag'] for item in items]
        indexes = {tag: i for i, tag in enumerate(labels)}
        accumulator = SceneAccumulator(threshold, app.TEMPORAL_POLICY)
        for row in segment['frames']:
            accumulator.add_sparse(((indexes[tag], score) for tag, score in row['tags']),
                                   Cue(row['start'], row['end'], '', (0, 0, 1, 1)))
        statistics = accumulator.statistics(labels)
        items = [{**item, 'statistics': statistics[item['tag']]} for item in items if item['tag'] in statistics]
        meta['scene_support_floor'] = threshold * app.TEMPORAL_POLICY.scene_support_ratio
    result = StoredSegment({**segment, 'meta': meta, 'items': items})
    result._evidence_digest, result._scene_projected = signature, True
    return result


def views(app, records):
    from stash_ai.evidence.identity import marker_segment_signature
    if not hasattr(app, 'SCENE_THRESHOLDS'):
        app = SimpleNamespace(**vars(app), SCENE_THRESHOLDS={
            marker_segment_signature(s): s[SCENE_OVERRIDE]
            for segments in records.values() for s in segments if SCENE_OVERRIDE in s})
    return app, {source: [project(app, s) for s in segments] for source, segments in records.items()}


def as_frames(segment):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.storage.segments import StoredSegment
    if segment['mode'] != 'scene':
        return segment
    value = StoredSegment(mode='frames', items=segment['frames'], meta={**segment['meta'],
                          'marker_score_floor': segment['meta']['scene_score_floor']})
    value._evidence_digest = marker_segment_signature(segment)
    return value


def frame_records(scenes, markers):
    """Prefer complete scene observations; retain old marker evidence as fallback."""
    result = {}
    for source in scenes.keys() | markers.keys():
        full = [s for s in scenes.get(source, []) if s['meta'].get('scene_frames_format') == 1 and 'frames' in s]
        runs = {s['meta'].get('inference_signature') for s in full}
        result[source] = [as_frames(s) for s in full] + [s for s in markers.get(source, [])
            if s['meta'].get('inference_signature') not in runs]
    return result
