"""Replay scene qualification from sparse frame evidence and exact aggregates."""
from types import SimpleNamespace

SCENE_OVERRIDE = 'scene_threshold_override'


def scene_threshold(segment):
    return segment.get(SCENE_OVERRIDE, segment['meta'].get('scene_threshold'))


def validate_scene(app, segment, threshold):
    from stash_thresholds import _score
    meta = segment['meta']
    needed = threshold * (app.TEMPORAL_POLICY.scene_support_ratio if app.TEMPORAL_POLICY.enabled else 1.0)
    floor = meta.get('scene_score_floor')
    if (segment['mode'] != 'scene' or meta.get('origin') != 'inference'
            or meta.get('scene_frames_format') != 1 or 'frames' not in segment):
        raise ValueError('scene frame evidence is missing; run inference once to populate the cache.')
    if not _score(floor) or floor > needed:
        raise ValueError(f'scene cache floor {floor!r} exceeds required {needed:g}; run inference.')


def project(app, segment):
    """A transient view; original scores, statistics and provenance stay immutable."""
    from stash_domain import Cue
    from stash_merge import marker_segment_signature
    from stash_performance import StoredSegment
    from stash_temporal import SceneAccumulator
    if getattr(segment, '_scene_projected', False):
        return segment
    signature = marker_segment_signature(segment)
    threshold = (app.SCENE_THRESHOLDS.get(signature, segment['meta'].get('scene_threshold'))
                 if hasattr(app, 'SCENE_THRESHOLDS') else scene_threshold(segment))
    if threshold == segment['meta'].get('scene_threshold'):
        return segment
    validate_scene(app, segment, threshold)
    items = segment['items']
    if app.TEMPORAL_POLICY.enabled:
        labels = [item['tag'] for item in items]
        indexes = {tag: i for i, tag in enumerate(labels)}
        accumulator = SceneAccumulator(threshold, app.TEMPORAL_POLICY)
        for row in segment['frames']:
            accumulator.add_sparse(((indexes[tag], score) for tag, score in row['tags']),
                                   Cue(row['start'], row['end'], '', (0, 0, 1, 1)))
        statistics = accumulator.statistics(labels)
        items = [{**item, 'statistics': statistics[item['tag']]} for item in items if item['tag'] in statistics]
    result = StoredSegment({**segment, 'meta': {**segment['meta'], 'scene_threshold': threshold}, 'items': items})
    result._evidence_digest, result._scene_projected = signature, True
    return result


def views(app, records):
    from stash_merge import marker_segment_signature
    if not hasattr(app, 'SCENE_THRESHOLDS'):
        app = SimpleNamespace(**vars(app), SCENE_THRESHOLDS={
            marker_segment_signature(s): s[SCENE_OVERRIDE]
            for segments in records.values() for s in segments if SCENE_OVERRIDE in s})
    return app, {source: [project(app, s) for s in segments] for source, segments in records.items()}


def as_frames(segment):
    from stash_merge import marker_segment_signature
    from stash_performance import StoredSegment
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


def plan(app, store, scene, config, *, scenes=True, markers=False):
    from stash_thresholds import OVERRIDE, validate
    result = {}
    if scenes:
        segments = store.records(scene, 'scene', config.source.key, borrow=True).get(config.source.key, [])
        if not segments:
            raise ValueError('no scene results; run inference for this scene/model.')
        for segment in segments:
            validate_scene(app, segment, config.scene_threshold)
        result['scene'] = segments
    if markers:
        result['markers'] = validate(app, store, scene, config, check_scene=not scenes)
    updated = {}
    for kind, segments in result.items():
        key, meta_key, threshold = ((SCENE_OVERRIDE, 'scene_threshold', config.scene_threshold) if kind == 'scene'
                                   else (OVERRIDE, 'marker_threshold', config.marker_threshold))
        values = []
        changed = False
        for segment in segments:
            effective = segment.get(key, segment['meta'][meta_key])
            if threshold == effective:
                # Already at the target policy; leave the stored segment alone.
                values.append(segment)
                continue
            value = dict(segment)
            value.pop(key, None)
            if threshold != segment['meta'][meta_key]:
                value[key] = threshold
            values.append(value)
            changed = True
        if changed:
            updated[kind] = values
    if updated and any(store.snapshot(scene, k, borrow=True)[1] is not None for k in ('scene', 'markers')):
        raise RuntimeError('Finish the pending merge or use --abandon-pending before changing thresholds.')
    return updated


def apply(store, scene, model, outputs):
    with store.connection:
        for kind, segments in outputs.items():
            store.replace_records(scene, model, kind, segments)


def rethreshold(app, store, scenes, settings, report, *, markers=False):
    errors = 0
    for scene in scenes:
        for config in settings:
            try:
                plan(app, store, scene, config, markers=markers)
            except (ValueError, RuntimeError) as exc:
                report.issue('scene thresholds', config.source.key, scene, exc, str(exc))
                errors += 1
    if errors:
        return dict(errors=errors)
    changed = 0
    for scene in scenes:
        for config in settings:
            outputs = plan(app, store, scene, config, markers=markers)
            apply(store, scene, config.source.key, outputs)
            changed += bool(outputs)
    return dict(scene_thresholds_updated=changed)


def reuse(app, store, scene, config, identity, cues, hashes, kinds):
    from stash_progress import completed, digest
    from stash_thresholds import indexed_signature
    records = {kind: store.records(scene, kind, config.source.key, borrow=True).get(config.source.key, []) for kind in kinds}
    if any(not segments for segments in records.values()):
        return False
    for kind, segments in records.items():
        segment = segments[-1]
        meta = segment['meta']
        signature = digest({'model': {**identity, 'scene_threshold': meta.get('scene_threshold'),
                                     'marker_threshold': meta.get('marker_threshold')},
            'cues': [[c.start, c.end, c.image, list(c.rect)] for c in cues], 'media': hashes})
        if not completed(store, scene, config.source.key, [kind], indexed_signature(segment, signature)):
            return False
    if 'scene' not in records:
        return False
    try:
        outputs = plan(app, store, scene, config, markers='markers' in records)
    except ValueError:
        return False
    apply(store, scene, config.source.key, outputs)
    return True
