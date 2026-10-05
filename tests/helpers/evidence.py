"""Shared offline helpers; no test collection or fixture registration."""

from copy import deepcopy

import numpy as np

from stash_ai.domain import Cue, FramePrediction
from tests.helpers.runtime import runtime as app
from stash_ai.models import SOURCES
from stash_ai.evidence.temporal.statistics import SceneAccumulator


def span(source, start, end, tag="common", score=.9, extras=()):
    return dict(source=source, tag=tag, start=start, end=end, score=score,
                hits=2, extra_tags=list(extras), meta={"origin":"test"})


def hit(source, label="common", score=.9):
    return dict(source=source,tag=label,score=score,meta={"origin":"test"})


def empty_policies():
    return {key:frozenset() for key in SOURCES}


def frames(rows, step=1):
    return [FramePrediction(Cue(i*step, (i+1)*step, "sprite", (0,0,1,1)), tuple(row)) for i,row in enumerate(rows)]


def segment(model, rows, identity='same'):
    # Synthetic evidence retains every supplied observation without filtering.
    return dict(mode='frames',meta=dict(origin='inference',analysis_format=4, scene_threshold=.8, scene_support_floor=.8 * app.TEMPORAL_POLICY.scene_support_ratio, marker_threshold=.8,marker_score_floor=0.,frame_input_signature=identity,
                inference_signature=model+'-'+identity),items=[dict(start=f.cue.start,end=f.cue.end,tags=list(f.tags)) for f in frames(rows)])


def policies(excluded=()):
    return {source: frozenset(excluded) for source in SOURCES}


def paired_records(values, threshold=.8, marker_threshold=.8):
    scenes, markers = {}, {}
    for source, scores in values.items():
        threshold_value = threshold[source] if isinstance(threshold, dict) else threshold
        data = frames([[('core', value)] if value else [] for value in scores])
        accumulator = SceneAccumulator(threshold_value, app.TEMPORAL_POLICY)
        accumulator.add(np.array(scores).reshape(-1, 1), [frame.cue for frame in data])
        stats = accumulator.statistics(['core'])
        marker = segment(source, [[('core', value)] if value >= marker_threshold * .85 else [] for value in scores])
        marker['meta'].update(scene_threshold=threshold_value, marker_threshold=marker_threshold,
                              marker_score_floor=marker_threshold * .85,
                              scene_support_floor=threshold_value * app.TEMPORAL_POLICY.scene_support_ratio)
        scene = dict(mode='scene', meta={**marker['meta'], 'scene_frames_format': 1, 'scene_score_floor': 0.},
                     frames=[dict(start=f.cue.start, end=f.cue.end, tags=list(f.tags)) for f in data],
                     items=[dict(tag='core', score=sum(sorted(scores, reverse=True)[:2]) / min(2, len(scores)),
                                 statistics=stats.get('core'))] if any(scores) else [])
        markers[source], scenes[source] = [marker], [scene]
    return scenes, markers


def current_record(value):
    """Build explicit complete observations for synthetic current-inference fixtures."""
    from stash_ai.signatures import digest
    meta = value['meta']
    if meta.get('origin') != 'inference':
        return value
    meta.pop('marker_tag_blocklist', None)
    meta.update(analysis_format=4)
    meta.setdefault('inference_signature', digest(value))
    meta.setdefault('frame_input_signature', 'fixture-input')
    meta.setdefault('scene_threshold', .4)
    meta.setdefault('marker_threshold', .8)
    meta.setdefault('marker_score_floor', 0.)
    meta.setdefault('scene_support_floor', meta['scene_threshold'] * app.TEMPORAL_POLICY.scene_support_ratio)
    if value['mode'] == 'scene':
        meta.update(scene_frames_format=1)
        meta.setdefault('scene_score_floor', 0.)
        if 'frames' not in value:
            value['frames'] = [dict(start=float(i), end=float(i + 1),
                tags=[[item['tag'], item['score']] for item in value['items']]) for i in range(2)]
        accumulator = SceneAccumulator(meta['scene_threshold'], app.TEMPORAL_POLICY)
        labels = [item['tag'] for item in value['items']]
        indexes = {tag: i for i, tag in enumerate(labels)}
        for row in value['frames']:
            accumulator.add_sparse(((indexes[tag], score) for tag, score in row['tags'] if tag in indexes),
                Cue(row['start'], row['end'], '', (0, 0, 1, 1)))
        statistics = accumulator.statistics(labels)
        for item in value['items']:
            item.setdefault('statistics', statistics.get(item['tag']))
    return value


def inject_records(store, scene, model, outputs):
    """Emulate an old writer or on-disk corruption, outside today's write contract."""
    from stash_ai.storage.metadata import FORMAT, summarize, encode
    from stash_ai.storage.progress import index_result
    from stash_ai.storage.thresholds import separate
    for kind, value in outputs.items():
        segments = value if isinstance(value, list) else [value]
        params = store.scope, scene, model, kind
        raw, thresholds = separate(segments, kind)
        with store.connection:
            store.connection.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?,?)',
                                     (*params, store.pack(raw)))
            store.connection.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?', params)
            if thresholds is not None:
                store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)', (*params, thresholds))
            store.connection.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                                     (*params, FORMAT, encode(summarize(raw))))
            index_result(store, scene, model, kind, segments)
    store.cache.clear()




def seed_history(store, scene, model, outputs, replace=None):
    """Construct historical multi-segment fixtures outside the inference API."""
    import json
    with store.connection:
        for kind, segment in outputs.items():
            previous = store.model_records(scene, kind, model) if replace is not None and not replace.get(kind, True) else []
            def semantic(value):
                value = deepcopy(value)
                value['meta'].pop('captured_at', None)
                return json.dumps(value, sort_keys=True)
            previous = [value for value in previous if semantic(value) != semantic(segment)]
            store.replace_records(scene, model, kind, previous + [segment])
