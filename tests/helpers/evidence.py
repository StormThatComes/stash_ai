"""Shared offline helpers; no test collection or fixture registration."""

from copy import deepcopy

import numpy as np

from stash_ai.domain import Cue, FramePrediction
from tests.helpers.runtime import runtime as app
from stash_ai.models import SOURCES
from stash_ai.evidence.temporal.statistics import SceneAccumulator


def span(source, start, end, tag="common", score=.9, extras=()):
    from stash_ai.evidence.temporal.policy import build_options
    from stash_ai.signatures import digest
    options = {**build_options(app), 'core_exclusions': {s: [] for s in SOURCES},
               'tag_exclusions': {s: [] for s in SOURCES}}
    support = {label: dict(intervals=[[start, end]], sample_times=[start, (start+end)/2])
               for label in extras if end is not None and end > start}
    return dict(source=source, tag=tag, start=start, end=end, score=score,
                hits=2, extra_tags=list(extras), meta={"origin":"test"},
                build_options=options, evidence_signature=digest([source, start, end, tag]),
                extra_support=support)


def hit(source, label="common", score=.9):
    return dict(source=source,tag=label,score=score,meta={"origin":"test"})


def empty_policies():
    return {key:frozenset() for key in SOURCES}


def frames(rows, step=1):
    return [FramePrediction(Cue(i*step, (i+1)*step, "sprite", (0,0,1,1)), tuple(row)) for i,row in enumerate(rows)]


def segment(model, rows, identity='same'):
    # Synthetic evidence retains every supplied observation without filtering.
    return dict(mode='frames',meta=dict(origin='inference',analysis_format=5, scene_threshold=.8, scene_support_floor=.8 * app.TEMPORAL_POLICY.scene_support_ratio, marker_threshold=.8,marker_score_floor=0.,frame_input_signature=identity,
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
        scene = dict(mode='scene', meta={**marker['meta'], 'scene_score_floor': 0.},
                     frames=[dict(start=f.cue.start, end=f.cue.end, tags=list(f.tags)) for f in data],
                     items=[dict(tag='core', score=sum(sorted(scores, reverse=True)[:2]) / min(2, len(scores)),
                                 statistics=stats.get('core'))] if any(scores) else [])
        markers[source], scenes[source] = marker, scene
    return scenes, markers


def inference_record(*, mode, items, meta=None, frames=None):
    """Construct current test evidence from explicit raw scores/observations.

    This builder rejects legacy modes, origins and sentinels. Migration tests
    construct historical payloads separately and use inject_records.
    """
    from stash_ai.evidence.formats import ANALYSIS_FORMAT, validate_current
    from stash_ai.signatures import digest
    metadata = dict(origin='inference', analysis_format=ANALYSIS_FORMAT,
        frame_input_signature='fixture-input', scene_threshold=.4,
        marker_threshold=.8, marker_score_floor=0.)
    metadata.update(meta or {})
    metadata.setdefault("scene_support_floor", metadata["scene_threshold"] * app.TEMPORAL_POLICY.scene_support_ratio)
    metadata.setdefault('inference_signature', digest(dict(mode=mode, items=items)))
    value = dict(mode=mode, meta=metadata, items=deepcopy(items))
    if mode == 'scene':
        metadata.setdefault('scene_score_floor', 0.)
        value['frames'] = deepcopy(frames) if frames is not None else [
            dict(start=float(i), end=float(i+1), tags=[(item['tag'], item['score']) for item in items])
            for i in range(2)]
        accumulator = SceneAccumulator(metadata['scene_threshold'], app.TEMPORAL_POLICY)
        labels = [item['tag'] for item in items]
        indexes = {tag: i for i, tag in enumerate(labels)}
        for row in value['frames']:
            accumulator.add_sparse(((indexes[tag], score) for tag, score in row['tags'] if tag in indexes),
                Cue(row['start'], row['end'], '', (0, 0, 1, 1)))
        statistics = accumulator.statistics(labels)
        for item in value['items']:
            item.setdefault('statistics', statistics.get(item['tag']))
    validate_current(value)
    import json
    return json.loads(json.dumps(value))



def inject_records(store, scene, model, outputs):
    """Emulate old writers or corruption outside today's active write contract."""
    from stash_ai.storage.metadata import FORMAT, summarize, encode
    from stash_ai.storage.progress import index_result
    from stash_ai.storage.thresholds import separate
    from tests.helpers import history as result_history
    for kind, value in outputs.items():
        params = store.scope, scene, model, kind
        old = isinstance(value, list)
        raw, threshold = (result_history.separate(value, kind) if old else separate(value, kind))
        version = 3 if old else FORMAT
        try:
            metadata = result_history.summarize(raw) if old else summarize(raw)
        except ValueError:
            # Deliberately corrupt authoritative evidence; force metadata repair.
            metadata, version = {}, 0
        with store.connection:
            store.connection.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?,?)', (*params, store.pack(raw)))
            store.connection.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?', params)
            if threshold is not None:
                store.connection.execute('PRAGMA ignore_check_constraints=ON')
                store.connection.execute('INSERT OR REPLACE INTO result_thresholds VALUES (?,?,?,?,?)', (*params, threshold))
                store.connection.execute('PRAGMA ignore_check_constraints=OFF')
            store.connection.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                                     (*params, version, encode(metadata)))
            if old:
                result_history.index_result(store, scene, model, kind, value)
            elif 'inference_signature' in value['meta']:
                index_result(store, scene, model, kind, value)
            else:
                # Deliberately incomplete provenance outside the current writer.
                store.connection.execute('INSERT OR REPLACE INTO result_index VALUES (?,?,?,?,?,?)',
                                         (*params, int(bool(value['items'])), None))
    store.cache.clear()
