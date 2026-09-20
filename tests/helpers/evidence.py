"""Shared offline helpers; no test collection or fixture registration."""
from copy import deepcopy

import numpy as np

import stash_ai as app
from stash_taggers import SOURCES
from stash_temporal import SceneAccumulator


def span(source, start, end, tag="common", score=.9, extras=()):
    return dict(source=source, tag=tag, start=start, end=end, score=score,
                hits=2, extra_tags=list(extras), meta={"origin":"test"})


def hit(source, label="common", score=.9):
    return dict(source=source,tag=label,score=score,meta={"origin":"test"})


def empty_policies():
    return {key:frozenset() for key in SOURCES}


def frames(rows, step=1):
    return [app.FramePrediction(app.Cue(i*step, (i+1)*step, "sprite", (0,0,1,1)), tuple(row)) for i,row in enumerate(rows)]


def segment(model, rows, identity='same'):
    return dict(mode='frames',meta=dict(origin='inference',marker_threshold=.8,frame_input_signature=identity,
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
                              marker_score_floor=marker_threshold * .85, analysis_format=2)
        scene = dict(mode='scene', meta=deepcopy(marker['meta']), items=[dict(tag='core',
            score=sum(sorted(scores, reverse=True)[:2]) / min(2, len(scores)), statistics=stats['core'])] if stats else [])
        markers[source], scenes[source] = [marker], [scene]
    return scenes, markers
