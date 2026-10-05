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
    return dict(mode='frames',meta=dict(origin='inference',marker_threshold=.8,marker_score_floor=0.,marker_tag_blocklist=[],frame_input_signature=identity,
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
