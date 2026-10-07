"""Current, validated synthetic evidence; no models, network or test fixtures."""
from stash_ai.domain import Cue
from stash_ai.evidence.formats import ANALYSIS_FORMAT, validate_current
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.evidence.temporal.statistics import SceneAccumulator


def record(source, rows, *, scene=False, threshold=.8, identity='benchmark'):
    meta = dict(origin='inference', analysis_format=ANALYSIS_FORMAT,
                scene_threshold=threshold, marker_threshold=threshold,
                marker_score_floor=.1, scene_support_floor=threshold * .85,
                frame_input_signature=identity, inference_signature=source + '-' + identity)
    value = dict(mode='frames', meta=meta, items=rows)
    if scene:
        meta['scene_score_floor'] = .1
        labels = sorted({tag for row in rows for tag, _ in row['tags']})
        indexes = {tag: i for i, tag in enumerate(labels)}
        scores = {tag: [] for tag in labels}
        accumulator = SceneAccumulator(threshold, TemporalPolicy())
        for row in rows:
            accumulator.add_sparse(((indexes[tag], score) for tag, score in row['tags']),
                                   Cue(row['start'], row['end'], '', (0, 0, 1, 1)))
            for tag, score in row['tags']:
                scores[tag].append(score)
        stats = accumulator.statistics(labels)
        value = dict(mode='scene', meta=meta, frames=rows, items=[
            dict(tag=tag, score=sum(sorted(scores[tag], reverse=True)[:2]) / min(2, len(scores[tag])),
                 statistics=stats.get(tag)) for tag in labels])
    validate_current(value)
    return value


def synthetic_records(sources, frames, labels):
    rows = [dict(start=float(i), end=float(i+1), tags=[(f'tag_{t}', .9)
            for t in range(labels) if (i // (4+t % 7)+t) % 4 != 0]) for i in range(frames)]
    return {source: record(source, rows) for source in sources}
