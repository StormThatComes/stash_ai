"""Representative points from accepted scene evidence, without another GPU pass.

Only one point per missing label is retained. Frame streams are merged by time:
memory is bounded by one sampled instant plus the best point for each label.
Model scores remain separate; ranking is not an ensemble probability.
"""
from __future__ import annotations

from collections import defaultdict
import heapq
from itertools import groupby
import math

from stash_ai.models import SOURCES


def _number(value, low=0.0, high=float('inf')):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _run(meta):
    return meta['inference_signature'], meta['frame_input_signature']


def _sources(hit):
    return hit.get('support_sources') or [hit['source']]


def _matches(hit, witness):
    if hit.get('derived') == 'shared_scene_frames':
        return witness.get('evidence_signature') == hit['scene_context'].get(witness['source'])
    return hit['source'] == witness['source'] and _run(hit['meta']) == _run(witness['meta'])


def _point(tag, time, witnesses, scene_evidence):
    sources = [source for source in SOURCES if any(w['source'] == source for w in witnesses)]
    return dict(tag=tag, start=time, end=None, source=sources[0], support_sources=sources[1:],
                score=None, hits=1, extra_tags=[], derived='occurrence', meta={'origin': 'occurrence'},
                occurrence_evidence=witnesses, scene_evidence=scene_evidence)


def _rank(witnesses):
    # Prefer agreement at this same input/time, then the margin above each
    # model's own scene threshold. Never average uncalibrated model scores.
    high = sum(w['score'] >= w['meta']['scene_threshold'] for w in witnesses)
    margin = max((w['score'] - w['meta']['scene_threshold']) /
                 max(1.0 - w['meta']['scene_threshold'], 1e-12) for w in witnesses)
    return high, len(witnesses), margin


def _observations(source, segment, signature, qualified, policy):
    meta = segment['meta']
    threshold = meta['scene_threshold']
    floor = threshold * policy.scene_support_ratio
    for row in segment['items']:
        values = [(tag, score, qualified[tag]) for tag, score in row['tags']
                  if tag in qualified and score >= floor]
        if values:
            yield row['start'], source, meta, signature, 'frame', values


def _candidate_streams(hits, records, blocked, policy):
    """Merge retained peaks and compatible frame streams without dense matrices."""
    by_run, by_frame = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    peaks = []
    for hit in hits:
        tag, source, meta = hit['tag'], hit['source'], hit['meta']
        if hit.get('derived') == 'shared_scene_frames':
            for member, signature in hit['scene_context'].items():
                if tag not in blocked[member]:
                    by_frame[member, signature][tag].append(hit)
            continue
        if tag in blocked[source]:
            continue
        by_run[source, _run(meta)][tag].append(hit)
        stats = hit.get('statistics') or {}
        time, score, threshold = stats.get('peak_time'), stats.get('max_score'), meta['scene_threshold']
        if (_number(time) and _number(score, high=1)
                and score >= threshold * policy.scene_support_ratio):
            peaks.append((time, source, meta, None, 'scene_peak', [(tag, score, [hit])]))

    streams = [iter(sorted(peaks, key=lambda event: event[0]))]
    from stash_ai.evidence.identity import marker_segment_signature
    for source, segment in records.items():
        if segment['mode'] != 'frames':
            continue
        signature = marker_segment_signature(segment)
        qualified = defaultdict(list)
        for mapping in (by_run.get((source, _run(segment['meta'])), {}), by_frame.get((source, signature), {})):
            for tag, proofs in mapping.items():
                if tag not in blocked[source]:
                    qualified[tag].extend(proofs)
        if qualified:
            streams.append(_observations(source, segment, signature, qualified, policy))
    return streams


def _best_points(streams):
    """Choose one deterministic point per label from one instant at a time."""
    best = {}
    for time, events in groupby(heapq.merge(*streams, key=lambda event: event[0]), key=lambda event: event[0]):
        groups = defaultdict(dict)
        for _, source, meta, signature, kind, values in events:
            identity = meta['frame_input_signature']
            for tag, score, proofs in values:
                group = groups[tag, identity]
                witness = dict(source=source, score=score, meta=meta, evidence_kind=kind,
                               evidence_signature=signature)
                old = group.get(source)
                if old is None or score > old[0]['score']:
                    group[source] = witness, proofs
        for (tag, _), values in groups.items():
            witnesses = [values[s][0] for s in SOURCES if s in values]
            ranking = (*_rank(witnesses), -time)
            if tag in best and ranking <= best[tag][0]:
                continue
            proofs, seen = [], set()
            for _, contributions in values.values():
                for proof in contributions:
                    if id(proof) not in seen:
                        seen.add(id(proof))
                        proofs.append(proof)
            best[tag] = ranking, _point(tag, time, witnesses, proofs)
    return best


def occurrence_contributions(app, scene_hits, records, spans, core_policies, tag_policies, metrics=None):
    policy = app.TEMPORAL_POLICY
    if not policy.occurrence_markers:
        return []
    from stash_ai.evidence.formats import validate_current
    for record in records.values():
        validate_current(record)
    covered = {item['tag'] for item in spans}
    blocked = {s: set(core_policies[s]) | set(tag_policies[s]) for s in SOURCES}
    hits = [hit for hit in scene_hits if hit['tag'] not in covered and hit['meta'].get('origin') == 'inference'
            and any(hit['tag'] not in blocked[s] for s in _sources(hit))]
    missing = {hit['tag'] for hit in hits}
    if not missing:
        return []

    streams = _candidate_streams(hits, records, blocked, policy)
    best = _best_points(streams)

    if metrics is not None:
        metrics['occurrence_candidates'] += len(best)
        metrics['occurrence_missing_timestamps'] += len(missing - set(best))
    return sorted((point for _, point in best.values()), key=lambda item: (item['start'], item['tag']))


def cleanup_occurrence(app, store, scene_id, point, selected, cores, tags):
    """Revoke published witnesses without moving the point or adding new votes."""
    from stash_ai.evidence.scenes import cleanup_scene_support
    hits = cleanup_scene_support(app, store, scene_id, point['scene_evidence'], selected, tags)
    witnesses = [w for w in point['occurrence_evidence']
                 if (w['source'] not in selected or point['tag'] not in cores[w['source']] | tags[w['source']])
                 and any(_matches(hit, w) for hit in hits)]
    if not witnesses:
        return None
    return _point(point['tag'], point['start'], witnesses,
                  [hit for hit in hits if any(_matches(hit, w) for w in witnesses)])
