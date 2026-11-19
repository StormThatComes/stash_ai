"""Detached per-joint measurements and precise output-filter diagnostics."""
from dataclasses import replace
import math


def score_kind(backend):
    if backend == 'pmpose':
        return 'expected OKS'
    return 'heatmap score' if backend in ('vitpose_plus_large', 'sapiens') else 'confidence'


def probability_kind(backend, probabilities):
    if not probabilities:
        return 'unavailable'
    if backend == 'pmpose':
        return 'measured'
    return 'confidence proxy' if backend else 'unknown'


def valid_xy(point):
    return len(point) == 2 and all(math.isfinite(v) and v >= 0 for v in point)


def output_person(person, index, *, raw=None, settings=None):
    from .detection_phases import Person, _details, _xy
    points = _xy(person['keypoints'])
    coordinates = _xy(raw) if raw is not None else points
    probabilities = tuple((key, tuple(float(v) for v in person['keypoint_' + key]))
                          for key in ('presence', 'visibility', 'flip_agreement')
                          if 'keypoint_' + key in person)
    kind = person.get('keypoint_score_kind', 'unavailable').replace('_', ' ')
    if kind == 'expected oks':
        kind = 'expected OKS'
    row = Person(f'P{index}', tuple(person['bounds']), points,
                 scores=tuple(float(v) for v in person.get('keypoint_scores', ())),
                 probabilities=probabilities, score_kind=kind,
                 probability_kind='measured' if person.get('keypoint_visibility_measured') else 'unavailable',
                 raw_points=coordinates, states=tuple(person.get('keypoint_states', ())))
    row = _details(row, person.get('pose_refinement', {}))
    return replace(row, point_reasons=tuple(
        rejection_reasons(row, k, settings) if not valid_xy(point) else () for k, point in enumerate(points)))


def rejection_reasons(person, index, settings):
    raw = person.raw_points[index]
    reasons = []
    if not valid_xy(raw):
        reasons.append('No usable coordinates from the preceding stage')
    else:
        x, y = raw
        left, top, right, bottom = person.bounds
        if not (left <= x < right and top <= y < bottom):
            reasons.append(f'Outside output box: {left:g} <= x < {right:g}, {top:g} <= y < {bottom:g}')
    if settings is not None:
        gates = [('score', person.scores, 'keypoint_score_threshold', 'pose.estimator.keypoint_score_threshold')]
        if settings.get('backend') == 'pmpose':
            probabilities = dict(person.probabilities)
            gates += [(key, probabilities.get(key, ()), 'pmpose_' + key + '_threshold',
                       'pose.estimator.pmpose.' + key + '_threshold') for key in ('presence', 'visibility')]
        for name, values, key, path in gates:
            threshold = settings.get(key, 0)
            if index < len(values) and (name == 'score' or threshold > 0) and values[index] <= threshold:
                reasons.append(f'{path}: {values[index]:.4f} <= {threshold:g}')
    return tuple(reasons or ['Unavailable at this stage'])


def final_people(record, history=None):
    """Final geometry is authoritative, including accepted crowded-box retries."""
    rows = []
    for i, person in enumerate(record['people']):
        row = output_person(person, i)
        # A final result may differ from the last recorded phase. Reuse removed
        # coordinates/reasons only when the complete output evidence matches.
        matches = [p for p in history.people if p.bounds == row.bounds and p.points == row.points
                   and p.scores == row.scores and p.probabilities == row.probabilities
                   and p.point_reasons] if history is not None else []
        if len(matches) == 1:
            row = replace(row, raw_points=matches[0].raw_points, point_reasons=matches[0].point_reasons)
        rows.append(row)
    return tuple(rows)
