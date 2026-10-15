import random

import pytest

from stash_ai.recognition.config import defaults
from stash_ai.recognition.quality import portrait_evaluation
from stash_ai.recognition.ranking import best_two
from stash_ai.recognition.visibility import CONTRACT_ID


def test_best_two_matches_stable_full_sort_with_ties_and_payloads():
    rng = random.Random(22)
    candidates = [(rng.choice((.4, .65, .9)), str(rng.randrange(20)), object()) for _ in range(2000)]
    ranked = []
    for index, candidate in enumerate(candidates):
        best_two(ranked, candidate)
        assert ranked == sorted(candidates[:index + 1], key=lambda item: (-item[0], item[1]))[:2]


def test_quality_evaluation_preserves_golden_scores_and_shortfalls():
    policy = defaults()['portraits']
    quality = {'pixels': 512, 'sharpness': 1000, 'pose_deviation': 0, 'exposure': 127.5,
               'clipped_fraction': 0, 'complete': True, 'landmarks_inside': True,
               'visible_face_fraction': 1., 'visibility': {'contract': CONTRACT_ID}}
    assert portrait_evaluation(quality, policy) == {'score': 1., 'quality_tiebreak': 1., 'quality_shortfalls': []}
    quality.update(pixels=48, sharpness=0, pose_deviation=1, visible_face_fraction=.4)
    result = portrait_evaluation(quality, policy)
    assert result['score'] == pytest.approx(.2171875)
    assert result['quality_tiebreak'] == pytest.approx(.2171875)
    assert result['quality_shortfalls'] == ['face_too_small', 'face_blurred', 'face_pose', 'face_occluded']


def test_expression_quality_is_validated_once_per_complete_evaluation(monkeypatch):
    import stash_ai.recognition.expression as expression
    policy = defaults()['portraits']
    policy['weights']['closed_mouth'] = 2.
    quality = {'pixels': 512, 'sharpness': 1000, 'pose_deviation': 0, 'exposure': 127.5,
               'clipped_fraction': 0, 'complete': True, 'landmarks_inside': True,
               'visible_face_fraction': 1., 'visibility': {'contract': CONTRACT_ID}}
    blendshapes = {name: 0. for name in expression.REQUIRED}
    blendshapes['jawOpen'] = .8
    quality.update(expression.measurement(blendshapes))
    calls, original = [], expression.cached
    def cached(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(expression, 'cached', cached)
    result = portrait_evaluation(quality, policy)
    assert result['score'] == pytest.approx(10.4 / 12)
    assert result['quality_tiebreak'] == 1. and result['quality_shortfalls'] == []
    assert len(calls) == 1
