"""Duplicate cores, crowded-scene vetoes, stable selection and evidence contracts."""
from copy import deepcopy

import numpy as np
import pytest

from stash_ai.recognition import pose_deduplication as dedup
from stash_ai.recognition.pose_deduplication_config import defaults as policy
from stash_ai.recognition.config import defaults, validate
from tests.test_recognition_pmpose import fingerprints
from tests.test_recognition_pose_refinement import pose_points, prediction


BOXES = [[0, 0, 90, 100], [20, 0, 60, 100]]  # Box IoU 0.444: both survive NMS=0.7.


def duplicates():
    result = prediction([pose_points(), pose_points()])
    result[1][0] = .8
    return result


def test_duplicate_suppression_prefers_core_quality_and_preserves_inputs():
    values = duplicates()
    before = deepcopy(values)
    report = dedup.select(BOXES, values, policy(), audit=True)
    assert report['kept_indices'] == [1] and report['suppressed_by'] == [1, None]
    audit = report['audit']
    assert audit['detections'][0]['person_index'] is None
    assert audit['detections'][1]['person_index'] == 0
    pair = audit['comparisons'][0]
    assert pair['decision'] == 'suppress' and pair['reason'] == 'duplicate_core'
    assert pair['box_iou'] == pytest.approx(4/9)
    assert pair['matched_regions'] == ['head', 'shoulders', 'hips']
    assert len(pair['matched_indices']) == 9
    for index in (0, 1):
        np.testing.assert_array_equal(values[index], before[index])
    for key in values[2]:
        np.testing.assert_array_equal(values[2][key], before[2][key])


@pytest.mark.parametrize('indices,reason', [
    ([0, 1, 2, 3, 4], 'distinct_heads'), ([5, 6], 'distinct_torsos'),
    ([11, 12], 'distinct_torsos'), ([0], 'conflicting_core'),
])
def test_distinct_heads_or_torsos_and_conflicting_core_veto_suppression(indices, reason):
    values = duplicates()
    values[0][1, indices, 1] += 15
    report = dedup.select(BOXES, values, policy(), audit=True)
    assert report['kept_indices'] == [0, 1]
    assert report['audit']['comparisons'][0]['reason'] == reason


@pytest.mark.parametrize('kind,reason', [
    ('head_only', 'single_region_support'), ('limbs_only', 'insufficient_shared_core'),
    ('complementary', 'insufficient_shared_core'), ('collapsed', 'collapsed_core_support'),
    ('one_shared', 'insufficient_shared_core'), ('low_presence', 'insufficient_shared_core'),
    ('low_visibility', 'insufficient_shared_core'),
])
def test_insufficient_independent_support_keeps_both_people(kind, reason):
    values = duplicates()
    if kind == 'head_only':
        values[1][:, 5:] = .1
    elif kind == 'limbs_only':
        values[1][:, dedup.CORE] = .1
    elif kind == 'complementary':
        values[1][0, 5:] = .1
        values[1][1, :5] = .1
    elif kind == 'collapsed':
        values[0][:] = [35, 30]
    elif kind == 'one_shared':
        values[1][1, 1:] = .1
    else:
        values[2][kind.removeprefix('low_')][1] = .5  # Gate is strictly greater.
    report = dedup.select(BOXES, values, policy(), audit=True)
    assert report['kept_indices'] == [0, 1]
    assert report['audit']['comparisons'][0]['reason'] == reason


def test_reliable_coverage_beats_a_few_very_high_scores():
    values = duplicates()
    values[1][0] = .85
    values[1][1] = 1.
    values[1][1, [3, 4, 11, 12]] = 0
    assert dedup.select(BOXES, values, policy())['kept_indices'] == [0]


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('scale', [.5, 3])
def test_sideways_upside_down_and_scaled_duplicates_do_not_require_upright_anatomy(angle, scale):
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    transform = lambda xy: np.asarray(xy) @ rotation.T * scale + [400, 400]
    values = duplicates()
    values[0][:] = transform(values[0])
    boxes = []
    for left, top, right, bottom in BOXES:
        corners = transform([[left, top], [right, top], [right, bottom], [left, bottom]])
        boxes.append([*corners.min(axis=0), *corners.max(axis=0)])
    assert dedup.select(boxes, values, policy())['kept_indices'] == [1]


def test_no_transitive_suppression_through_a_discarded_candidate():
    # A~B and B~C, but A and C have distinct heads. B must not remove C.
    values = prediction([pose_points(), pose_points()+[4, 0], pose_points()+[8, 0]])
    values[1][:] = .9
    report = dedup.select([[0, 0, 90, 100]]*3, values, policy(), audit=True)
    assert report['kept_indices'] == [0, 2]
    assert report['suppressed_by'] == [None, 0, None]
    assert [p['detector_indices'] for p in report['audit']['comparisons']] == [[0, 1], [0, 2]]


def test_anatomical_labels_cannot_be_replaced_by_nearest_arbitrary_joints():
    values = duplicates()
    values[0][1, [5, 6]] = values[0][1, [6, 5]]
    assert dedup.select(BOXES, values, policy())['kept_indices'] == [0, 1]


def test_low_confidence_or_outside_points_cannot_supply_shared_core_support():
    values = duplicates()
    values[0][1, :5] = [200, 200]
    values[1][1, 5:7] = 0
    report = dedup.select(BOXES, values, policy(), audit=True)
    assert report['kept_indices'] == [0, 1]
    assert report['audit']['comparisons'][0]['shared_indices'] == [11, 12]


@pytest.mark.parametrize('bad', ['nan', 'score_range', 'missing_probability', 'shape', 'box'])
def test_bad_measurements_raise_before_any_candidate_can_be_hidden(bad):
    values = duplicates()
    boxes = deepcopy(BOXES)
    if bad == 'nan': values[0][0, 0, 0] = np.nan
    elif bad == 'score_range': values[1][0, 0] = 1.1
    elif bad == 'missing_probability': del values[2]['visibility']
    elif bad == 'shape': values = (values[0][:1], values[1], values[2])
    else: boxes[0][2] = 0
    with pytest.raises(ValueError, match='deduplication requires'):
        dedup.select(boxes, values, policy())


def test_audit_pair_details_are_bounded_and_computation_budget_fails_explicitly(monkeypatch):
    values = prediction([pose_points(), pose_points()+[15, 0], pose_points()+[30, 0]])
    boxes = [[0, 0, 100, 100]]*3
    monkeypatch.setattr(dedup, 'MAX_AUDIT_KEPT_PAIRS', 1)
    report = dedup.select(boxes, values, policy(), audit=True)['audit']
    assert report['comparison_count'] == 3
    assert len(report['comparisons']) == 1 and report['omitted_kept_pairs'] == 2
    assert sum(report['reason_counts'].values()) == 3
    monkeypatch.setattr(dedup, 'MAX_COMPARISONS', 1)
    with pytest.raises(ValueError, match='comparison budget'):
        dedup.select(boxes, values, policy())


def test_empty_single_and_nonoverlapping_frames_are_well_defined():
    for n in (0, 1):
        result = dedup.select(BOXES[:n], prediction([pose_points()]*n), policy(), audit=True)
        assert result['kept_indices'] == list(range(n))
        assert result['audit']['comparisons'] == []
    values = prediction([pose_points(), pose_points()+[100, 0]])
    report = dedup.select([BOXES[0], [100, 0, 190, 100]], values, policy(), audit=True)
    assert report['kept_indices'] == [0, 1]
    assert report['audit']['reason_counts'] == {'no_box_overlap': 1}


@pytest.mark.parametrize('backend', ['yolo', 'vitpose_plus_large', 'sapiens', 'pmpose'])
def test_disabled_policy_preserves_previous_contracts(backend):
    data = defaults()
    data['sapiensid']['pose']['backend'] = backend
    expected = fingerprints(data)
    data['sapiensid']['pose'].pop('deduplication')
    assert fingerprints(data) == expected


def test_all_active_settings_invalidate_person_body_and_identity_contracts():
    data = defaults()
    pose = data['sapiensid']['pose']
    pose['backend'] = 'pmpose'
    old = fingerprints(data)
    pose['deduplication']['enabled'] = True
    active = fingerprints(data)
    assert all(a != b for a, b in zip(old, active))
    for key, value in dict(min_keypoint_score=.6, min_presence=.6, min_visibility=.6,
                           min_shared_points=5, max_point_distance=.04, min_match_fraction=.9).items():
        original = pose['deduplication'][key]
        pose['deduplication'][key] = value
        assert all(a != b for a, b in zip(active, fingerprints(data)))
        pose['deduplication'][key] = original


@pytest.mark.parametrize('key,value', [('enabled', 1), ('min_shared_points', 3), ('min_shared_points', 10),
    ('min_presence', True), ('min_visibility', float('nan')), ('max_point_distance', 0),
    ('max_point_distance', .5), ('min_match_fraction', .4)])
def test_invalid_settings_are_rejected(key, value):
    data = defaults()
    data['sapiensid']['pose']['deduplication'][key] = value
    with pytest.raises(ValueError, match='pose.deduplication'):
        validate(data)


def test_deduplication_requires_pmpose():
    data = defaults()
    data['sapiensid']['pose']['deduplication']['enabled'] = True
    with pytest.raises(ValueError, match='requires pose.backend: pmpose'):
        validate(data)
