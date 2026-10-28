import math

import pytest

from stash_ai.recognition.merge_report_policy import parameter, patched
from stash_ai.recognition.merge_solver_domains import registry, normalize, dimensions
from stash_ai.recognition.merge_solver_bidirectional import relaxed_seeds
from stash_ai.recognition.merge_solver_constraints import learned_groups, interaction_proposals
from stash_ai.recognition.threshold_boundaries import at_or_below, at_or_above, strictly_above, sides
from benchmarks.benchmark_merge_solver import fixture


@pytest.mark.parametrize('value', [0., .1, .000249, .431237, .799999, .8, 1., math.nextafter(.8, 0), math.nextafter(.8, 1)])
def test_boundary_proposals_survive_float_and_yaml_precision(value):
    import yaml
    low, high = at_or_below(value), strictly_above(value)
    assert low <= value < high
    assert at_or_below(low) == low
    assert at_or_above(value) >= value
    assert yaml.safe_load(yaml.safe_dump([low, high])) == [low, high]
    assert all(0 <= v <= 1 for v in sides(value))
    assert low in sides(value)


def test_domains_project_correlated_quality_and_matching_gates(tmp_path):
    config, _ = fixture(tmp_path, 'quality')
    patch = normalize(config, {
        'recognition.match_threshold': .3, 'clustering.cohesion_threshold': .8,
        'clustering.min_anchor_face_pixels': 1, 'clustering.min_auxiliary_face_pixels': 200,
        'clustering.view_min_support': 1, 'clustering.same_scene_match_bonus': .5,
        'clustering.max_anchor_pose_deviation': 1.,
    })
    variant = patched(config, patch)
    assert variant.data['clustering']['cohesion_threshold'] <= variant.data['recognition']['match_threshold']
    assert variant.data['clustering']['min_auxiliary_face_pixels'] == variant.data['clustering']['min_anchor_face_pixels']
    assert variant.data['clustering']['min_anchor_face_pixels'] >= config.data['detection']['min_face_pixels']
    assert variant.data['clustering']['view_min_support'] == 2
    assert variant.data['clustering']['same_scene_match_bonus'] == .1
    assert variant.data['clustering']['max_anchor_pose_deviation'] <= config.data['detection']['max_pose_deviation']
    with pytest.raises(ValueError): normalize(config, {'models.recognizer': 'adaface_ir50'})


def test_projection_preserves_acceptance_side_for_measured_quality(tmp_path):
    config, _ = fixture(tmp_path, 'quality')
    patch = normalize(config, {'clustering.min_anchor_face_pixels': 81.1234567,
                              'clustering.max_anchor_pose_deviation': .4539017504})
    assert patch['clustering.min_anchor_face_pixels'] <= 81.1234567
    assert patch['clustering.max_anchor_pose_deviation'] >= .4539017504
    patched(config, patch)


def test_seed_values_follow_measurements_not_fixed_thresholds(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    key = 'sapiensid.matching.face_floor'
    first = list(relaxed_seeds(config, {'options': {key: [.123456, .678901]}}))
    second = list(relaxed_seeds(config, {'options': {key: [.234567, .678901]}}))
    assert first[0][key] == .123456 and second[0][key] == .234567
    for patch in first+second:
        patched(config, patch)


def test_rotating_tpe_dimensions_reach_beyond_eight_active_keys(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    options = {k: [parameter(config.data, k)] for k in registry(config) if parameter(config.data, k) is not None}
    active = {k: i for i, k in enumerate(options)}
    selected = set()
    for sequence in range(15, 15+5*len(options), 5):
        keys = dimensions(config, options, active, sequence)
        assert len(keys) == 8
        selected.update(keys)
    assert selected == set(options)


def test_cross_family_interactions_come_from_shared_blocked_observations(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    face, pose = 'sapiensid.matching.face_floor', 'clustering.max_anchor_pose_deviation'
    feedback = dict(provenance=dict(blocker_trace=[dict(observation='q', parameters=[face])]),
                    quality=[dict(observation='q', patch={pose:.4})])
    assert learned_groups(feedback) == [tuple(sorted((face, pose)))]
    proposals = list(interaction_proposals(config, {}, {face:[.43], pose:[.4]}, feedback))
    assert proposals and any(p.get(face) == .43 and p.get(pose) == .4 for p in proposals)
    for patch in proposals:
        patched(config, patch)
    feedback['quality'][0]['observation'] = 'unrelated'
    assert not learned_groups(feedback)
