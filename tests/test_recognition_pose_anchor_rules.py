"""Regional quality gates keep anatomical labels and replay existing evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import yaml

from stash_ai.recognition.config import defaults, load, plain
from stash_ai.recognition.person_anchor import anchor_gate
from stash_ai.recognition.person_geometry import body_points
from stash_ai.recognition.pose_anchor_rules import PARAMETER, POINTS, evaluate, validate


def metadata(missing=()):
    # Detector left labels are on image left: encoder canonicalization swaps them.
    points = [[50, 20], [40, 18], [60, 18], [30, 20], [70, 20],
              [30, 50], [70, 50], [20, 80], [80, 80], [10, 100], [90, 100],
              [35, 120], [65, 120], [35, 160], [65, 160], [35, 200], [65, 200]]
    for name in missing:
        points[POINTS.index(name)] = [-1, -1]
    return dict(person=dict(bounds=[0, 0, 100, 240], keypoints=points, score=.95),
                quality=dict(keypoints=17, sharpness=100, width=100, height=240, overlap=0))


def settings(rules=None):
    result = defaults()['sapiensid']['identities']
    if rules is not None:
        result['anchor_keypoint_rules'] = rules
    return result


@pytest.mark.parametrize('rules', [None, [], {'other': []}, {'all': {}}, {'any': None},
    {'all': [None]}, {'all': [{'group': 'neck', 'min': 1}]},
    {'all': [{'group': 'hips', 'points': ['left_hip'], 'min': 1}]},
    {'all': [{'group': 'hips'}]}, {'any': [{'group': 'eyes', 'min': 3}]},
    {'all': [{'group': 'hips', 'min': True}]}, {'all': [{'group': 'hips', 'min': 1.5}]},
    {'all': [{'group': 'hips', 'min': -1}]}, {'all': [{'points': [], 'min': 0}]},
    {'all': [{'points': ['left_eye', 'left_eye'], 'min': 2}]},
    {'all': [{'points': ['left_eye', 'mouth'], 'min': 1}]},
    {'all': [{'points': [None], 'min': 1}]}, {'all': [{'points': 'left_eye', 'min': 1}]},
    {'all': [{'group': 'hips', 'min': 1, 'extra': True}]},
])
def test_invalid_rules_fail_at_config_load(tmp_path, rules):
    path = tmp_path/'invalid.yml'
    path.write_text(yaml.safe_dump(dict(version=1, sapiensid=dict(identities=dict(anchor_keypoint_rules=rules)))))
    with pytest.raises(ValueError, match='anchor_keypoint_rules'):
        load(path)


def test_frozen_yaml_custom_points_and_empty_legacy_defaults(tmp_path):
    old = load(root=tmp_path)
    assert plain(old.data['sapiensid']['identities']['anchor_keypoint_rules']) == {'all': [], 'any': []}
    rules = {'all': [{'group': 'shoulders', 'min': 1}],
             'any': [{'points': ['left_hip', 'left_knee'], 'min': 2}, {'group': 'right_leg', 'min': 2}]}
    path = tmp_path/'rules.yml'
    path.write_text(yaml.safe_dump(dict(version=1, sapiensid=dict(identities=dict(anchor_keypoint_rules=rules)))))
    config = load(path)
    validate(config.data['sapiensid']['identities']['anchor_keypoint_rules'])
    assert anchor_gate(metadata(), config.data['sapiensid']['identities'])['eligible']


def test_all_any_and_global_count_are_conjunctive():
    rules = {'all': [{'group': 'shoulders', 'min': 1}, {'group': 'hips', 'min': 1}],
             'any': [{'group': 'left_leg', 'min': 3}, {'group': 'right_leg', 'min': 3}]}
    value = metadata(['left_knee', 'left_ankle'])
    gate = anchor_gate(value, settings(rules))
    assert gate['eligible'] and gate['keypoints'] == 15
    assert [r['passed'] for r in gate['keypoint_rules']['any']] == [False, True]
    value['person']['keypoints'][16] = [-1, -1]
    assert not anchor_gate(value, settings(rules))['eligible']
    value = metadata(['left_hip', 'right_hip'])
    gate = anchor_gate(value, settings(rules))
    assert gate['reason'] == 'body_anchor_keypoint_rules'
    assert gate['keypoint_rules']['all'][1]['measured'] == 0
    strict = settings({'all': [{'group': 'hips', 'min': 1}]})
    strict['anchor_min_keypoints'] = 17
    assert anchor_gate(metadata(['nose']), strict)['reason'] == 'body_anchor_keypoints'


def test_anatomical_labels_survive_swaps_crossed_limbs_and_missing_counterparts():
    value = metadata(['right_shoulder', 'right_elbow', 'right_wrist'])
    value['person']['keypoints'][9] = [95, 100]  # Left wrist crosses image midline.
    before = deepcopy(value)
    rules = {'all': [{'group': 'left_arm', 'min': 3}], 'any': []}
    gate = anchor_gate(value, settings(rules))
    assert gate['eligible']
    assert gate['keypoint_rules']['all'][0]['present'] == ['left_shoulder', 'left_elbow', 'left_wrist']
    assert not anchor_gate(value, settings({'all': [{'group': 'right_arm', 'min': 1}]}))['eligible']
    assert value == before
    canonical = body_points(value['person'])
    anatomical = body_points(value['person'], anatomical=True)
    assert sorted(canonical) == sorted(anatomical)
    assert canonical[6] == anatomical[5]  # Encoder output itself stays canonical.


def test_regional_counts_exclude_head_outliers_crop_edges_and_missing_points():
    value = metadata(['right_ear'])
    value['person']['keypoints'][9] = [100-1e-7, 100]  # float32 rounds to excluded crop edge.
    gate = anchor_gate(value, settings({'all': [{'points': ['left_wrist'], 'min': 1}]}))
    assert gate['keypoint_rules']['all'][0]['measured'] == 0
    value = metadata()
    value['person']['keypoints'][0] = [95, 20]  # Outside the ears/shoulders envelope.
    gate = anchor_gate(value, settings({'all': [{'group': 'head_without_eyes', 'min': 3}]}))
    assert gate['keypoint_rules']['all'][0]['present'] == ['left_ear', 'right_ear']
    assert gate['keypoints'] == 16 and not gate['eligible']


def test_overlapping_regions_do_not_inflate_global_count_and_back_view_needs_no_head():
    value = metadata(POINTS[:5])
    rules = {'all': [{'group': 'shoulders', 'min': 2}, {'group': 'torso', 'min': 4},
                     {'group': 'body', 'min': 12}]}
    gate = anchor_gate(value, settings(rules))
    assert gate['eligible'] and gate['keypoints'] == 12


def test_legacy_missing_geometry_and_zero_minimum_have_explicit_semantics():
    legacy = {'quality': {'keypoints': 17, 'sharpness': 100}}
    assert anchor_gate(legacy, settings())['eligible']
    rules = {'all': [{'group': 'hips', 'min': 1}]}
    gate = anchor_gate(legacy, settings(rules))
    assert gate['reason'] == 'body_anchor_keypoint_rules_missing'
    assert gate['keypoint_rules']['all'][0]['measured'] is None
    rules['all'][0]['min'] = 0
    assert anchor_gate(legacy, settings(rules))['eligible']
    # Any has a zero-minimum alternative, so missing evidence need not satisfy the other alternative.
    assert evaluate({'any': [{'group': 'hips', 'min': 2}, {'group': 'eyes', 'min': 0}]}, None)['eligible']


@pytest.mark.parametrize('geometry', [{}, {'bounds': [0, 0, 1, 1], 'keypoints': [[0, 0]]},
    {'bounds': [0, 0, 1, 1], 'keypoints': [[float('nan'), 0]]*17}])
def test_invalid_geometry_never_fabricates_regional_counts(geometry):
    value = dict(person=geometry, quality=dict(keypoints=17, sharpness=100))
    gate = anchor_gate(value, settings({'all': [{'group': 'hips', 'min': 1}]}))
    assert not gate['eligible']
    assert gate['keypoint_rules']['all'][0]['measured'] is None


def test_both_reference_routes_apply_rules_and_only_matching_policy_changes(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.database import FaceDatabase, encoded
    from stash_ai.recognition.hybrid_references import body_rows
    from stash_ai.recognition.body_identity import anchor
    from stash_ai.recognition.clustering import matching_policy
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    c, people = prepared(tmp_path)
    changed = replace(c, data=plain(c.data))
    changed.data['sapiensid']['identities']['anchor_keypoint_rules'] = {'all': [{'group': 'hips', 'min': 1}], 'any': []}
    assert matching_policy(c) != matching_policy(changed)
    assert contract(c) == contract(changed) and body_contract(c) == body_contract(changed)
    value = metadata(['left_hip', 'right_hip'])
    assert not anchor(dict(quality=encoded(value)), changed.data['sapiensid']['identities'])
    with FaceDatabase(c.database) as state:
        db = state.connection
        db.execute('UPDATE person_evidence SET metadata=?', (encoded(value),))
        ns = db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        assert body_rows(db, c, 'scope', ns, people[0])
        assert not body_rows(db, changed, 'scope', ns, people[0])


def test_report_explains_or_without_marking_passing_anchor_failed(tmp_path):
    from stash_ai.recognition import admission_report, admission_panel
    from types import SimpleNamespace
    from PIL import Image
    c = SimpleNamespace(data=defaults())
    c.data['sapiensid']['identities']['anchor_keypoint_rules'] = {
        'all': [{'group': 'hips', 'min': 1}],
        'any': [{'group': 'left_leg', 'min': 3}, {'group': 'right_leg', 'min': 3}]}
    policies = SimpleNamespace(resolve=lambda *args: (c, 'current_configuration'))
    value = metadata(['left_knee', 'left_ankle'])
    for independent in (False, True):
        report = admission_report.person(value, c, policies, '', independent=independent)
        groups = {g['stage']: g for g in report['groups']}
        assert report['anchor_gate']['eligible']
        assert groups['person_anchor_keypoints_any']['status'] == 'pass'
        assert [r['status'] for r in groups['person_anchor_keypoints_any']['checks']] == ['fail', 'pass']
    with Image.new('RGB', (100, 240)) as image:
        image.save(tmp_path/'crop.png')
    result = admission_panel.export(dict(status='exported', file='crop.png'), report, tmp_path, 'crop.png',
                                    observation='test', role='anchor', rejection=None)
    assert (tmp_path/result['file']).is_file()
    payload = json.loads((tmp_path/result['json_file']).read_text())
    assert payload['anchor_gate']['keypoint_rules']['any'][0]['measured'] == 1


def test_indexed_solver_patch_preserves_rules_and_round_trips_yaml(tmp_path):
    from stash_ai.recognition.merge_report_policy import parameter, patched, proposal
    from stash_ai.recognition.merge_solver_domains import registry, normalize
    from stash_ai.recognition.merge_solver_profiles import admission
    from stash_ai.recognition.threshold_tuning import configurations
    from stash_ai.recognition.config import _merge
    c = load(root=tmp_path)
    data = plain(c.data)
    data['sapiensid']['enabled'] = True
    data['sapiensid']['identities']['anchor_keypoint_rules'] = {
        'all': [{'group': 'hips', 'min': 2}],
        'any': [{'group': 'left_leg', 'min': 3}, {'points': ['right_knee', 'right_ankle'], 'min': 2}]}
    c = replace(c, data=data)
    changes = {PARAMETER+'.all.0.min': 1, PARAMETER+'.any.1.min': 1}
    result = proposal(c, changes)
    copied = plain(c.data)
    _merge(copied, yaml.safe_load(result['yaml']))
    assert copied == patched(c, changes).data
    assert parameter(c.data, PARAMETER+'.any.0.min') == 3
    assert normalize(c, changes) == changes
    assert registry(c)[PARAMETER+'.any.1.min'].high == 2
    assert admission(c)[PARAMETER+'.all.0.min'] == 0
    trials, _, _ = configurations(c, {PARAMETER+'.all.0.min': [0, 1]}, 3)
    assert len(trials) == 2
    with pytest.raises(ValueError):
        configurations(c, {PARAMETER+'.all.0.group': [0]}, 3)
    data['sapiensid']['enabled'] = False
    assert not any(k.startswith(PARAMETER) for k in registry(c))


def test_solver_diagnostics_name_only_blocking_regional_parameters():
    from types import SimpleNamespace
    from stash_ai.recognition.merge_solver_diagnostics import diagnostic_record
    rules = {'all': [{'group': 'hips', 'min': 2}],
             'any': [{'group': 'left_leg', 'min': 3}, {'group': 'right_leg', 'min': 3}]}
    gate = anchor_gate(metadata(['left_hip']), settings(rules))['keypoint_rules']
    record = diagnostic_record(SimpleNamespace(data=defaults()), [], {}, {}, [dict(
        observation='test', blockers=[gate['reason']], keypoint_rules=gate)])
    assert record['blocker_trace'][0]['parameters'] == [PARAMETER+'.all.0.min']
    assert record['active_parameters'] == {PARAMETER+'.all.0.min': 1}
    assert not record['blocker_trace'][0]['requires_evidence']
    record = diagnostic_record(SimpleNamespace(data=defaults()), [dict(
        observation='test', status='withheld', reason='body_anchor_pixels', keypoint_rules=gate)], {}, {}, [])
    assert record['blocker_trace'][0]['parameters'] == [
        'sapiensid.identities.anchor_min_person_pixels', PARAMETER+'.all.0.min']
    missing = evaluate(rules, None)
    record = diagnostic_record(SimpleNamespace(data=defaults()), [], {}, {}, [dict(
        observation='old', blockers=[missing['reason']], keypoint_rules=missing)])
    assert record['blocker_trace'][0]['requires_evidence']


def test_region_minima_are_native_integer_tpe_dimensions(tmp_path):
    pytest.importorskip('optuna')
    from stash_ai.recognition.merge_solver_sampler import create
    from stash_ai.recognition.merge_report_policy import patched, parameter
    path = tmp_path/'sampler.yml'
    path.write_text(yaml.safe_dump(dict(version=1, sapiensid=dict(enabled=True, identities=dict(
        anchor_keypoint_rules={'all': [{'group': 'hips', 'min': 1}], 'any': []})))))
    c = load(path)
    key = PARAMETER+'.all.0.min'
    result = create('tpe', 42).suggest(c, [({}, (0,))], {key: [0, 1, 2]}, {}, 15)
    assert type(result[key]) is int and 0 <= result[key] <= 2
    assert parameter(patched(c, result).data, key) == result[key]


def test_solver_relaxes_regional_gate_using_offline_production_replay(tmp_path, monkeypatch):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.database import FaceDatabase, encoded
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve, evaluate as replay
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_report_policy import patched
    c, _, ids, _ = seeded(tmp_path, vectors=[[1, 0]])
    c.data['sapiensid']['global_matching']['enabled'] = False
    with FaceDatabase(c.database) as state:
        for index, oid in enumerate(ids[0]):
            value = metadata(['right_hip'])
            value['quality']['pixel_digest'] = str(index)
            state.connection.execute('UPDATE body_observations SET quality=? WHERE id=?', (encoded(value), oid))
        state.connection.commit()
    c.data['sapiensid']['identities']['anchor_keypoint_rules'] = {'all': [{'group': 'hips', 'min': 2}], 'any': []}
    monkeypatch.setattr('stash_ai.recognition.models.load_backend', lambda *a, **k: pytest.fail('offline inference'))
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900']], policy='strict')
        assert not replay(c, goals, db)['trials']['baseline']['goal_result']['fully_solved']
        result = solve(c, goals, db, maximum=24)
        assert result['status'] == 'fully_solved'
        changes = {r['parameter']: r['proposed'] for r in result['recommendation']['changes']}
        assert changes[PARAMETER+'.all.0.min'] <= 1
        assert replay(patched(c, changes), goals, db)['trials']['baseline']['goal_result']['fully_solved']
    assert c.database.read_bytes() == before
