"""Unavailable scenes must be explicit without stopping every usable replay."""
import json
import uuid

import pytest

from tests.test_recognition_merge_report import seeded
from tests.test_recognition_merge_solver_hybrid import prepared
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, evaluate
from stash_ai.recognition.merge_goal_output import export, summary
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.replay_inputs import ReplayInputError


def supersede(config, scene, *, reason):
    with FaceDatabase(config.database) as state:
        db = state.connection
        old = dict(db.execute('SELECT * FROM scans WHERE scene=? ORDER BY rowid DESC LIMIT 1', (scene,)).fetchone())
        new = dict(old, id=str(uuid.uuid4()), input_sha='new-input')
        if reason == 'policy':
            new['policy_sha'] = 'incompatible-detection-policy'
        elif reason == 'partial':
            new.update(status='partial', cursor=0)
        elif reason == 'cursor':
            new.update(status='complete', cursor=0)
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', tuple(new.values()))
        db.commit()
    return old, new


def test_unrelated_incompatible_scene_does_not_block_all_trials(tmp_path, monkeypatch):
    from stash_ai.recognition import replay_simulation
    from stash_ai.recognition.config import load
    from stash_ai.recognition.merge_goals import assess
    config, people = seeded(tmp_path, vectors=[[1., 0, 0, 0], [.7, .71414284285, 0, 0], [0, 0, 1., 0]])
    old, newest = supersede(config, '3', reason='policy')
    before = config.database.read_bytes()
    original = replay_simulation.rebuild_selected
    selections = []
    def rebuild(state, selected, settings):
        selections.append(selected)
        assert {s['scene'] for s in selected} == {'1', '2'}
        # Exclusion must not erase competitors or fall back to older scans.
        assert state.connection.execute('SELECT 1 FROM scans WHERE id=?', (newest['id'],)).fetchone()
        assert state.connection.execute('SELECT 1 FROM scans WHERE id=?', (old['id'],)).fetchone()
        assert state.connection.execute('SELECT 1 FROM members WHERE person=?', (people[2],)).fetchone()
        return original(state, selected, settings)
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', rebuild)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people[:2]], by_performer=False)
        result = dict(goals, **solve(config, goals, db, maximum=12))
    assert result['status'] == 'fully_solved' and selections
    inputs = result['simulation']['input_selection']
    assert inputs['selected_scene_count'] == 2 and inputs['latest_scene_count'] == 3
    assert not inputs['complete_scope']
    assert inputs['excluded_scans'][0]['scene'] == '3'
    assert inputs['excluded_scans'][0]['reasons'] == ['scan_policy_mismatch']
    export(config, result, tmp_path/'result.json', tmp_path/'result.yml')
    assert 'Partial input scope: verified on 2/3 scenes' in (tmp_path/'result.yml').read_text()
    assert result['run_argv'][-4:] == ['--scene-id', '1', '--scene-id', '2']
    assert result['explanation']['input_blockers'] == inputs['excluded_scans']
    assert 'input scope is partial' in summary(result)
    with _open_readonly(config.database) as db:
        repeated = replay_simulation.replay(load(tmp_path/'result.yml'), {}, source=db, scope=goals['scope'],
            scenes=result['simulation']['selection'], consume=lambda name, observations, retained, predicted:
                dict(goal_result=assess(goals, observations, retained, predicted)))
    assert repeated['trials']['baseline']['goal_result']['fully_solved']
    assert config.database.read_bytes() == before


@pytest.mark.parametrize('reason', ['policy', 'partial', 'cursor'])
def test_unavailable_goal_observations_make_result_best_effort(tmp_path, reason):
    config, people = seeded(tmp_path)
    old, newest = supersede(config, '2', reason=reason)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = dict(goals, **solve(config, goals, db, maximum=1))
    assert result['status'] == 'best_effort'
    assert result['simulation']['selection'] == ['1']
    missing = next(v for v in result['best_result']['unmet_conditions'] if v['reason'] == 'missing_latest_observations')
    assert missing['identity'] == people[1] and missing['count'] == 3
    assert not result['best_result']['fully_solved']
    blocker = result['simulation']['input_selection']['excluded_scans'][0]
    assert blocker['scan'] == newest['id'] and blocker['scan'] != old['id']
    assert blocker['reasons'] == ['scan_policy_mismatch' if reason == 'policy' else 'incomplete_scan']
    assert config.database.read_bytes() == before


def test_no_compatible_scans_writes_diagnostics_without_yaml_or_trials(tmp_path, capsys):
    from stash_ai.recognition.cli import main
    config, people = seeded(tmp_path)
    for scene in ('1', '2'):
        supersede(config, scene, reason='policy')
    before = config.database.read_bytes()
    output, yaml = tmp_path/'result.json', tmp_path/'result.yml'
    args = ['report-merge', '--face-config', str(config.source_path), '--faces-db', str(config.database),
        '--solve', '--max-trials', '2', '--output', str(output), '--yaml-output', str(yaml)]
    for person in people:
        args.extend(['--identity-id', person])
    assert main(args) == 2
    result = json.loads(output.read_text())
    assert result['status'] == 'no_compatible_scans'
    assert result['best_result'] is None and result['selected_trial'] is None
    assert result['simulation']['trials'] == {} and result['search']['executed'] == 0
    assert not result['yaml_exported'] and not yaml.exists()
    assert len(result['explanation']['input_blockers']) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert 'no completed trial' in captured.err
    assert 'scan_policy_mismatch: scenes=1,2' in captured.err
    assert 'sapiensid.matching.* changes alone do not require a rescan' in captured.err
    assert 'run_argv' not in captured.err
    assert config.database.read_bytes() == before


def test_strict_simulation_keeps_input_requirement_with_actionable_diagnostics(tmp_path):
    config, people = seeded(tmp_path)
    supersede(config, '2', reason='policy')
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        with pytest.raises(ReplayInputError, match='before baseline') as caught:
            evaluate(config, goals, db)
    assert caught.value.diagnostics['excluded_scans'][0]['scene'] == '2'
    assert 'scan_policy_mismatch: scenes=2' in str(caught.value)


@pytest.mark.parametrize('gate,value', [('face_floor', .78), ('person_threshold', .9),
    ('ambiguity_margin', .05), ('min_support', 2)])
def test_matching_changes_do_not_invalidate_collected_inputs(tmp_path, gate, value):
    config, people = prepared(tmp_path, [[1., 0, 0, 0], [.7, .71414284285, 0, 0]], [[1., 0], [1., 0]])
    updated = patched(config, {'sapiensid.matching.'+gate: value})
    assert scan_identity(config, namespace(config)) == scan_identity(updated, namespace(updated))
    with _open_readonly(config.database) as db:
        goals = resolve(db, updated, [people], by_performer=False)
        result = solve(updated, goals, db, maximum=1)
    assert result['search']['executed'] == 1
    assert result['simulation']['input_selection']['complete_scope']


def test_real_detection_change_is_not_treated_as_threshold_retuning(tmp_path):
    config, people = seeded(tmp_path)
    updated = patched(config, {'detection.min_sharpness': config.data['detection']['min_sharpness']+1})
    with _open_readonly(config.database) as db:
        goals = resolve(db, updated, [people], by_performer=False)
        result = solve(updated, goals, db, maximum=2)
    assert result['status'] == 'no_compatible_scans'
    assert result['simulation']['trials'] == {}
    assert {e['scene'] for e in result['simulation']['input_selection']['excluded_scans']} == {'1', '2'}
