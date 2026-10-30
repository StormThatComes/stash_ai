from stash_ai.recognition.resource_errors import ResourceLimit
"""Constructive witnesses use real production replay, without weakening goals."""
import json
import uuid

import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, assess, evaluate
from stash_ai.recognition.merge_report_policy import patched, parameter
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_feasibility import Feasibility, resource_limit
from stash_ai.recognition.merge_solver_profiles import profiles, admission
from tests.test_recognition_merge_report import seeded


@pytest.mark.parametrize('case,profile', [('face', 'matching_floor'), ('quality', 'admission_floor')])
def test_initial_production_probe_constructs_merge_and_exports_safe_winner(tmp_path, case, profile):
    from benchmarks.benchmark_merge_solver import fixture
    c, goals = fixture(tmp_path, case)
    before = c.database.read_bytes()
    with _open_readonly(c.database) as db:
        result = solve(c, goals, db, targeted_trials=0, maximum=8)
        witness = result['feasibility']['joint_witness_trial']
        assert result['feasibility']['status'] == 'possible'
        trial = result['simulation']['trials'][witness]
        assert trial['search_phase'] == 'feasibility' and trial['search_strategy'] == profile
        assert trial['goal_result']['fully_solved']
        assert result['search']['attempted'] <= 8
        check = patched(c, {r['parameter']: r['proposed'] for r in result['recommendation']['changes']})
        assert evaluate(check, goals, db)['trials']['baseline']['goal_result']['fully_solved']
    assert c.database.read_bytes() == before
    json.dumps(result, allow_nan=False)


def test_combined_probe_crosses_matching_and_admission_gates(tmp_path):
    c, _ = seeded(tmp_path, clustering=dict(min_anchor_face_pixels=100, min_auxiliary_face_pixels=100))
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        result = solve(c, goals, db, targeted_trials=0, maximum=8)
    assert result['feasibility']['status'] == 'possible'
    assert result['feasibility']['joint_witness_trial'] == 'trial-3'
    assert result['simulation']['trials']['trial-3']['search_strategy'] == 'all_floors'


def test_overmerge_is_a_witness_and_repair_parent_but_not_a_winner(tmp_path, monkeypatch):
    from tests.test_recognition_solver_bidirectional import plateau
    c, goals = plateau(tmp_path, monkeypatch, size=20)
    with _open_readonly(c.database) as db:
        result = solve(c, goals, db, targeted_trials=0, maximum=8)
    trials = result['simulation']['trials']
    first = trials['trial-1']
    assert first['search_phase'] == 'feasibility'
    assert first['goal_result']['construction']['status'] == 'possible_with_conflicts'
    assert all(g['merged'] for g in first['goal_result']['construction']['groups'])
    assert not first['goal_result']['admissible'] and first['best_trial'] != 'trial-1'
    assert result['feasibility']['probes_completed'] == 1
    assert result['best_result']['fully_solved']
    assert any(t.get('search_parent') == 'trial-1' for t in trials.values())


def test_baseline_witness_avoids_redundant_probes_even_with_negative_conflict(tmp_path):
    from tests.test_recognition_scene_exclusions import fixture
    c, _ = fixture(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], scene_exclusions=[('900', '3')])
        result = solve(c, goals, db, targeted_trials=0, maximum=12)
    baseline = result['simulation']['trials']['baseline']['goal_result']
    assert baseline['construction']['status'] == 'possible_with_conflicts'
    assert not baseline['scene_exclusions_satisfied']
    assert result['feasibility']['probes_completed'] == 0
    assert result['best_result']['fully_solved']


@pytest.mark.parametrize('maximum,probes', [(1, 6), (4, 0), (2, 6)])
def test_budget_and_opt_out_do_not_claim_impossibility(tmp_path, maximum, probes):
    c, _ = seeded(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        result = solve(c, goals, db, targeted_trials=0, maximum=maximum, feasibility_trials=probes)
    assert result['search']['attempted'] <= maximum
    assert result['feasibility']['probe_budget'] == min(probes, maximum-1, max(1, (maximum-1)//2))
    if maximum == 1:
        assert result['feasibility']['status'] == 'not_demonstrated'
        assert not result['feasibility']['groups'][0]['input_certificates']
    if probes == 0:
        assert result['feasibility']['probes_completed'] == 0
    assert result['search']['proves_infeasibility'] is False


def test_profile_floors_preserve_contracts_and_disabled_routes(tmp_path):
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.person_signal import same_contract
    c, _ = seeded(tmp_path)
    c = replace(c, data=plain(c.data))
    c.data['clustering']['min_anchor_visible_face_fraction'] = .5
    c.data['clustering']['min_auxiliary_visible_face_fraction'] = None
    c.data['detection']['min_visible_face_fraction'] = .8
    liberal = admission(c)
    assert 'clustering.min_anchor_visible_face_fraction' not in liberal
    assert 'clustering.min_auxiliary_visible_face_fraction' not in liberal
    c.data['detection']['min_visible_face_fraction'] = .4  # Valid production configuration.
    patches = dict(profiles(c))
    assert patches['matching_floor']['recognition.match_threshold'] == 0.
    for patch in patches.values():
        variant = patched(c, patch)
        assert same_contract(c, variant)
        assert scan_identity(c, namespace(c)) == scan_identity(variant, namespace(c))
        assert c.data['sapiensid']['identities']['enabled'] == variant.data['sapiensid']['identities']['enabled']
        assert c.data['sapiensid']['global_matching']['enabled'] == variant.data['sapiensid']['global_matching']['enabled']
    c.data['sapiensid']['enabled'] = False
    assert all(not k.startswith('sapiensid.') for _, p in profiles(c) for k in p)


def test_negative_feedback_cannot_overwrite_liberal_similarity_floors(tmp_path):
    from stash_ai.recognition.merge_solver_bidirectional import relaxed_seeds
    c, _ = seeded(tmp_path)
    feedback = dict(options={'recognition.match_threshold': [.7, .8]},
        proposals={'face': [{'recognition.match_threshold': .965926}]},
        quality=[dict(patch={'clustering.max_anchor_pose_deviation': .4,
                             'recognition.match_threshold': .99})])
    seeds = list(relaxed_seeds(c, feedback))
    assert any(p.get('recognition.match_threshold') == 0. for p in seeds)
    assert any(p.get('recognition.match_threshold') == .7 for p in seeds)
    assert all(p.get('recognition.match_threshold', 0.) <= parameter(c.data, 'recognition.match_threshold') for p in seeds)
    assert all(p['clustering.max_anchor_pose_deviation'] == .4 for p in seeds)


def test_missing_selected_scene_gets_narrow_input_certificate(tmp_path):
    c, _ = seeded(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        scans = list(db.execute("SELECT id FROM scans WHERE scene='1'"))
        diagnostic = Feasibility(db, c, goals, scans)
        report = diagnostic.report({}, {}, 0)
    assert report['status'] == 'blocked_inputs'
    certs = report['groups'][0]['input_certificates']
    assert dict(reason='required_scene_has_no_available_goal_witness', scene='2') in certs
    assert 'selected retained scans' in report['certificate_scope']


def test_observed_blocker_is_not_an_input_impossibility_certificate(tmp_path):
    c, _ = seeded(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        diagnostic = Feasibility(db, c, goals, list(db.execute('SELECT id FROM scans')))
        rows = [r for g in goals['groups'] for m in g['members'] for r in m['observations']]
        observations = {r['id']: r['scene'] for r in rows}
        predicted = {r['id']: r['scene'] for r in rows}
        status = assess(goals, observations, {}, predicted)
        status['construction'] = diagnostic.assess(status, observations, predicted,
            [dict(stage='matching', reason='strict_core_conflict', observation=rows[0]['id'])])
        report = diagnostic.report({'baseline': dict(goal_result=status)}, {}, 0)
    assert report['status'] == 'not_demonstrated'
    assert report['groups'][0]['best_observation']['observed_blockers'][0]['reason'] == 'strict_core_conflict'
    assert not report['groups'][0]['input_certificates']


def test_strict_policy_keeps_missing_sample_failure_separate_from_construction(tmp_path):
    c, _ = seeded(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']], policy='strict')
        diagnostic = Feasibility(db, c, goals, list(db.execute('SELECT id FROM scans')))
        rows = [r for g in goals['groups'] for m in g['members'] for r in m['observations']]
        observations = {r['id']: r['scene'] for r in rows}
        predicted = dict.fromkeys(observations, 'one')
        predicted[rows[0]['id']] = None
        status = assess(goals, observations, {}, predicted)
        construction = diagnostic.assess(status, observations, predicted, [])
    assert not status['fully_solved']
    assert construction['status'] == 'possible_goal_mismatch'


def test_different_per_set_witnesses_do_not_prove_joint_reachability(tmp_path):
    c, _ = seeded(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900'], ['901']])
        diagnostic = Feasibility(db, c, goals, list(db.execute('SELECT id FROM scans')))
        observations = {r['id']: r['scene'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
        trials = {}
        for i, group in enumerate(goals['groups']):
            predicted = {r['id']: 'one' for m in group['members'] for r in m['observations']}
            status = assess(goals, observations, {}, predicted)
            status['construction'] = diagnostic.assess(status, observations, predicted, [])
            trials[str(i)] = dict(goal_result=status)
        report = diagnostic.report(trials, {}, 0)
    assert all(g['status'] == 'possible' for g in report['groups'])
    assert report['status'] == 'not_demonstrated' and report['joint_witness_trial'] is None


def body_handoff(tmp_path):
    from tests.test_recognition_body_integration import seeded as bodies
    c, selected, ids, people = bodies(tmp_path, vectors=[[1, 0]])
    c.data['clustering']['min_anchor_visible_face_fraction'] = .8
    c.data['clustering']['min_auxiliary_visible_face_fraction'] = .8
    from stash_ai.recognition.visibility import CONTRACT_ID
    faces = []
    with FaceDatabase(c.database) as state:
        db = state.connection
        for cue, body in enumerate(ids[0]):
            face = str(uuid.uuid4()); faces.append(face)
            quality = dict(pixels=100, sharpness=100., pose_deviation=0.,
                           visible_face_fraction=.6, visibility_contract=CONTRACT_ID)
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (face, selected[0]['scan'], cue, float(cue), '', '[]', '[]', encoded(quality),
                 None, 4, np.asarray([1, 0, 0, 0], dtype='<f4').tobytes(), None))
            db.execute('INSERT INTO members VALUES (?,?)', (face, people[0]))
            db.execute('UPDATE body_observations SET face=? WHERE id=?', (face, body))
            from stash_ai.recognition.person_signal import save, contract
            vector = np.zeros(4096, dtype='<f4'); vector[0] = 1.
            save(db, face, contract(c), {}, vector)
        from stash_ai.recognition.models import namespace
        from stash_ai.recognition.scanning import scan_identity
        db.execute('UPDATE scans SET policy_sha=?', (scan_identity(c, namespace(c)),))
        db.commit()
    return c, faces, ids[0]


@pytest.mark.parametrize('policy', ['identity', 'strict'])
def test_body_handoff_is_construction_witness_without_relaxing_goal_policy(tmp_path, policy):
    c, faces, bodies = body_handoff(tmp_path)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900']], policy=policy)
        assert len(goals['groups'][0]['members'][0]['observations']) == 3
        result = solve(c, goals, db, targeted_trials=0, maximum=1)
        status = result['simulation']['trials']['baseline']['goal_result']
        assert not status['fully_solved']
        assert result['feasibility']['status'] == 'possible_goal_mismatch'
        assert result['feasibility']['groups'][0]['inputs']['linked_body_alternatives'] == 3
        diagnostic = Feasibility(db, c, goals, list(db.execute('SELECT id FROM scans')))
        observations = dict.fromkeys(faces+bodies, '1')
        # Two addresses are one independent vote, never twice the core support.
        both = dict.fromkeys(faces+bodies, 'one')
        measured = diagnostic.assess(assess(goals, observations, {}, both), observations, both, [])
        assert measured['groups'][0]['core_support'] == 3
        conflict = dict.fromkeys(faces, 'face') | dict.fromkeys(bodies, 'body')
        measured = diagnostic.assess(assess(goals, observations, {}, conflict), observations, conflict, [])
        assert not measured['merged'] and measured['groups'][0]['core_support'] == 0


def test_resource_limited_probe_continues_and_resume_does_not_retry_failure(tmp_path, monkeypatch):
    import stash_ai.recognition.replay_simulation as replay
    c, _ = seeded(tmp_path)
    checkpoint = tmp_path/'journal.json'
    before = c.database.read_bytes()
    original, calls = replay.rebuild_selected, []

    def guarded(state, selected, settings):
        calls.append(settings.data['recognition']['match_threshold'])
        if len(calls) == 2:
            raise ResourceLimit('Reference plan exceeds sapiensid.execution.plan_bytes.')
        return original(state, selected, settings)

    monkeypatch.setattr(replay, 'rebuild_selected', guarded)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        result = solve(c, goals, db, targeted_trials=0, maximum=6, checkpoint=checkpoint)
        assert result['feasibility']['probes_failed'] >= 1
        assert result['feasibility']['failed_probes']['trial-1']['reason'] == 'resource_limit'
        assert 'trial-1' not in result['simulation']['trials'] and result['selected_trial'] != 'trial-1'
        assert result['search']['attempted'] == 6
        assert len(calls) == 6
        calls.clear()
        resumed = solve(c, goals, db, targeted_trials=0, maximum=6, checkpoint=checkpoint, resume=True)
        assert not calls and resumed['search']['restored_trials'] == 6
        assert resumed['simulation']['trials'] == result['simulation']['trials']
        assert resumed['feasibility'] == result['feasibility']
        with pytest.raises(ValueError, match='incompatible'):
            solve(c, goals, db, targeted_trials=0, maximum=6, feasibility_trials=0, checkpoint=checkpoint, resume=True)
    assert c.database.read_bytes() == before


@pytest.mark.parametrize('error', [RuntimeError('reference_bytes exceeded'), ValueError('invalid policy'),
                                  ValueError('unexpected budget exceeded')])
def test_unrelated_probe_errors_are_not_suppressed(tmp_path, monkeypatch, error):
    import stash_ai.recognition.replay_simulation as replay
    c, _ = seeded(tmp_path)
    original = replay.rebuild_selected

    def broken(state, selected, settings):
        if settings.data['recognition']['match_threshold'] == 0.:
            raise error
        return original(state, selected, settings)

    monkeypatch.setattr(replay, 'rebuild_selected', broken)
    assert not resource_limit(error)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        with pytest.raises(type(error), match=str(error)):
            solve(c, goals, db, targeted_trials=0, maximum=3)


def test_interrupt_during_probe_keeps_only_completed_results(tmp_path, monkeypatch):
    import stash_ai.recognition.replay_simulation as replay
    c, _ = seeded(tmp_path)
    original = replay.rebuild_selected

    def interrupted(state, selected, settings):
        if settings.data['recognition']['match_threshold'] == 0.:
            raise KeyboardInterrupt
        return original(state, selected, settings)

    monkeypatch.setattr(replay, 'rebuild_selected', interrupted)
    with _open_readonly(c.database) as db:
        goals = resolve(db, c, [['900', '901']])
        result = solve(c, goals, db, targeted_trials=0, maximum=4)
    assert result['interrupted'] and list(result['simulation']['trials']) == ['baseline']
    assert result['feasibility']['status'] == 'not_demonstrated'
    assert not result['feasibility']['failed_probes']


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_cli_exposes_feasibility_and_rejects_negative_budget(tmp_path, capsys, style):
    from stash_ai.recognition.cli import main, parser
    c, _ = seeded(tmp_path)
    args = ['report-merge', '--face-config', str(c.source_path), '--faces-db', str(c.database),
            '--performer-set', '900,901', '--solve', '--targeted-trials', '0', '--max-trials', '2', '--progress-style', style]
    assert main(args) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result['explanation']['feasibility_status'] == 'possible'
    if style == 'auto':
        assert 'Search: phase=feasibility profile=matching_floor' in output.err
        assert 'Construction: possible merged_sets=[1]' in output.err
        assert 'Merge feasibility: possible' in output.err
    else:
        assert output.err == ''
    with pytest.raises(SystemExit):
        parser().parse_args(args + ['--feasibility-trials', '-1'])
