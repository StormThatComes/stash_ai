"""Strict feasible thresholds must win over a loose baseline or smaller diff."""
from types import SimpleNamespace

import pytest

from stash_ai.recognition.config import defaults
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, evaluate
from stash_ai.recognition.merge_report_policy import patched, parameter
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_tighten import Tightener, route_parameters, FACE, VIEW, HYBRID
from stash_ai.recognition.merge_solver_targeted import TargetedSearch, route_order
from tests.test_recognition_merge_report import seeded


MATCH = 'recognition.match_threshold'
COHESION = 'clustering.cohesion_threshold'


def lane(keys, changes=None):
    config = SimpleNamespace(data=defaults())
    config.data['sapiensid']['enabled'] = True
    config.data['recognition']['match_threshold'] = .2
    config.data['clustering']['cohesion_threshold'] = .1
    config.data['clustering']['max_anchor_pose_deviation'] = .2
    search = Tightener(config)
    search.begin('baseline', changes or {}, {'matching_parameters':dict.fromkeys(keys, 1)})
    return search


def test_binary_boundary_preserves_last_feasible_configuration():
    search = lane([MATCH])
    changes = {}
    search.observe(changes, True)
    for number in range(40):
        try:
            _, _, _, patch = search.next(str(number), changes, {})
        except StopIteration:
            break
        accepted = patch[MATCH] <= .432137
        search.observe(patch, accepted)
        if accepted:
            changes = patch
    else:
        pytest.fail('Boundary search failed to converge on the six-decimal lattice')
    assert changes[MATCH] == .432137
    assert search.exhausted


def test_boundary_is_invalidated_when_another_gate_changes():
    search = lane([MATCH, 'recognition.ambiguity_margin'])
    search.observe({MATCH:.4}, False)
    candidate = search.candidate({}, (MATCH,), {})
    assert candidate[MATCH] < .4
    changed_context = {'recognition.ambiguity_margin':.2}
    candidate = search.candidate(changed_context, (MATCH,), {})
    assert candidate[MATCH] > .4


def test_inverse_integer_and_admission_bounds_do_not_drop_more_evidence():
    bonus = 'clustering.same_scene_match_bonus'
    support = 'recognition.min_merge_support'
    quality = 'clustering.max_anchor_pose_deviation'
    search = lane([bonus, support], {quality:.8})
    search.config.data['clustering']['same_scene_match_bonus'] = .05
    search.config.data['recognition']['min_merge_support'] = 2
    patch = search.candidate({}, (bonus,), {})
    assert patch[bonus] == 0
    assert search.score(patch) > search.score({})
    for value in (8, 5, 4):
        search.observe({support:value}, False)
    assert search.candidate({}, (support,), {})[support] == 3
    assert search.candidate({quality:.8}, (quality,), {}).get(quality,
        parameter(search.config.data, quality)) == parameter(search.config.data, quality)
    assert search.score({quality:0}) == search.score({})


@pytest.mark.parametrize('references', [4, 32])
def test_support_proposals_and_tightening_respect_the_cap(references):
    from stash_ai.recognition.merge_solver_domains import normalize
    from stash_ai.recognition.merge_solver_options import compact

    config = SimpleNamespace(data=defaults())
    body = config.data['sapiensid']
    body['enabled'] = body['collection']['enabled'] = body['identities']['enabled'] = True
    body['global_matching'].update(enabled=True, max_references=references)
    keys = {
        'recognition.min_identity_support', 'recognition.min_merge_support',
        'clustering.same_scene_min_support', 'clustering.auxiliary_min_support',
        'clustering.view_min_support', 'clustering.view_attach_min_support',
        'sapiensid.matching.min_support', 'sapiensid.identities.min_support',
        'sapiensid.identities.attach_min_support', 'sapiensid.global_matching.min_support',
    }
    search = Tightener(config)
    search.begin('baseline', {}, {'matching_parameters': dict.fromkeys(keys, 1)})
    for key in keys:
        cap = min(8, references) if key == 'sapiensid.global_matching.min_support' else 8
        assert compact(config, {key: [cap - 1, 1000]}, {key: [9, 800]})[key] == [cap, cap - 1]
        assert search.candidate({}, (key,), {})[key] == cap
        for proposed in (cap - 1, cap, 9, 1000):
            patch = normalize(config, {key: proposed})
            assert search.value(patch, key) == min(proposed, cap)
    assert search.domains['clustering.view_min_support'].low == 2
    assert search.domains['sapiensid.identities.min_support'].low == 2
    assert normalize(config, {'sapiensid.identities.anchor_min_keypoints': 12}) == {
        'sapiensid.identities.anchor_min_keypoints': 12}


@pytest.mark.parametrize(('mode', 'trials'), [('diff', 1), ('full', 5)])
def test_solver_caps_legacy_input_before_replay_and_exports_the_change(tmp_path, capsys, mode, trials):
    import json
    import yaml
    from stash_ai.recognition.cli import main
    from stash_ai.recognition.config import digest

    config, _ = seeded(tmp_path, recognition={'min_merge_support': 1000})
    original = config.source_path.read_bytes()
    winner = tmp_path / 'winner.yml'
    checkpoint = tmp_path / 'solver.json'
    assert main(['report-merge', '--solve', '--face-config', str(config.source_path),
                 '--faces-db', str(config.database), '--performer-set', '900,901',
                 '--checkpoint', str(checkpoint),
                 '--max-trials', str(trials), '--yaml-mode', mode, '--yaml-output', str(winner)]) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    replays = report['simulation']['trials']
    assert len(replays) == trials
    assert replays['baseline']['changes_from_previous'] == [
        dict(parameter='recognition.min_merge_support', previous=1000, current=8)]
    for trial in replays.values():
        assert 1 <= trial['yaml_patch']['recognition']['min_merge_support'] <= 8
    journal = json.loads(checkpoint.read_text())
    for entry in journal['records']:
        record = json.loads((tmp_path / 'solver.json.trials' / entry['file']).read_text())
        changes = {row['parameter']: row['proposed'] for row in replays[record['name']]['changes']}
        assert record['configuration'] == digest(patched(config, changes).data)
    selected = report['best_result']
    assert selected['changed']
    assert winner.read_text() == journal['winner']['text']
    assert yaml.safe_load(winner.read_text())['recognition']['min_merge_support'] == (
        selected['yaml_patch']['recognition']['min_merge_support'])
    assert config.source_path.read_bytes() == original


def test_disabled_and_unused_gates_do_not_improve_strength():
    unused = 'recognition.cooccurrence_merge_threshold'
    search = lane([MATCH], {unused:.99})
    assert search.score({unused:1}) == search.score({})
    assert search.limit(unused) == parameter(search.config.data, unused)
    assert search.score({MATCH:.7}) > search.score({MATCH:.6, HYBRID[1]:1})


def test_dependency_projection_never_relaxes_another_parameter():
    search = lane([MATCH, COHESION])
    patch = search.candidate({MATCH:.4, COHESION:.4}, (COHESION,), {})
    # Cohesion cannot exceed the already patched match threshold. The coupled
    # task is required to try strengthening both gates.
    assert patch is None
    joint = search.candidate({MATCH:.4, COHESION:.4}, (MATCH, COHESION), {})
    assert joint[MATCH] == joint[COHESION] == 1


def test_production_routes_keep_face_views_hybrid_and_body_distinct():
    rows = [dict(status='accepted', reason='candidate_group'),
            dict(status='accepted', reason='view_identity'),
            dict(status='accepted', reason='hybrid_query', hybrid_query={'signal':'face_and_body'}),
            dict(status='withheld', reason='same_scene_identity'),
            dict(status='accepted', evidence_kind='person', reason='measured_face_association')]
    active = route_parameters(rows)
    assert set(FACE+VIEW+HYBRID) <= active.keys()
    assert 'clustering.same_scene_match_bonus' not in active
    assert not any(k.startswith('sapiensid.identities.') for k in active)
    face_only = route_parameters([dict(status='accepted', reason='hybrid_query', hybrid_query={'signal':'face'})])
    assert set(FACE) <= face_only.keys() and not set(HYBRID) & face_only.keys()


def test_face_preference_has_bounded_fallback_and_keeps_uncertainty_explicit():
    candidates = [dict(set=0, route='face', patch={MATCH:v}) for v in (.7,.6,.5,.4)]
    candidates += [dict(set=0, route='query', patch={HYBRID[0]:.6}),
                   dict(set=0, route='face_query', uncertain_face=True, patch={MATCH:.3})]
    source = {'targeted':{'candidates':list(route_order(candidates))}}
    search = TargetedSearch(8)
    assert [search.next('baseline', source, {})['route'] for _ in range(3)] == ['face','face','query']
    assert source['targeted']['candidates'][3]['uncertain_face']


def test_loose_solved_baseline_is_tightened_and_all_goals_are_replayed(tmp_path):
    config, _ = seeded(tmp_path, recognition={'match_threshold':.3}, clustering={'cohesion_threshold':.2})
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900','901']])
        result = solve(config, goals, db, maximum=60)
        changes = {r['parameter']:r['proposed'] for r in result['best_result']['changes']}
        assert evaluate(patched(config, changes), goals, db)['trials']['baseline']['goal_result']['fully_solved']
    assert result['search']['tightening']['first_solution'] == 'baseline'
    assert result['selected_trial'] != 'baseline'
    assert .69 < changes[MATCH] <= .7
    assert .69 < changes[COHESION] <= changes[MATCH]
    assert result['search']['executed_by_phase']['tightening'] > 0
    assert result['search']['attempted'] <= 60
    assert not result['search']['proves_optimality']
    assert before == config.database.read_bytes()


def test_stricter_threshold_cannot_win_by_replacing_good_faces_with_body(tmp_path):
    import math
    from tests.test_recognition_merge_solver_hybrid import prepared
    config, people = prepared(tmp_path, [[1.,0,0,0],[.9,math.sqrt(.19),0,0]], [[1.,0],[1.,0]],
                              matching={'min_support':3})
    config.data['recognition']['match_threshold'] = .85
    config.data['clustering']['cohesion_threshold'] = .8
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = solve(config, goals, db, maximum=40)
    trials = result['simulation']['trials']
    assert trials['baseline']['goal_result']['fully_solved']
    assert trials['baseline']['solver_feedback']['fallback_references'] == 0
    assert any(t['goal_result']['fully_solved'] and t['solver_feedback']['fallback_references'] > 0
               for t in trials.values())
    winner = trials[result['selected_trial']]
    assert winner['solver_feedback']['fallback_references'] == 0
    changes = {r['parameter']:r['proposed'] for r in result['best_result']['changes']}
    assert .85 < changes[MATCH] <= .900001


def test_first_solution_during_polishing_reopens_tightening(tmp_path, monkeypatch):
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    from stash_ai.recognition.merge_solver_search import BeamSearch
    from stash_ai.recognition.merge_solver_bidirectional import BidirectionalSearch
    from stash_ai.recognition.merge_solver_polish import Polisher
    config, goals = auxiliary_goals(tmp_path)
    def exhausted(self):
        self.stop_reason = 'frontier_exhausted'
        raise StopIteration
    original = Polisher.next
    def offer(self, winner, parents, changes, feedback):
        if winner == 'baseline':
            return 'minimization', winner, 'parameter_ablation', {HYBRID[0]:.78}
        return original(self, winner, parents, changes, feedback)
    monkeypatch.setattr(BeamSearch, 'next', exhausted)
    monkeypatch.setattr(BidirectionalSearch, 'next', lambda *args, **kwargs: None)
    monkeypatch.setattr(Polisher, 'next', offer)
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=30, targeted_trials=0, feasibility_trials=0)
    assert result['search']['tightening']['first_solution'] == 'trial-1'
    assert result['simulation']['trials']['trial-1']['search_phase'] == 'minimization'
    assert result['search']['tightening']['executed'] > 0
    assert result['search']['stop_reason'] != 'frontier_exhausted'


@pytest.mark.parametrize('scheduling', ['batch', 'async'])
def test_parallel_tightening_drains_pending_boundaries_and_resumes(tmp_path, scheduling):
    config, _ = seeded(tmp_path, recognition={'match_threshold':.3}, clustering={'cohesion_threshold':.2})
    checkpoint = tmp_path/'checkpoint.json'
    execution = dict(cpus=4, trials=2, scheduling=scheduling, memory_bytes=128*1024**3)
    def interrupt(name, stage, item):
        if name == 'trial-4' and stage == 'complete':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900','901']])
        first = solve(config, goals, db, maximum=24, execution=execution, checkpoint=checkpoint, progress=interrupt)
        assert first['interrupted']
        result = solve(config, goals, db, maximum=24, execution=execution, checkpoint=checkpoint, resume=True)
        repeated = solve(config, goals, db, maximum=24, execution=execution, checkpoint=checkpoint, resume=True)
        assert repeated['simulation']['trials'] == result['simulation']['trials']
        assert repeated['best_result'] == result['best_result']
        changes = {r['parameter']:r['proposed'] for r in result['best_result']['changes']}
        assert evaluate(patched(config, changes), goals, db)['trials']['baseline']['goal_result']['fully_solved']
    assert result['search']['tightening']['executed'] > 2
    assert changes[MATCH] > .3
