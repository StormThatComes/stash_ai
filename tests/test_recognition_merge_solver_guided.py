"""Production replays, search limits and interruption of blocker-directed solve."""
import json
import math

import numpy as np
import pytest

from tests.test_recognition_merge_solver_hybrid import prepared
from tests.test_recognition_merge_solver_exploration import auxiliary_goals
from tests.test_recognition_merge_best_effort import incomplete_source
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve, evaluate, objective
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_feedback import QueryFeedback, PREFIX
from stash_ai.recognition.merge_solver_search import BeamSearch, refinement


def make_auxiliary(config, people):
    with FaceDatabase(config.database) as state:
        for person in people[1::2]:
            state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.4) "
                'WHERE id IN (SELECT observation FROM members WHERE person=?)', (person,))
        state.connection.commit()
        return resolve(state.connection, config, [people[i:i+2] for i in range(0,len(people),2)], by_performer=False)


def test_query_evidence_clears_three_simultaneous_gates_outside_local_grid(tmp_path):
    face, body = .431237, .967123
    config, people = prepared(tmp_path, [[1.,0,0,0], [face,math.sqrt(1-face**2),0,0]],
        [[1.,0], [body,math.sqrt(1-body**2)]], matching=dict(face_floor=.8, person_threshold=.99, min_support=4))
    goals = make_auxiliary(config, people)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=2)
    assert result['status'] == 'fully_solved' and result['selected_trial'] == 'trial-1'
    changes = {c['parameter']:c['proposed'] for c in result['best_result']['changes']}
    assert set(changes) == {PREFIX+k for k in ('face_floor','person_threshold','min_support')}
    assert abs(changes[PREFIX+'face_floor']-face) < .000002
    assert abs(changes[PREFIX+'person_threshold']-body) < .000002
    assert changes[PREFIX+'min_support'] == 3
    feedback = result['simulation']['trials']['baseline']['solver_feedback']['query']
    assert feedback['queries'] and feedback['comparisons']
    assert any({'face_floor','person_threshold','reference_support','query_support'} <= set(q['blockers'])
               for q in feedback['queries'])
    assert result['simulation']['trials']['trial-1']['search_strategy'] == 'measured_blockers'
    assert config.database.read_bytes() == before
    json.dumps(result, allow_nan=False)


def test_quality_hints_jointly_admit_rows_filtered_before_query_recovery(tmp_path):
    from tests.test_recognition_merge_report import seeded
    config, people = seeded(tmp_path, vectors=[[1.,0,0,0]]*2, clustering=dict(
        min_anchor_face_pixels=100, min_auxiliary_face_pixels=100,
        min_anchor_sharpness=50, min_auxiliary_sharpness=50))
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = solve(config, goals, db, maximum=2)
    assert result['status'] == 'fully_solved' and result['selected_trial'] == 'trial-1'
    changes = {c['parameter']:c['proposed'] for c in result['best_result']['changes']}
    assert changes == {'clustering.min_anchor_face_pixels':80, 'clustering.min_auxiliary_face_pixels':80,
                       'clustering.min_anchor_sharpness':30, 'clustering.min_auxiliary_sharpness':30}
    assert result['simulation']['trials']['trial-1']['search_strategy'] == 'measured_blockers'


@pytest.mark.parametrize('between', [.783, .785])
def test_query_separation_interval_and_overlapping_scores(tmp_path, between):
    # Identical bodies cannot separate the sets. A .002-wide face interval can;
    # the previous -.02 probe admits the competing set and blocks recovery.
    gram = np.full((4,4), between)
    np.fill_diagonal(gram, 1.)
    gram[0,1] = gram[1,0] = gram[2,3] = gram[3,2] = .785
    faces = np.linalg.cholesky(gram).tolist()
    config, people = prepared(tmp_path, faces, [[1.,0]]*4,
        matching=dict(face_floor=.8, person_threshold=.9, min_support=3))
    goals = make_auxiliary(config, people)
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=2)
    feedback = result['simulation']['trials']['baseline']['solver_feedback']['query']
    interval = next(i for i in feedback['separation_intervals'] if i['parameter'] == PREFIX+'face_floor')
    assert interval['feasible'] is (between < .785)
    if between < .785:
        assert result['status'] == 'fully_solved'
        assert result['simulation']['trials']['trial-1']['goal_result']['disjoint']
    else:
        assert interval['reason'] == 'overlapping_within_and_between_scores'
        assert not result['best_result']['fully_solved']
        assert not result['search']['proves_infeasibility']


def test_observer_does_not_change_production_decisions_or_leak_after_exit(tmp_path):
    from stash_ai.recognition.hybrid_query_observer import observe_queries, query_observer
    config, goals = auxiliary_goals(tmp_path)
    with _open_readonly(config.database) as db:
        ordinary = evaluate(config, goals, db)['trials']['baseline']['goal_result']
        observer = QueryFeedback(goals)
        with observe_queries(observer):
            observed = evaluate(config, goals, db)['trials']['baseline']['goal_result']
        assert objective(ordinary) == objective(observed)
        assert observer.finish(config)['queries']
    assert query_observer(None, (), None, ()) is None


@pytest.mark.parametrize('limit', ['max_candidate_pairs', 'plan_bytes'])
@pytest.mark.parametrize('sets', [1, 2])
def test_query_diagnostics_are_budgeted_and_context_resets(tmp_path, limit, sets):
    from stash_ai.recognition.hybrid_query_observer import observe_queries, query_observer
    config, goals = auxiliary_goals(tmp_path, sets=sets)
    # Tighten only inside the diagnostic, keeping the production catalogue valid.
    observer = QueryFeedback(goals)
    original = observer.begin
    def begin(db, banks, settings, births):
        from dataclasses import replace
        data = plain(settings.data)
        data['sapiensid']['execution'][limit] = 1
        return original(db, banks, replace(settings, data=data), births)
    observer.begin = begin
    with _open_readonly(config.database) as db:
        with observe_queries(observer):
            result = evaluate(config, goals, db)
    feedback = observer.finish(config)
    assert result['trials']['baseline']['goal_result']
    assert feedback['truncated'] and limit in feedback['truncation_reasons']
    assert feedback['skipped_query_visits'] > 0
    if limit == 'max_candidate_pairs' and sets == 1:
        assert feedback['proposals']  # One completely measured query survives.
    else:
        assert not feedback['proposals'] and not feedback['separation_intervals']
    assert query_observer(None, (), None, ()) is None


def test_beam_discards_repeated_bad_plateaus_and_preserves_diversity():
    search = BeamSearch(['hybrid','face'], lambda *a: iter(()), 500)
    search.observe('baseline', {}, (10,0,0), 'baseline')
    for i in range(100):
        search.observe(f'bad-{i}', {'same':i}, (0,-1,-i), 'same_partition', family='face')
    assert len(search.beam) <= 5 and search.pruned >= 95
    for i in range(10):
        search.observe(f'good-{i}', {f'gate-{i}':i}, (20+i,-1,-i), str(i), family='hybrid')
    assert len(search.beam) == search.peak_beam == 5
    assert 'good-9' in search.beam
    assert all(not n.startswith('bad') for n in search.beam)


def test_adaptive_budget_favours_productive_family_without_starving_other():
    def factory(parent, changes, family, refine):
        for i in range(1000):
            yield 'probe', {'i':i}
    search = BeamSearch(['hybrid','face'], factory, 500)
    search.observe('baseline', {}, (0,0,0), 'base')
    wins, seen = 0, []
    for i in range(70):
        family, phase, parent, strategy, changes = search.next()
        seen.append(family)
        wins += family == 'hybrid'
        rank = (wins,0,0) if family == 'hybrid' else (0,0,0)
        search.completed(phase)
        search.observe(f'trial-{i}', changes, rank, str(wins), family=family)
    assert seen.count('hybrid') > seen.count('face')
    assert all('face' in seen[i:i+5] for i in range(len(seen)-4))
    assert search.explored > 13 and search.stop_reason is None


def test_refinement_improvement_resumes_coarse_search():
    def factory(parent, changes, family, refine):
        for i in range(100):
            if refine:
                yield 'face', 'fine', {'value':i}
            else:
                yield 'coarse', {'value':i}
    search = BeamSearch(['face'], factory, 500)
    search.observe('baseline', {}, (0,0,0), 'base')
    for i in range(search.patience+1):
        family, phase, _, _, changes = search.next()
        search.completed(phase)
        search.observe(f'trial-{i}', changes, (int(phase == 'refinement'),0,0), str(i), family=family)
    assert phase == 'refinement' and not search.refining and search.stale == 0
    assert search.next()[1] != 'refinement'


def test_stagnation_refines_and_stops_without_consuming_500_replays(tmp_path):
    config, people = incomplete_source(tmp_path)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = solve(config, goals, db)
    assert result['search']['max_trials'] == 500
    assert result['search']['executed'] < 150
    assert result['search']['stop_reason'] in ('stagnation_after_refinement','refinement_exhausted')
    assert result['search']['executed_by_phase']['refinement'] > 0
    assert result['search']['pruned_trials'] > 0
    assert result['selected_trial'] != 'baseline'
    assert result['best_result']['quality'] == 'best_effort'
    assert result['best_result']['unmet_conditions']


def test_refinement_uses_each_coordinate_and_its_own_measured_bracket(tmp_path):
    config, _ = auxiliary_goals(tmp_path)
    changes = {PREFIX+'face_floor':.7, PREFIX+'person_threshold':.8, PREFIX+'min_support':2}
    options = {PREFIX+'face_floor':[.699,.702], PREFIX+'person_threshold':[.79,.81]}
    candidates = list(refinement(config, changes, options))
    assert any(p[PREFIX+'face_floor'] == .6995 for p in candidates)
    assert any(p[PREFIX+'person_threshold'] == .805 for p in candidates)
    assert any(p[PREFIX+'min_support'] == 1 for p in candidates)
    assert all(sum(p[k] != changes[k] for k in changes) == 1 for p in candidates)


def test_cli_interrupt_exports_logged_winner_with_default_500_trials(tmp_path, monkeypatch, capsys):
    import yaml
    from stash_ai.recognition.cli import main
    from stash_ai.recognition import replay_simulation
    config, goals = auxiliary_goals(tmp_path, sets=2)
    config.source_path.write_text(yaml.safe_dump(plain(config.data)))
    original = replay_simulation.rebuild_selected
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(None)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', interrupted)
    output, document = tmp_path/'report.json', tmp_path/'winner.yml'
    argv = ['report-merge','--face-config',str(config.source_path),'--faces-db',str(config.database),
            '--solve','--output',str(output),'--yaml-output',str(document)]
    for group in goals['groups']:
        argv.extend(['--performer-set',','.join(m['performer_id'] for m in group['members'])])
    assert main(argv) == 130
    result = json.loads(output.read_text())
    assert result['search']['max_trials'] == 500
    assert result['selected_trial'] == 'trial-1' and result['interrupted']
    assert set(result['simulation']['trials']) == {'baseline','trial-1'}
    captured = capsys.readouterr()
    completed = [line for line in captured.err.splitlines() if ': complete ' in line]
    assert len(completed) == 2
    assert 'best=baseline best_number=0' in completed[0]
    assert 'best=trial-1 best_number=1' in completed[1]
    assert captured.out == ''
    assert 'best completed trial trial-1' in document.read_text().splitlines()[0]
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', original)
    with _open_readonly(config.database) as db:
        repeated = evaluate(load(document), goals, db)['trials']['baseline']['goal_result']
    assert repeated['fully_solved'] and repeated['disjoint']


def test_interrupt_in_refinement_keeps_best_completed_yaml(tmp_path):
    from stash_ai.recognition.merge_goal_output import export
    config, people = incomplete_source(tmp_path)
    completed = []
    def progress(name, stage, item):
        if stage == 'complete':
            completed.append((name,item['best_trial']))
            # The default patience is 64; this lands in the refinement window.
            if len(completed) >= 75 and name != item['best_trial']:
                raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = dict(goals, **solve(config, goals, db, progress=progress))
    assert result['interrupted']
    assert result['search']['executed_by_phase']['refinement'] > 0
    assert result['selected_trial'] == completed[-1][1]
    assert result['selected_trial'] != completed[-1][0]
    export(config, result, tmp_path/'report.json', tmp_path/'winner.yml')
    with _open_readonly(config.database) as db:
        repeated = evaluate(load(tmp_path/'winner.yml'), goals, db)['trials']['baseline']['goal_result']
    chosen = result['simulation']['trials'][result['selected_trial']]['goal_result']
    assert objective(repeated) == objective(chosen)
