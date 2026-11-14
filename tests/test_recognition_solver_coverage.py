"""Global recovery must affect the exported winner without bypassing goals."""
import sqlite3

import pytest

from stash_ai.recognition.merge_solver_coverage import measure
from stash_ai.recognition.partition_delta import PartitionImpact, Predictions
from stash_ai.recognition.replay_simulation import impact
from stash_ai.recognition.solver_slot_index import Slots


@pytest.mark.parametrize('storage', ['dict', 'delta', 'slots'])
def test_coverage_counts_goal_and_non_goal_faces_and_bodies_once(storage):
    before = {'goal': 'a', 'goal-soft': None, 'body:free': None, 'face:free': None,
              'old-outsider': 'b', 'lost': 'c', 'rejected': None}
    after = dict(before, **{'goal-soft': 'a', 'body:free': 'a', 'face:free': 'new',
                           'old-outsider': 'a', 'lost': None})
    targets = {'goal', 'goal-soft', 'missing-scan'}
    groups = [dict(resulting_identities={'a': 2})]
    expected = dict(total=7, assigned=5, unassigned=2, gained=3, lost=1,
                    recovered_target_attachments=1)
    with sqlite3.connect(':memory:') as db:
        if storage == 'slots':
            db.execute('CREATE TABLE slots(id PRIMARY KEY, baseline)')
            db.executemany('INSERT INTO slots VALUES (?,?)', before.items())
            baseline = Slots(db, 'baseline', len(before))
        else:
            baseline = before
        if storage == 'dict':
            predicted = after
            outside = impact({k:v for k,v in before.items() if k not in targets},
                             {k:v for k,v in after.items() if k not in targets})
        else:
            delta = {k:v for k,v in after.items() if v != before[k]}
            predicted = Predictions(baseline, delta)
            outside = PartitionImpact(baseline, baseline, targets).measure(delta)
        assert measure(baseline, predicted, targets, outside, groups) == expected


def test_sparse_coverage_does_not_iterate_the_corpus():
    class NoIteration(dict):
        def __iter__(self):
            pytest.fail('Coverage must reuse partition counts and sparse deltas')
        def items(self):
            pytest.fail('Coverage must not materialize baseline rows')
    baseline = NoIteration({'goal': 'a', 'free': None, 'rejected': None})
    predicted = Predictions(baseline, {'free': 'a'})
    outside = dict(assigned=1, coverage_gained=1, coverage_lost=0)
    result = measure(baseline, predicted, {'goal'}, outside,
                     [dict(resulting_identities={'a': 1})])
    assert result['assigned'] == 2 and result['unassigned'] == 1
    assert result['recovered_target_attachments'] == 1


def recovery_fixture(tmp_path, policy='identity', exclusion=False):
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve
    config, original = auxiliary_goals(tmp_path)
    primary, query = original['groups'][0]['members']
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [[primary['performer_id']]], policy=policy,
            observation_exclusions=[(primary['performer_id'], query['observations'][0]['id'])] if exclusion else ())
    return config, goals


def proposals(monkeypatch, values):
    from stash_ai.recognition.merge_solver_tighten import Tightener
    patches = iter(values)
    def offer(self, winner, changes, feedback):
        return 'tightening', winner, 'coverage_regression_probe', next(patches)
    monkeypatch.setattr(Tightener, 'next', offer)


@pytest.mark.parametrize('policy', ['identity', 'strict'])
def test_recovery_beats_stricter_solved_trial_and_survives_resume(tmp_path, monkeypatch, capsys, policy):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_checkpoint import winner_file
    from stash_ai.recognition.merge_solver_report import compact_document
    from stash_ai.recognition.merge_goal_output import explain, summary
    from stash_ai.recognition.solver_verbose import verbose_progress
    config, goals = recovery_fixture(tmp_path, policy)
    recovery = {'sapiensid.matching.face_floor': .78}
    stricter = {'recognition.match_threshold': 1., 'clustering.cohesion_threshold': 1.}
    checkpoint, yaml = tmp_path/'solver.json', tmp_path/'winner.yml'
    source = config.database.read_bytes()
    for resume in (False, True):
        proposals(monkeypatch, [recovery, stricter])
        with _open_readonly(config.database) as db:
            result = solve(config, goals, db, maximum=3, checkpoint=checkpoint, resume=resume,
                           winner_writer=winner_file(yaml, resume=resume), progress=verbose_progress(3))
        assert 'assigned=6/6 unassigned=0 gained=3 lost=0' in capsys.readouterr().err
        assert result['selected_trial'] == 'trial-1'
        rows = result['simulation']['trials']
        assert all(row['goal_result']['fully_solved'] for row in rows.values())
        baseline, recovered, tightened = [rows[n]['goal_result'] for n in ('baseline','trial-1','trial-2')]
        assert baseline['assignment_coverage']['unassigned'] == 3
        assert recovered['assignment_coverage'] == dict(total=6, assigned=6, unassigned=0,
            gained=3, lost=0, recovered_target_attachments=3)
        assert recovered['groups'][0]['outsider_observations'] == 3
        assert tightened['assignment_coverage']['unassigned'] == 3
        assert result['best_result']['assignment_coverage'] == recovered['assignment_coverage']
        assert 'trial-1' in yaml.read_text()
        assert 'face_floor: 0.78' in yaml.read_text()
        if resume:
            assert result['search']['restored_trials'] == 3
            assert result['simulation']['trials'] == previous
        previous = rows
        document = dict(goals, **result)
        explain(document)
        compact = compact_document(document, config)
        for name, row in compact['simulation']['trials'].items():
            assert row['goal_result']['assignment_coverage'] == rows[name]['goal_result']['assignment_coverage']
        assert 'assigned=6/6 unassigned=0 gained=3 lost=0' in summary(result)
    assert config.database.read_bytes() == source


@pytest.mark.parametrize('guard', ['outside_limit', 'observation_exclusion'])
def test_more_assignments_cannot_override_explicit_constraints(tmp_path, monkeypatch, guard):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    config, goals = recovery_fixture(tmp_path, exclusion=guard == 'observation_exclusion')
    proposals(monkeypatch, [{'sapiensid.matching.face_floor': .78}])
    limits = {'new_outsider_observations': 0} if guard == 'outside_limit' else None
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=2, outside_limits=limits)
    assert result['selected_trial'] == 'baseline'
    recovered = result['simulation']['trials']['trial-1']['goal_result']
    assert recovered['assignment_coverage']['unassigned'] == 0
    assert not recovered['fully_solved']
