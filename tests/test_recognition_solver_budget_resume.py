"""Budget extensions replay the old schedule, not the old production work."""
import json

import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_checkpoint import winner_file


@pytest.mark.parametrize('interrupted', [False, True])
@pytest.mark.parametrize('sampler', ['heuristic', 'tpe'])
def test_serial_extensions_preserve_completed_trials_and_owned_winner(tmp_path, monkeypatch, interrupted, sampler):
    import stash_ai.recognition.replay_simulation as replay
    config, goals = fixture(tmp_path/'input', 'query')
    path, yaml = tmp_path/'checkpoint.json', tmp_path/'best.yml'
    original = config.database.read_bytes()
    calls = []
    rebuild = replay.rebuild_selected
    def count(*args, **kwargs):
        calls.append(1)
        return rebuild(*args, **kwargs)
    monkeypatch.setattr(replay, 'rebuild_selected', count)
    def stop(name, phase, item):
        if interrupted and name == 'trial-2' and phase == 'complete':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, maximum=7, sampler=sampler, checkpoint=path,
                      winner_writer=winner_file(yaml), progress=stop)
        calls.clear()
        extended = solve(config, goals, db, maximum=16, sampler=sampler, checkpoint=path,
                         resume=True, winner_writer=winner_file(yaml, resume=True))
        old = first['simulation']['trials']
        all_trials = extended['simulation']['trials']
        assert extended['search']['restored_trials'] == len(old)
        assert len(calls) == len(all_trials)-len(old) > 0
        assert all(all_trials[name] == trial for name, trial in old.items())
        assert extended['search']['budget_history'] == [7, 16]
        assert extended['selected_trial'] in yaml.read_text()
        calls.clear()
        repeated = solve(config, goals, db, maximum=16, sampler=sampler, checkpoint=path,
                         resume=True, winner_writer=winner_file(yaml, resume=True))
        assert not calls and repeated['simulation']['trials'] == all_trials
        third = solve(config, goals, db, maximum=20, sampler=sampler, checkpoint=path,
                      resume=True, winner_writer=winner_file(yaml, resume=True))
        assert third['search']['budget_history'] == [7, 16, 20]
        assert third['search']['restored_trials'] == len(all_trials)
        assert third['search']['attempted'] <= 20
        signatures = [json.dumps(t.get('yaml_patch', {}), sort_keys=True)
                      for t in third['simulation']['trials'].values()]
        assert len(set(signatures)) == len(signatures)
    assert config.database.read_bytes() == original


@pytest.mark.parametrize('scheduling', ['batch', 'async'])
@pytest.mark.parametrize('resource_failure', [False, True])
def test_eight_worker_extension_reconstructs_dispatch_and_retirement(tmp_path, monkeypatch, scheduling, resource_failure):
    from stash_ai.recognition import solver_execution
    from stash_ai.recognition.resource_errors import ResourceLimit
    from stash_ai.recognition.solver_pool import Trials
    monkeypatch.setattr(solver_execution.os, 'process_cpu_count', lambda: 16, raising=False)
    unpack, injected = Trials.unpack, []
    def fail_first_result(self, result, **kwargs):
        # Keep real workers and dispatch order; fail one coordinator admission
        # deterministically instead of depending on the CI runner's resources.
        evaluated = unpack(self, result, **kwargs)
        if resource_failure and not injected:
            injected.append(True)
            raise ResourceLimit('Synthetic parallel trial resource limit')
        return evaluated
    monkeypatch.setattr(Trials, 'unpack', fail_first_result)
    config, goals = fixture(tmp_path/'input', 'disjoint')
    path = tmp_path/'checkpoint.json'
    execution = dict(cpus=16, trials=8, scheduling=scheduling, memory_bytes=16*1024**3)
    def stop(name, phase, item):
        candidate = name != 'baseline' if resource_failure else name == 'trial-2'
        if candidate and phase == 'complete':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, maximum=10, checkpoint=path, progress=stop, execution=execution)
        assert first['interrupted']
        failed = first['simulation'].get('failed_trials', {})
        if resource_failure:
            assert injected and any(t['error'] == 'Synthetic parallel trial resource limit' for t in failed.values())
        assert first['search']['attempted'] == len(first['simulation']['trials']) + len(failed)
        extended = solve(config, goals, db, maximum=18, checkpoint=path, resume=True, execution=execution)
        repeated = solve(config, goals, db, maximum=18, checkpoint=path, resume=True, execution=execution)
    # Durable resource failures consume a trial and are restored without retry.
    assert extended['search']['restored_trials'] == first['search']['attempted']
    assert all(extended['simulation']['failed_trials'][name] == trial for name, trial in failed.items())
    assert repeated['simulation']['trials'] == extended['simulation']['trials']
    assert repeated['simulation'].get('failed_trials', {}) == extended['simulation'].get('failed_trials', {})
    assert repeated['search']['restored_trials'] == extended['search']['attempted']
    assert repeated['search']['attempted'] <= 18 and repeated['best_result']['fully_solved']
    document = json.loads(path.read_text())
    assert document['budgets'] == [10, 18]
    active = peak = 0
    for event in document['schedule']:
        active += 1 if event['kind'] == 'dispatch' else -1
        peak = max(peak, active)
    assert active == 0 and peak == 8


def test_extension_rejects_budget_reduction_and_other_search_changes(tmp_path):
    config, goals = fixture(tmp_path/'input', 'query')
    path = tmp_path/'checkpoint.json'
    with _open_readonly(config.database) as db:
        solve(config, goals, db, maximum=1, checkpoint=path)
        solve(config, goals, db, maximum=3, checkpoint=path, resume=True)
        for options in [dict(maximum=2), dict(maximum=5, seed=42), dict(maximum=5, stop_on_stagnation=True)]:
            with pytest.raises(ValueError, match='incompatible'):
                solve(config, goals, db, checkpoint=path, resume=True, **options)
    assert json.loads(path.read_text())['budgets'] == [1, 3]


def test_extension_after_early_stop_keeps_unconsumed_candidates(tmp_path, monkeypatch):
    from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX
    from stash_ai.recognition.merge_solver_bidirectional import BidirectionalSearch
    from stash_ai.recognition.merge_solver_polish import Polisher
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    config, goals = auxiliary_goals(tmp_path)
    path = tmp_path/'checkpoint.json'
    monkeypatch.setattr(BidirectionalSearch, 'next', lambda *a, **k: None)
    def offer(self):
        if self.maximum <= 20 and self.executed >= 1:
            self.stop_reason = 'stagnation_after_refinement'
            raise StopIteration
        floor = .81 if self.executed == 0 else .78
        return 'hybrid', 'baseline_exploration', 'baseline', 'coordinate_probe', {PREFIX+'face_floor': floor}
    def no_polish(*args):
        raise StopIteration
    monkeypatch.setattr(BeamSearch, 'next', offer)
    monkeypatch.setattr(Polisher, 'next', no_polish)
    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, maximum=20, targeted_trials=0, feasibility_trials=0, checkpoint=path)
        assert first['search']['executed'] == 2 and not first['best_result']['fully_solved']
        extended = solve(config, goals, db, maximum=24, targeted_trials=0, feasibility_trials=0,
                         checkpoint=path, resume=True)
        repeated = solve(config, goals, db, maximum=24, targeted_trials=0, feasibility_trials=0,
                         checkpoint=path, resume=True)
    assert extended['best_result']['fully_solved'] and extended['search']['restored_trials'] == 2
    assert repeated['simulation']['trials'] == extended['simulation']['trials']


@pytest.mark.parametrize('parallel', [False, True])
def test_extension_preserves_anchor_comparisons_and_strict_input_control(tmp_path, parallel):
    config, goals = fixture(tmp_path/'input', 'query')
    config.data['recognition']['min_identity_support'] = 1
    path = tmp_path/'checkpoint.json'
    options = dict(strict=True, compare_anchor_rules=True,
                   execution=dict(cpus=4, trials=2, memory_bytes=8*1024**3) if parallel else None)
    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, maximum=6, checkpoint=path, **options)
        extended = solve(config, goals, db, maximum=12, checkpoint=path, resume=True, **options)
        repeated = solve(config, goals, db, maximum=12, checkpoint=path, resume=True, **options)
    assert 'input_control' in first['simulation']
    assert extended['search']['restored_trials'] == first['search']['attempted']
    assert repeated['simulation']['trials'] == extended['simulation']['trials']
    assert extended['search']['attempted'] <= 12
    signatures = [json.dumps(t.get('yaml_patch', {}), sort_keys=True)
                  for t in extended['simulation']['trials'].values()]
    assert len(set(signatures)) == len(signatures)


def test_failed_probe_consumes_budget_and_is_not_retried_on_extension(tmp_path, monkeypatch):
    import stash_ai.recognition.replay_simulation as replay
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.resource_errors import ResourceLimit
    from tests.test_recognition_merge_report import seeded
    (tmp_path/'input').mkdir()
    config, _ = seeded(tmp_path/'input')
    path = tmp_path/'input'/'checkpoint.json'
    original, calls = replay.rebuild_selected, []
    def guarded(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ResourceLimit('Synthetic probe resource limit')
        return original(*args, **kwargs)
    monkeypatch.setattr(replay, 'rebuild_selected', guarded)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900', '901']])
        first = solve(config, goals, db, maximum=6, targeted_trials=0, checkpoint=path)
        assert first['search']['failed_trials'] == 1
        count = len(calls)
        extended = solve(config, goals, db, maximum=10, targeted_trials=0, checkpoint=path, resume=True)
    assert extended['search']['restored_trials'] == 6
    assert extended['search']['failed_trials'] == 1
    assert len(calls)-count == extended['search']['attempted']-6
    assert extended['search']['attempted'] <= 10


def test_budget_extension_event_survives_failure_before_snapshot(tmp_path, monkeypatch):
    from stash_ai.recognition.merge_solver_checkpoint import Journal
    from stash_ai.recognition.solver_journal_events import Events, load_document
    path = tmp_path/'checkpoint.json'
    Journal(path, {'settings': {'maximum': 1}}).close()
    append = Events.append
    def interrupted(self, event):
        append(self, event)
        if event['kind'] == 'budget':
            raise OSError('after durable extension')
    with monkeypatch.context() as patch:
        patch.setattr(Events, 'append', interrupted)
        with pytest.raises(OSError, match='durable extension'):
            Journal(path, {'settings': {'maximum': 4}}, resume=True)
    assert load_document(path)['budgets'] == [1, 4]
    resumed = Journal(path, {'settings': {'maximum': 4}}, resume=True)
    assert resumed.budgets == (1, 4)
    resumed.close()
