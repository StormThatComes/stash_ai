from pathlib import Path

from stash_ai.recognition.config import load, plain
from stash_ai.recognition.solver_execution import plan, scope
from stash_ai.recognition.compute import Compute
from stash_ai.recognition.trial_evaluation import settings_for_replay


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def test_profiles_preserve_models_matching_and_input_contracts():
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract as body_contract

    run = load(ROOT / 'facial_recognition.yml')
    solver = load(ROOT / 'facial_recognition_solver.yml')
    run_data, solver_data = plain(run.data), plain(solver.data)
    # These are the only resource groups outside the top-level execution map.
    for data in (run_data, solver_data):
        data.pop('execution')
        data['clustering'].pop('max_reference_bytes')
        data['sapiensid'].pop('execution')
    assert run_data == solver_data
    assert namespace(run) == namespace(solver)
    assert scan_identity(run, namespace(run)) == scan_identity(solver, namespace(solver))
    assert person_contract(run) == person_contract(solver)
    assert body_contract(run) == body_contract(solver)
    assert run.database == solver.database
    assert run.model_dir == solver.model_dir


def test_shipped_solver_profile_admits_four_trials_without_shrinking_working_memory(monkeypatch):
    monkeypatch.setattr('stash_ai.recognition.solver_execution.os.process_cpu_count', lambda: 24, raising=False)
    config = load(ROOT / 'facial_recognition_solver.yml')
    # No source-size discount: validate the worst-case configured reference limits.
    runtime = plan(config, dict(cpus=16, trials=4, memory_bytes=64 * GIB))
    assert runtime.trials == runtime.requested_trials == 4
    assert runtime.rebuild_memory_bytes == config.data['execution']['rebuild_memory_bytes'] == 10 * GIB
    assert runtime.trial_bytes * runtime.trials == 59 * GIB
    assert runtime.memory_bytes - runtime.trial_bytes * runtime.trials == 5 * GIB
    replay = settings_for_replay(config, runtime)
    assert replay.data['execution']['rebuild_storage'] == 'disk'
    assert replay.data['execution']['rebuild_cohort_backend'] == 'threads'
    assert replay.data['execution']['rebuild_nomination_device'] == 'cpu'
    assert replay.data['execution']['rebuild_prefetch_scenes'] is True
    with scope(runtime):
        pool = Compute(replay)
        assert pool.workers == 3 and pool.native_threads == 1
        replay.data['execution']['rebuild_parallel_scenes'] = False
        from concurrent.futures import Future
        from stash_ai.recognition.replay_memory import Lookahead
        with Lookahead(':memory:', replay) as reader:
            reader.pending = Future()
            assert pool.workers == 2
            reader.pending.set_result({})
            assert pool.workers == 3
