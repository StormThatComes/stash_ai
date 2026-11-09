from dataclasses import replace
import pickle
from pathlib import Path

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.config import plain
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.models import namespace
from stash_ai.recognition.offline_evaluation import _selection
from stash_ai.recognition.replay_simulation import prepared_history, replay
from stash_ai.recognition.merge_solver_feasibility import Feasibility
from stash_ai.recognition.solver_execution import plan
from stash_ai.recognition.solver_trial import Request, evaluate


def test_isolated_trial_matches_production_and_preserves_inputs(tmp_path):
    config, goals = fixture(tmp_path, 'query')
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        scope, scans = _selection(db, config, None, goals['scope'])
        observations = dict(db.execute('SELECT o.id,s.scene FROM observations o JOIN scans s ON s.id=o.scan'))
        installation = db.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
        frozen = Feasibility(db, config, goals, scans)
        expected = {}
        replay(config, {}, source=db, consume=lambda n, o, r, p: expected.update(p) or {})
        with prepared_history(db, config, scope, namespace(config), installation) as seed:
            seed_path = seed.execute('PRAGMA database_list').fetchone()[2]
            request = Request('baseline', replace(config, data=plain(config.data)), seed_path,
                scope, namespace(config), installation, scans, observations, goals, frozen,
                (), None, True, {}, {}, plan(config, dict(cpus=1)))
            decoded = pickle.loads(pickle.dumps(request))
            first = evaluate(decoded)
            second = evaluate(decoded)
    assert first['predictions'] == second['predictions'] == expected
    assert first['feedback'] == second['feedback']
    assert request.visits == {} and request.priorities == {}
    assert config.database.read_bytes() == before


def test_trial_journal_tracks_inflight_configuration(tmp_path):
    import pytest
    from stash_ai.recognition.merge_solver_checkpoint import Journal
    from stash_ai.recognition.merge_report_policy import patched
    config, _ = fixture(tmp_path/'input', 'face')
    path = tmp_path/'journal.json'
    journal = Journal(path, {'test': 1})
    journal.task('trial-1', config, 'queued')
    journal.task('trial-1', config, 'running')
    journal.close()
    resumed = Journal(path, {'test': 1}, resume=True)
    assert resumed.document['tasks']['trial-1']['status'] == 'running'
    with pytest.raises(ValueError, match='configuration changed'):
        resumed.task('trial-1', patched(config, {'recognition.match_threshold': .9}), 'queued')


def test_raw_result_checks_configuration_and_checksum(tmp_path):
    import pytest
    from stash_ai.recognition.merge_solver_checkpoint import Journal
    from stash_ai.recognition.merge_report_policy import patched
    config, _ = fixture(tmp_path/'input', 'face')
    journal = Journal(tmp_path/'journal.json', {'test': 1})
    journal.save_raw_result('trial-1', config, {'result': 3})
    assert journal.raw_result('trial-1', config) == {'result': 3}
    with pytest.raises(ValueError, match='configuration changed'):
        journal.raw_result('trial-1', patched(config, {'recognition.match_threshold': .9}))
    entry = journal.document['raw_results']['trial-1']
    (journal.directory/entry['file']).write_text('{}')
    with pytest.raises(ValueError, match='Corrupt'):
        journal.raw_result('trial-1', config)


def test_dense_delta_survives_worker_cleanup_and_journal_resume(tmp_path, monkeypatch):
    from stash_ai.recognition.solver_delta_transport import collect, worker_target, Delta
    from stash_ai.recognition.merge_solver_checkpoint import Journal
    from types import SimpleNamespace
    config = SimpleNamespace(data={'test': 1})
    directory = tmp_path/'worker'
    directory.mkdir()
    monkeypatch.setattr('stash_ai.recognition.solver_delta_transport.INLINE_ROWS', 2)
    with worker_target(directory, 1024**2):
        delta = collect(iter([('a', 'p'), ('b', None), ('c', 'q')]))
    assert isinstance(delta, Delta)
    journal = Journal(tmp_path/'state.json', {'test': 1})
    journal.save_raw_result('trial-1', config, {'prediction_delta':delta}, delta_directory=directory)
    journal.close()
    (directory/delta.filename).unlink()
    journal = Journal(tmp_path/'state.json', {'test': 1}, resume=True)
    try:
        result = journal.raw_result('trial-1')['prediction_delta']
        assert result.journal
        assert result.read(journal.directory) == {'a':'p','b':None,'c':'q'}
    finally:
        journal.close()


def test_cpu_loans_leave_room_for_a_second_live_coordinator():
    import multiprocessing
    from stash_ai.recognition import solver_cpu_loans as loans
    state = multiprocessing.get_context('spawn').Array('i', (12, 0, 0))
    loans.initialize(state)
    try:
        with loans.active(), loans.active(), loans.borrow() as first, loans.borrow() as second:
            assert first == second == 6
            assert not loans.prefetch()
        with loans.active(), loans.borrow() as lone:
            assert lone == 12
        assert state[:] == [12, 0, 0]
    finally:
        loans.initialize(None)
