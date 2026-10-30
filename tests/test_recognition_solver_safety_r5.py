"""Search failures and reports never compromise the retained input snapshot."""
from contextlib import closing
import json
import sqlite3

import pytest

from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import fingerprint, resolve
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.resource_errors import ResourceLimit
from tests.test_recognition_merge_report import seeded


@pytest.mark.parametrize('execution', [None, {'cpus': 1, 'trials': 1}])
def test_candidate_resource_failure_preserves_baseline(tmp_path, monkeypatch, execution):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition import trial_evaluation
    config, goals = fixture(tmp_path, 'face')
    original = trial_evaluation.evaluate
    def evaluate(name, *args, **kwargs):
        if name != 'baseline':
            raise ResourceLimit('candidate limit')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(trial_evaluation, 'evaluate', evaluate)
    with closing(_open_readonly(config.database)) as db:
        result = solve(config, goals, db, maximum=2, feasibility_trials=0,
                       targeted_trials=0, execution=execution)
    assert result['selected_trial'] == 'baseline'
    assert result['search']['failed_trials'] == 1


def test_optional_hints_can_be_incomplete_without_aborting_baseline(tmp_path, monkeypatch):
    from stash_ai.recognition import merge_solver_hybrid, body_solver
    def limited(*args, **kwargs):
        raise ResourceLimit('diagnostic limit')
    monkeypatch.setattr(merge_solver_hybrid, 'analyse', limited)
    monkeypatch.setattr(body_solver, 'analyse', limited)
    config, _ = seeded(tmp_path)
    with closing(_open_readonly(config.database)) as db:
        goals = resolve(db, config, [['900', '901']])
        result = solve(config, goals, db, maximum=1)
    assert result['selected_trial'] == 'baseline'
    assert 'diagnostic_resource_limit' in json.dumps(result)


def test_detached_report_releases_source_and_remains_consistent(tmp_path):
    from stash_ai.recognition.report_snapshot import snapshot
    config, _ = seeded(tmp_path)
    with snapshot(config, '900', by_performer=True) as db:
        before = db.execute('SELECT performer FROM people ORDER BY id').fetchall()
        with sqlite3.connect(config.database, timeout=.05) as writer:
            writer.execute("UPDATE people SET performer='changed'")
        assert db.execute('SELECT performer FROM people ORDER BY id').fetchall() == before
        with pytest.raises(sqlite3.OperationalError):
            db.execute("UPDATE people SET performer='forbidden'")


def test_report_fingerprint_includes_reference_roles(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    config, _ = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        before = fingerprint(db, config, 'scope')
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_role','attachment')")
        assert fingerprint(db, config, 'scope') != before
