from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import sqlite3
import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.replay_simulation import prepared_history
from stash_ai.recognition.offline_evaluation import _ram
from stash_ai.recognition.config import plain


def test_automatic_admission_reserves_headroom_and_limits_parallelism(tmp_path, monkeypatch):
    from stash_ai.recognition import solver_execution as execution
    config, _ = fixture(tmp_path, 'face')
    monkeypatch.setattr(execution.os, 'process_cpu_count', lambda:16)
    ceiling = execution.plan(config).trial_bytes
    monkeypatch.setattr(execution, 'available_memory', lambda:ceiling*3)
    with pytest.raises(ValueError, match='resources admit only 2'):
        execution.plan(config, dict(cpus=8,trials=8))
    admitted = execution.plan(config, dict(cpus=8,trials=2))
    assert admitted.trials == 2 and admitted.memory_bytes == ceiling*3*4//5
    monkeypatch.setattr(execution, 'available_memory', lambda:1024)
    with pytest.raises(ValueError, match='resources admit only 1'):
        execution.plan(config, dict(cpus=8,trials=8))
    admitted = execution.plan(config, dict(cpus=8,trials=1))
    assert admitted.trials == 1 and admitted.memory_admission == 'automatic_serial_ceiling_exceeds_available'
    with pytest.raises(ValueError, match='resources admit only 4'):
        execution.plan(config, dict(cpus=8,trials=8,memory_bytes=ceiling*4))
    assert execution.plan(config, dict(cpus=8,trials=4,memory_bytes=ceiling*4)).trials == 4


def test_inner_replay_shares_vectors_and_lookahead_sees_private_metadata(tmp_path):
    from stash_ai.recognition.trial_inputs import prepare
    from stash_ai.recognition.rebuild import _copy_selection
    from stash_ai.recognition.replay_memory import memory_replay, Lookahead
    config, goals = fixture(tmp_path, 'face')
    data = plain(config.data); data['execution']['rebuild_prefetch_scenes'] = True
    config = replace(config, data=data)
    original = config.database.read_bytes()
    with closing(_open_readonly(config.database)) as source:
        installation = source.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
        with prepared_history(source, config, goals['scope'], goals['namespace'], installation) as seed, closing(_ram(config)) as trial:
            prepare(trial, seed)
            selected = [dict(scene=r['scene'],scan=r['id']) for r in trial.execute('SELECT * FROM scans')]
            with memory_replay(config) as (inner, uri):
                _copy_selection(SimpleNamespace(connection=trial),inner,selected,goals['scope'],goals['namespace'])
                assert inner.execute('SELECT SUM(length(embedding)) FROM main.observations').fetchone()[0] is None
                assert inner.execute('SELECT SUM(length(embedding)) FROM observations').fetchone()[0] > 0
                inner.execute("UPDATE main.observations SET quality='{\"private\":true}'")
                inner.commit()
                assert all(r[0]=='{"private":true}' for r in inner.execute('SELECT quality FROM observations'))
                with Lookahead(uri, config) as ahead:
                    ahead.submit(selected[0]['scan']); ahead.take(selected[0]['scan'])
                    assert ahead.prepared and all(row[1] for row in ahead.prepared.values())
    assert config.database.read_bytes() == original


def test_fingerprints_reuse_only_inside_immutable_snapshot(tmp_path):
    from stash_ai.recognition import replay_signature as signatures
    config, goals = fixture(tmp_path, 'face')
    with closing(_open_readonly(config.database)) as source:
        installation = source.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
        with patch.object(signatures,'_fingerprint',wraps=signatures._fingerprint) as call:
            with prepared_history(source, config, goals['scope'], goals['namespace'], installation) as seed:
                one = signatures.fingerprint(seed,goals['scope'],goals['namespace'])
                assert signatures.fingerprint(seed,goals['scope'],goals['namespace']) == one
                assert call.call_count == 1
            signatures.fingerprint(source,goals['scope'],goals['namespace'])
            signatures.fingerprint(source,goals['scope'],goals['namespace'])
            assert call.call_count == 3


def test_bulk_import_restores_triggers_and_rolls_back_on_failure():
    from stash_ai.recognition.snapshot_import import bulk
    from stash_ai.recognition.database import SCHEMA
    with sqlite3.connect(':memory:') as db:
        db.executescript(SCHEMA)
        triggers=list(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name"))
        with pytest.raises(RuntimeError):
            with bulk(db):
                db.execute("INSERT INTO metadata VALUES ('installation','rolled-back')")
                raise RuntimeError('failed input copy')
        assert not db.execute('SELECT * FROM metadata').fetchall()
        assert list(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name")) == triggers
        with bulk(db):
            db.execute("INSERT INTO metadata VALUES ('installation','bulk')")
        assert not db.execute('SELECT * FROM evidence_versions').fetchall()
        db.execute("UPDATE metadata SET value='next'")
        assert db.execute('SELECT * FROM evidence_versions').fetchall()
