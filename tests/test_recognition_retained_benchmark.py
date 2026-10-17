"""Real-evidence benchmark input stays read-only and emits bounded hashes/counts."""
import hashlib
import json
from dataclasses import replace
import sqlite3

import pytest

from benchmarks import benchmark_recognition_retained as benchmark
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_clustering import setup
from benchmarks.benchmark_recognition_replay import fixture


def test_retained_benchmark_never_opens_a_writer_or_reads_images(tmp_path, monkeypatch):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        fixture(state, 20, 128, 16)
        state.connection.execute("UPDATE observations SET source='https://private.invalid/secret-token', quality=?,crop=?",
                                 ('{"private":"quality"}', b'private JPEG payload'))
        state.connection.commit()
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}
    original = benchmark._open_readonly
    def readonly(path):
        db = original(path)
        db.set_authorizer(lambda action, table, column, *_: sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and (table == 'observations' and column in ('quality', 'crop')
                                                 or table == 'portraits' and column == 'image') else sqlite3.SQLITE_OK)
        return db
    monkeypatch.setattr(benchmark, '_open_readonly', readonly)
    monkeypatch.setattr(FaceDatabase, '__init__', lambda *_: pytest.fail('writer constructor used'))
    data = plain(config.data)
    data['execution']['rebuild_checkpoint_scenes'] = 1  # Benchmark must override disk checkpoints.
    result = benchmark.run(replace(config, data=data), all_scenes=True, namespace='ns', workers=(1, 2))
    assert result['identical_semantics'] and result['source_read_only']
    assert result['manifest']['scenes'] == 20
    assert {t['mode'] for t in result['trials']} == {'cold', 'cached'}
    assert all(t['memory']['method'] == 'sampled-process-rss' for t in result['trials'])
    assert all(t['profile']['cpu']['submitted_blocks'] == 0 for t in result['trials'] if t['mode'] == 'cached')
    text = json.dumps(result)
    assert not any(v in text for v in ('private.invalid', 'secret-token', 'private JPEG', 'private"'))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}
    assert before == after
    assert not list(tmp_path.glob('*replay*'))


def test_latest_partial_scan_cannot_fall_back_to_older_complete_evidence(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        fixture(state, 16, 128, 16)
        state.connection.execute("INSERT INTO scans SELECT 'newer',scope,scene,namespace,'new-input',policy_sha,"
                                 "manifest,coverage,total,0,'partial' FROM scans WHERE scene='1'")
        state.connection.commit()
    with pytest.raises(ValueError, match='latest complete'):
        benchmark.run(config, ['1'], namespace='ns', workers=(1,))
    result = benchmark.run(config, all_scenes=True, namespace='ns', workers=(1,))
    assert result['manifest']['scenes'] == 15
    assert result['manifest']['skipped_latest_incomplete'] == 1


def test_foreign_or_executable_schema_is_refused_without_modification(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        fixture(state, 16, 128, 16)
        state.connection.execute('CREATE VIEW foreign_view AS SELECT * FROM scans')
        state.connection.commit()
    before = config.database.read_bytes()
    with pytest.raises(ValueError, match='triggers or views'):
        benchmark.run(config, all_scenes=True, namespace='ns', workers=(1,))
    assert config.database.read_bytes() == before
