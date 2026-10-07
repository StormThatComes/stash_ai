"""Retention changes only finished diagnostic rows, atomically and by scope."""
from datetime import datetime, timezone
import sqlite3

import pytest

from stash_ai.history.store import HistoryStore
from stash_ai.history.retention import prune
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_temporal')


def populate(path):
    with HistoryStore(path) as store:
        for identifier, scope, finished in (
                ('old', 'scope', '2025-01-01T00:00:00+00:00'),
                ('mid', 'scope', '2026-09-01T00:00:00+00:00'),
                ('new', 'scope', '2026-10-06T00:00:00+00:00'),
                ('active', 'scope', None), ('other', 'other', '2025-01-01T00:00:00+00:00')):
            store.start(identifier, scope, 'inference', '2025-01-01T00:00:00+00:00', {'fingerprint': 'x'})
            store.record_decision(identifier, scope, identifier, 'wd', 'infer', 'finished', [], {})
            if finished:
                store.finish(identifier, finished, 'finished', 1, 1, models={}, counts={}, outcomes={}, metrics=[], issues={})
        store.put_proof('scope', 'proof', 'wd', 'scene', 'signature', {'keep': True})


NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


@pytest.mark.parametrize('criteria,expected', [
    ({'keep_last': 1}, {'old', 'mid'}),
    ({'max_age_days': 60}, {'old'}),
    ({'keep_last': 2, 'max_age_days': 30}, {'old', 'mid'}),
    ({'keep_last': 0}, {'old', 'mid', 'new'}),
])
def test_preview_and_delete_protect_active_other_scope_and_proofs(tmp_path, criteria, expected):
    path = tmp_path / 'history.sqlite3'
    populate(path)
    before = path.read_bytes()
    with HistoryStore(path, readonly=True) as store:
        preview = prune(store, 'scope', now=NOW, **criteria)
    assert set(preview['run_ids']) == expected
    assert preview['candidate_decisions'] == len(expected)
    assert path.read_bytes() == before
    with HistoryStore(path) as store:
        proof = store.connection.execute('SELECT * FROM reuse_proofs').fetchall()
        result = prune(store, 'scope', now=NOW, dry_run=False, **criteria)
        assert result['run_ids'] == preview['run_ids']
        ids = {row[0] for row in store.connection.execute('SELECT id FROM runs')}
        assert ids == {'old', 'mid', 'new', 'active', 'other'} - expected
        assert store.connection.execute('SELECT * FROM reuse_proofs').fetchall() == proof
        assert not store.connection.execute('SELECT 1 FROM reuse_decisions WHERE run_id NOT IN (SELECT id FROM runs)').fetchone()


def test_delete_failure_rolls_back_runs_and_decisions(tmp_path):
    path = tmp_path / 'history.sqlite3'
    populate(path)
    with HistoryStore(path) as store:
        db = store.connection
        db.execute("CREATE TRIGGER fail_retention BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT,'fail'); END")
        db.commit()
        before = {table: db.execute('SELECT * FROM '+table).fetchall() for table in ('runs', 'reuse_decisions', 'reuse_proofs')}
        with pytest.raises(sqlite3.IntegrityError):
            prune(store, 'scope', keep_last=0, dry_run=False)
        assert not db.in_transaction
        assert before == {table: db.execute('SELECT * FROM '+table).fetchall() for table in before}


def test_active_writer_prevents_retention_and_later_finished_run_is_kept(tmp_path):
    path = tmp_path / 'history.sqlite3'
    populate(path)
    writer = sqlite3.connect(path)
    try:
        with HistoryStore(path) as store:
            store.connection.execute('PRAGMA busy_timeout=10')
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("UPDATE runs SET finished_at=?,status='finished' WHERE id='active'", ('2026-10-07T00:00:00+00:00',))
            with pytest.raises(sqlite3.OperationalError):
                prune(store, 'scope', keep_last=1, dry_run=False)
        writer.commit()
        with HistoryStore(path) as store:
            result = prune(store, 'scope', keep_last=1, dry_run=False)
            assert 'active' not in result['run_ids']
            assert store.connection.execute("SELECT 1 FROM runs WHERE id='active'").fetchone()
    finally:
        writer.close()


def test_retention_cli_is_offline_and_source_state_stays_identical(tmp_path, monkeypatch):
    from stash_ai.history.store import history_path
    from stash_ai.storage.store import scope_for
    path, state = tmp_path / 'history.sqlite3', tmp_path / 'state.sqlite3'
    state.write_bytes(b'source state must never be opened')
    with HistoryStore(path) as store:
        store.start('one', scope_for(app.STASH_URL), 'inference', '2026-01-01T00:00:00+00:00', {'fingerprint': 'x'})
        store.finish('one', '2026-01-01T00:00:00+00:00', 'finished', 1, 1, models={}, counts={}, outcomes={}, metrics=[], issues={})
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *a: pytest.fail('Stash opened'))
    args = ['--prune-run-history', '--history-keep-last', '0', '--run-history-db', str(path), '--state-db', str(state)]
    before = path.read_bytes()
    assert app.main(args + ['--dry-run']) == 0
    assert path.read_bytes() == before
    assert app.main(args) == 0
    assert state.read_bytes() == b'source state must never be opened'
    with HistoryStore(path, readonly=True) as store:
        assert not store.connection.execute('SELECT 1 FROM runs').fetchone()


@pytest.mark.parametrize('args', [
    ['--prune-run-history'], ['--prune-run-history', '--history-keep-last', '-1'],
    ['--prune-run-history', '--history-max-age-days', 'nan'],
    ['--history-keep-last', '3'], ['--prune-run-history', '--history-keep-last', '1', '--scene-id', '1'],
    ['--prune-run-history', '--history-keep-last', '1', '--restart'],
])
def test_invalid_retention_options_are_rejected(args):
    from stash_ai.commands.arguments import parse_args
    with pytest.raises(SystemExit):
        parse_args(app, args)
