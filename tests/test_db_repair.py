"""Fast no-op, explicit repair and atomic data-revision completion."""
import sqlite3

import pytest

from stash_ai.db import update, duplicate_aggregates
from stash_ai.db.repair import repair_database
from stash_ai.db.checks import check_database
from stash_ai.storage import codec
from stash_ai.storage.store import StateStore
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from tests.test_db_update import seed_version, rows
from tests.helpers.runtime import runtime as app
from tests.test_evidence_integrity import record_with_duplicate
from stash_ai.evidence.identity import marker_segment_signature


def test_complete_update_has_no_blob_decode_integrity_scan_lock_or_mutation(tmp_path, monkeypatch):
    path = tmp_path/'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    before, statements = path.read_bytes(), []
    connect = update.connect_readonly
    def readonly(*args):
        db = connect(*args)
        db.set_trace_callback(statements.append)
        return db
    monkeypatch.setattr(update, 'connect_readonly', readonly)
    monkeypatch.setattr(codec, 'decode', lambda *_: pytest.fail('no-op decoded a BLOB'))
    monkeypatch.setattr(update, 'StateDatabase', lambda *_a, **_k: pytest.fail('no-op opened a writer'))
    assert update.update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)
    assert not any('quick_check' in query.lower() or 'integrity_check' in query.lower() for query in statements)
    assert path.read_bytes() == before


@pytest.mark.parametrize('dry', [False, True])
def test_explicit_repair_fixes_metadata_once_and_dry_run_is_read_only(tmp_path, monkeypatch, dry):
    path = tmp_path/'state.sqlite3'
    seed_version(path, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE result_metadata SET data='{}'")  # Valid JSON, semantically broken.
    before = path.read_bytes()
    decode, calls = duplicate_aggregates.decode, []
    def counted(data):
        calls.append(1)
        return decode(data)
    monkeypatch.setattr(duplicate_aggregates, 'decode', counted)
    result = repair_database(path, dry_run=dry)
    assert result['metadata_repaired'] == result['results_checked'] == len(calls) == 2
    if dry:
        assert path.read_bytes() == before
    else:
        assert check_database(path, 'scope-a')['ok']


@pytest.mark.parametrize('conflict', [False, True])
def test_repair_preserves_pins_and_rolls_back_conflicting_duplicates(tmp_path, conflict):
    path = tmp_path/'state.sqlite3'
    raw = record_with_duplicate()
    if conflict:
        raw['items'][1]['score'] = .5
    signature = marker_segment_signature(raw)
    published = dict(owned_ids=['native'], contributions=[dict(raw['items'][0], source='wd',
        meta=raw['meta'], evidence_signature=signature)])
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', {'scene': {**raw, 'items': raw['items'][:1]}})
        store.connection.execute('UPDATE results SET data=?', (codec.encode(raw),))
        store.publish('1', {'scene': published})
        store.prepare('1', {'scene': published})
        store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
            ('scope', '1', 'wd', 'scene', signature, codec.encode(raw)))
        store.connection.commit()
    before = {table: rows(path, table) for table in ('results', 'snapshots', 'inference_archive', 'result_metadata')}
    if conflict:
        with pytest.raises(ValueError, match='Conflicting'):
            repair_database(path)
        assert before == {table: rows(path, table) for table in before}
    else:
        result = repair_database(path)
        assert result['duplicates_repaired'] == 2
        assert check_database(path, 'scope')['ok']
        with StateStore(path, 'scope') as store:
            raw = store.model_record('1', 'scene', 'wd')
            assert len(raw['items']) == 1
            for value in store.snapshot('1', 'scene'):
                assert value['owned_ids'] == ['native']
                assert value['contributions'][0]['evidence_signature'] == marker_segment_signature(raw)


def test_failed_required_conversion_never_marks_data_revision(tmp_path, monkeypatch):
    path = tmp_path/'state.sqlite3'
    seed_version(path, 26)
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM data_revisions')
    monkeypatch.setattr(duplicate_aggregates, 'migrate', lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError('required conversion failed')))
    with pytest.raises(RuntimeError, match='required conversion failed'):
        update.update_database(path)
    with sqlite3.connect(path) as db:
        assert not db.execute('SELECT * FROM data_revisions').fetchall()
        assert db.execute('PRAGMA user_version').fetchone() == (26,)


def test_ddl_only_upgrade_with_completed_revision_has_no_full_repair(tmp_path, monkeypatch):
    path = tmp_path/'state.sqlite3'
    seed_version(path, 26)
    monkeypatch.setattr(duplicate_aggregates, 'migrate', lambda *_a, **_k: pytest.fail('unnecessary global repair'))
    assert update.update_database(path) == (26, LATEST_SCHEMA_VERSION)


def test_repair_cli_stays_offline_and_requires_current_schema(tmp_path, monkeypatch):
    path = tmp_path/'state.sqlite3'
    seed_version(path, 26)
    for name in ('create_stash_client', 'load_tagger', 'load_vocabulary', 'load_model_settings'):
        monkeypatch.setattr(app.services, name, lambda *_a, **_k: pytest.fail('repair used Stash/models'))
    assert app.main(['--db-repair', '--state-db', str(path)]) == 1
    update.update_database(path)
    assert app.main(['--db-repair', '--state-db', str(path), '--dry-run']) == 0
    assert app.main(['--db-repair', '--state-db', str(path)]) == 0
