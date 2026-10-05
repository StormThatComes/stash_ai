"""Schema 17 history upgrades preserve thresholds, pins and unfinished plans."""
from copy import deepcopy
import sqlite3

import pytest

from stash_ai import db
from stash_ai.db.result_history import encode_thresholds
from stash_ai.db.single_record_migration import install_triggers
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.thresholds import indexed_signature
from stash_ai.storage.archive import records as archived_records
from stash_ai.storage.codec import encode, decode
from stash_ai.storage.metadata import FORMAT, decode as decode_metadata
from stash_ai.storage.progress import completed
from stash_ai.storage.store import StateStore
from tests.test_inference_archive import proof
from tests.test_result_thresholds import outputs


def old_threshold_table(connection):
    connection.execute('DROP TRIGGER results_delete_threshold')
    connection.execute('DROP TABLE result_thresholds')
    connection.execute('CREATE TABLE result_thresholds ('
        'scope TEXT NOT NULL,scene TEXT NOT NULL,model TEXT NOT NULL,kind TEXT NOT NULL,'
        'thresholds TEXT NOT NULL CHECK(json_valid(thresholds)),PRIMARY KEY(scope,scene,model,kind))')
    install_triggers(connection)
    connection.execute('PRAGMA user_version=17')


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery'])
@pytest.mark.parametrize('threshold', [None, 0., .8])
def test_v17_history_becomes_one_record_with_last_policy_and_exact_pins(tmp_path, location, threshold):
    path = tmp_path / 'state.sqlite3'
    versions = [outputs(signature=name) for name in ('pinned', 'unpinned', 'latest')]
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            store.put('1', 'wd', versions[-1])
            saved = {kind: {**proof(versions[0][kind]), 'owned_ids': ['900']}
                     for kind in ('scene', 'markers')}
            if location == 'published':
                store.publish('1', saved)
            else:
                store.prepare('1', saved)
                if location == 'recovery':
                    from stash_ai.storage.recovery import abandon
                    abandon(store, lambda _: None)
    with sqlite3.connect(path) as connection:
        old_threshold_table(connection)
        for kind in ('scene', 'markers'):
            connection.execute('UPDATE results SET data=? WHERE kind=?',
                               (encode([version[kind] for version in versions]), kind))
        for scope in ('first', 'second'):
            for kind in ('scene', 'markers'):
                connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)',
                    (scope, '1', 'wd', kind, encode_thresholds([.65, .7, threshold])))
        before = {table: connection.execute('SELECT * FROM ' + table).fetchall()
                  for table in ('snapshots', 'recovery_publications')}
    original = path.read_bytes()
    assert db.update_database(path, dry_run=True) == (17, 18)
    assert path.read_bytes() == original
    assert db.update_database(path) == (17, 18)
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            for kind in ('scene', 'markers'):
                effective = deepcopy(versions[-1][kind])
                if threshold is not None:
                    effective[kind.rstrip('s') + '_threshold_override'] = threshold
                assert store.records('1', kind) == {'wd': effective}
                raw, = store.connection.execute('SELECT data FROM results WHERE scope=? AND kind=?',
                                                (scope, kind)).fetchone()
                assert decode(raw) == versions[-1][kind]
                policies = store.connection.execute(
                    'SELECT threshold,typeof(threshold) FROM result_thresholds WHERE scope=? AND kind=?',
                    (scope, kind)).fetchall()
                assert policies == ([] if threshold is None else [(threshold, 'real')])
                format_, data = store.connection.execute(
                    'SELECT format,data FROM result_metadata WHERE scope=? AND kind=?', (scope, kind)).fetchone()
                assert format_ == FORMAT == 4
                assert decode_metadata(data)['meta']['inference_signature'] == 'latest'
                assert completed(store, '1', 'wd', [kind], indexed_signature(effective))
                archive = archived_records(store, '1', kind)['wd']
                assert len(archive) == 1
                assert marker_segment_signature(archive[0]) == marker_segment_signature(versions[0][kind])
                assert not any(key.endswith('_override') for key in archive[0])
        assert db.check_database(path, scope)['ok']
    with sqlite3.connect(path) as connection:
        assert {table: connection.execute('SELECT * FROM ' + table).fetchall() for table in before} == before
        # The table swap restores invalidation and cascading deletion triggers.
        connection.execute("INSERT INTO merge_checks VALUES ('first','1','policy','fingerprint')")
        connection.execute("INSERT OR REPLACE INTO result_thresholds VALUES ('first','1','wd','scene',.9)")
        assert not connection.execute("SELECT 1 FROM merge_checks WHERE scope='first'").fetchone()
        connection.execute("DELETE FROM results WHERE scope='first' AND kind='scene'")
        assert not connection.execute("SELECT 1 FROM result_thresholds WHERE scope='first' AND kind='scene'").fetchone()


def test_invalid_history_policy_rolls_back_conversion_and_archive(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        store.publish('1', {'markers': proof(outputs()['markers'])})
    with sqlite3.connect(path) as connection:
        old_threshold_table(connection)
        for kind in ('scene', 'markers'):
            connection.execute('UPDATE results SET data=? WHERE kind=?', (encode([outputs()[kind]]), kind))
        connection.execute("INSERT INTO result_thresholds VALUES ('scope','1','wd','markers','[0.7,0.8]')")
        tables = ('results', 'result_thresholds', 'result_metadata', 'result_index', 'inference_archive', 'snapshots')
        before = {table: connection.execute('SELECT * FROM ' + table).fetchall() for table in tables}
    with pytest.raises(ValueError, match='do not match segment history'):
        db.update_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone() == (17,)
        assert {table: connection.execute('SELECT * FROM ' + table).fetchall() for table in tables} == before
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name='result_thresholds_update'").fetchone()


def test_list_writer_is_rejected_without_losing_current_result_or_publication(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        saved = outputs()
        store.put('1', 'wd', saved)
        store.publish('1', {'markers': proof(saved['markers'])})
        before = store.connection.execute('SELECT * FROM results').fetchall()
        with pytest.raises(ValueError, match='Incomplete result record'):
            store.replace_record('1', 'wd', 'markers', [saved['markers']])
        assert store.connection.execute('SELECT * FROM results').fetchall() == before
        assert not archived_records(store, '1', 'markers')
        assert store.model_record('1', 'markers', 'missing') is None
