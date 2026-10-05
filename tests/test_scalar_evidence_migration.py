"""Schema 19 proof contexts become scalar without losing durable archive pins."""
from copy import deepcopy
import sqlite3

import pytest

from stash_ai import db
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.storage.archive import pinned, records as archived_records
from stash_ai.storage.codec import encode
from stash_ai.storage.snapshots import FORMAT_KEY, require_rebased
from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import outputs


def old_snapshot(signatures):
    return encode({FORMAT_KEY: 4, 'proofs': [], 'snapshot': {
        'owned_ids': ['native-marker'], 'contributions': [{
            'source': 'wd', 'support_context': {'wd': signatures},
            'source_evidence': {'wd': {'signatures': signatures}},
            'scene_evidence': [{'derived': 'shared_scene_frames',
                'scene_context': {'wd': [signatures[0]]},
                'frame_context': {'wd': [signatures[0]]}}],
        }]}})


def old_index(connection):
    connection.execute('ALTER TABLE result_index RENAME TO result_index_new')
    connection.execute('CREATE TABLE result_index (scope TEXT, scene TEXT, model TEXT, kind TEXT, '
                       'inference INTEGER, populated INTEGER, signature TEXT, '
                       'PRIMARY KEY(scope,scene,model,kind))')
    connection.execute('INSERT INTO result_index SELECT scope,scene,model,kind,1,populated,signature '
                       'FROM result_index_new')
    connection.execute('DROP TABLE result_index_new')
    connection.execute('PRAGMA user_version=19')


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery'])
@pytest.mark.parametrize('multiple', [False, True])
def test_v19_conversion_keeps_active_inference_thresholds_and_exact_pins(tmp_path, location, multiple):
    path = tmp_path/'state.sqlite3'
    old = outputs(signature='old')
    signature = marker_segment_signature(old['markers'])
    signatures = [signature, 'second-pin'] if multiple else [signature]
    blob = old_snapshot(signatures)
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            current = deepcopy(old)
            current['markers']['publication_threshold'] = .7
            store.put('1', 'wd', current)
            if location == 'recovery':
                store.connection.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)',
                                         (scope, '1', 'markers', 'abandoned', blob))
            else:
                store.connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)',
                    (scope, '1', 'markers', blob if location == 'published' else None,
                     blob if location == 'pending' else None))
            store.connection.commit()
    with sqlite3.connect(path) as connection:
        old_index(connection)
        before = connection.execute('SELECT * FROM results').fetchall()
        completion = connection.execute('SELECT scope,scene,model,kind,populated,signature FROM result_index').fetchall()
    raw = path.read_bytes()
    assert db.update_database(path, dry_run=True) == (19, 20)
    assert path.read_bytes() == raw
    assert db.update_database(path) == (19, 20)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT * FROM results').fetchall() == before
        assert connection.execute('SELECT * FROM result_index').fetchall() == completion
        assert 'inference' not in {row[1] for row in connection.execute('PRAGMA table_info(result_index)')}
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            assert pinned(store, '1') == {('wd', value) for value in signatures}
            if location == 'recovery':
                saved = store.unpack_snapshot(store.connection.execute(
                    'SELECT data FROM recovery_publications WHERE scope=?', (scope,)).fetchone()[0])
            else:
                column = 'pending' if location == 'pending' else 'published'
                saved = store.unpack_snapshot(store.connection.execute(
                    f'SELECT {column} FROM snapshots WHERE scope=?', (scope,)).fetchone()[0])
            assert saved['owned_ids'] == ['native-marker']
            item = saved['contributions'][0]
            assert item['scene_evidence'][0]['scene_context'] == {'wd': signature}
            assert 'frame_context' not in item['scene_evidence'][0]
            if multiple:
                assert item['retired_support_context'] == {'wd': signatures}
                with pytest.raises(RuntimeError, match='Retired publication proof'):
                    require_rebased(saved['contributions'])
            else:
                assert item['support_context'] == {'wd': signature}
                assert item['source_evidence']['wd']['signature'] == signature
                require_rebased(saved['contributions'])
            assert store.model_record('1', 'markers', 'wd')['publication_threshold'] == .7
            store.put('1', 'wd', outputs(signature='new'))
            assert marker_segment_signature(archived_records(store, '1', 'markers')['wd'][0]) == signature
        assert db.check_database(path, scope)['ok']


def test_invalid_context_rolls_back_snapshot_and_index_migration(tmp_path):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        blob = old_snapshot([None])
        store.connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)', ('scope', '1', 'markers', blob, None))
        store.connection.commit()
    with sqlite3.connect(path) as connection:
        old_index(connection)
        before = connection.execute('SELECT * FROM snapshots').fetchall()
    with pytest.raises(ValueError, match='historical proof signature'):
        db.update_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone() == (19,)
        assert connection.execute('SELECT * FROM snapshots').fetchall() == before
        assert 'inference' in {row[1] for row in connection.execute('PRAGMA table_info(result_index)')}
