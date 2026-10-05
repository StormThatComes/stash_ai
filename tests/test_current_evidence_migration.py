"""Retiring active inference preserves publication pins and permits fresh inference."""
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from copy import deepcopy
import sqlite3
import zlib

import pytest

from stash_ai import db
from stash_ai.evidence import formats
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.storage.archive import records as archived_records
from stash_ai.storage.codec import encode
from stash_ai.storage.store import StateStore, scope_for
from tests.helpers.evidence import inject_records, paired_records
from tests.helpers.runtime import runtime as app
from tests.test_inference_archive import proof
from tests.test_result_thresholds import outputs


def obsolete():
    saved = outputs(signature='old')
    for value in saved.values():
        value['meta']['analysis_format'] = 3
        value['meta']['marker_tag_blocklist'] = []
    saved['scene'].pop('frames')
    for field in ('scene_frames_format', 'scene_score_floor', 'scene_support_floor'):
        saved['scene']['meta'].pop(field, None)
    saved['scene']['items'][0]['statistics'].pop('peak_time')
    return saved


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery', 'nested'])
def test_upgrade_retires_old_results_across_scopes_and_preserves_exact_pins(tmp_path, location):
    path = tmp_path / 'state.sqlite3'
    old = obsolete()
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            inject_records(store, '1', 'wd', {'scene': old['scene'], 'markers': old['markers']})
            store.put('1', 'dtq', outputs(signature='current'))
            saved = {**proof(old['markers'], nested=location == 'nested'), 'owned_ids': ['900']}
            if location in ('published', 'nested'):
                store.publish('1', {'markers': saved})
            else:
                store.prepare('1', {'markers': saved})
                if location == 'recovery':
                    from stash_ai.storage.recovery import abandon
                    abandon(store, lambda _: None)
            with store.connection:
                for model in ('wd', 'dtq'):
                    store.connection.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)',
                        (scope, '1', model, 'partial', 1, 2, 1, zlib.compress(b'\x00\x00\x00\x00')))
                store.connection.execute('INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)',
                    (scope, '1', 'wd', 1, encode([])))
                store.connection.execute('INSERT INTO checkpoint_analysis VALUES (?,?,?,?)',
                    (scope, '1', 'wd', encode({'obsolete': True})))
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=18')
        snapshots = connection.execute('SELECT * FROM snapshots').fetchall()
        recovery = connection.execute('SELECT * FROM recovery_publications').fetchall()
    before = path.read_bytes()
    assert db.update_database(path, dry_run=True) == (18, 21)
    assert path.read_bytes() == before
    assert db.update_database(path) == (18, 21)
    for scope in ('first', 'second'):
        with StateStore(path, scope) as store:
            assert 'wd' not in store.records('1', 'scene')
            assert store.model_record('1', 'markers', 'wd') is None
            assert store.model_record('1', 'scene', 'dtq') == outputs(signature='current')['scene']
            assert archived_records(store, '1', 'scene') == {}
            archived = archived_records(store, '1', 'markers')['wd']
            assert len(archived) == 1
            assert marker_segment_signature(archived[0]) == marker_segment_signature(old['markers'])
            assert set(store.connection.execute('SELECT model,kind FROM refresh_requests WHERE scope=?', (scope,))) == {
                ('wd', 'scene'), ('wd', 'markers')}
            assert list(store.connection.execute('SELECT model FROM checkpoints WHERE scope=?', (scope,))) == [('dtq',)]
            for table in ('checkpoint_frames', 'checkpoint_analysis'):
                assert not store.connection.execute('SELECT 1 FROM ' + table + ' WHERE scope=?', (scope,)).fetchone()
        assert db.check_database(path, scope)['ok']
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT * FROM snapshots').fetchall() == snapshots
        assert connection.execute('SELECT * FROM recovery_publications').fetchall() == recovery
    assert db.update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)


def test_retirement_and_archive_roll_back_together(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    old = obsolete()
    with StateStore(path, 'scope') as store:
        inject_records(store, '1', 'wd', old)
        store.publish('1', {'markers': proof(old['markers'])})
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=18')
        before = connection.execute('SELECT * FROM results').fetchall()
    from stash_ai.db import current_evidence_migration
    retain = current_evidence_migration.retain

    def fail(*args):
        retain(*args)
        raise RuntimeError('interrupted migration')

    monkeypatch.setattr(current_evidence_migration, 'retain', fail)
    with pytest.raises(RuntimeError, match='interrupted migration'):
        db.update_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone() == (18,)
        assert connection.execute('SELECT * FROM results').fetchall() == before
        assert not connection.execute('SELECT 1 FROM inference_archive').fetchone()
        assert not connection.execute('SELECT 1 FROM refresh_requests').fetchone()


@pytest.mark.parametrize('version', [None, 1, 2, 3, 4])
def test_writer_rejects_obsolete_inference_without_changing_current_results(tmp_path, version):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        current = outputs()
        store.put('1', 'wd', current)
        old = deepcopy(current)
        for value in old.values():
            value['meta']['analysis_format'] = version
        with pytest.raises(ValueError, match='Obsolete active inference'):
            store.put('1', 'wd', old)
        assert store.model_record('1', 'scene', 'wd') == current['scene']
        assert store.model_record('1', 'markers', 'wd') == current['markers']


@pytest.mark.parametrize('field', sorted(formats.STATISTIC_FIELDS))
def test_nonnull_statistics_require_every_field(field):
    scenes, _ = paired_records({'wd': [.9, .9]})
    saved = scenes['wd']
    saved['items'][0]['statistics'].pop(field)
    with pytest.raises(ValueError, match='statistics must be complete'):
        formats.validate_current(saved)


@pytest.mark.parametrize('changes', [
    {'hits': True}, {'high_hits': 3}, {'longest_hits': 3}, {'sampled_frames': 1},
    {'max_score': float('nan')}, {'coverage_fraction': 1.1}, {'peak_time': -1},
])
def test_statistics_reject_impossible_counts_and_scores(changes):
    scenes, _ = paired_records({'wd': [.9, .9]})
    saved = scenes['wd']
    saved['items'][0]['statistics'].update(changes)
    with pytest.raises(ValueError, match='internally consistent'):
        formats.validate_current(saved)


def test_upgrade_does_not_discard_unknown_future_evidence(tmp_path):
    path = tmp_path / 'state.sqlite3'
    saved = outputs()
    saved['scene']['meta']['analysis_format'] = 99
    with StateStore(path, 'scope') as store:
        inject_records(store, '1', 'wd', saved)
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=18')
        before = connection.execute('SELECT * FROM results').fetchall()
    with pytest.raises(ValueError, match='Unsupported archived inference'):
        db.update_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT * FROM results').fetchall() == before
        assert connection.execute('PRAGMA user_version').fetchone() == (18,)


def test_upgrade_then_reinference_keeps_publication_ownership(pipeline, monkeypatch):
    p = pipeline
    from stash_ai.inference import results
    capture = results.capture_outputs

    def capture_old(*args, **kwargs):
        saved = capture(*args, **kwargs)
        for value in saved.values():
            value['meta']['analysis_format'] = 3
            value['meta']['marker_tag_blocklist'] = []
        return saved

    with monkeypatch.context() as old:
        old.setattr(results, 'capture_outputs', capture_old)
        old.setattr(formats, 'validate_current', lambda _: None)
        from stash_ai.storage import metadata
        old.setattr(metadata, 'summarize', lambda record: dict(mode=record['mode'],
            meta={k: v for k, v in record['meta'].items() if k in metadata.META_KEYS},
            has_frames='frames' in record, populated=bool(record['items'])))
        assert app.main(p.args) == 0
    published = deepcopy(p.stash.scene)
    with sqlite3.connect(p.path) as connection:
        connection.execute('PRAGMA user_version=18')
    db.update_database(p.path)
    assert p.stash.scene == published
    assert app.main(p.args) == 0
    assert len(p.loaded) == 2
    assert {marker['id'] for marker in p.stash.scene['scene_markers']} == {
        marker['id'] for marker in published['scene_markers']}
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert all(formats.current(value['meta']) for kind in ('scene', 'markers')
                   for value in store.records('1', kind).values())
        assert not store.connection.execute('SELECT 1 FROM refresh_requests').fetchone()
    assert db.check_database(p.path, scope_for(app.STASH_URL))['ok']


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery'])
def test_v18_retires_v4_with_scalar_threshold_and_keeps_exact_publication_pins(tmp_path, location):
    path = tmp_path/'state.sqlite3'
    old = outputs(signature='v4')
    for record in old.values():
        record['meta']['analysis_format'] = 4
    old['scene']['meta']['scene_frames_format'] = 1
    with StateStore(path, 'scope') as store:
        inject_records(store, '1', 'wd', old)
        saved = {**proof(old['markers']), 'owned_ids': ['native-marker']}
        saved['contributions'][0].update(meta=old['markers']['meta'], tag='core', start=0., end=1., extra_tags=['old-extra'])
        if location == 'published':
            store.publish('1', {'markers': saved})
        else:
            store.prepare('1', {'markers': saved})
            if location == 'recovery':
                from stash_ai.storage.recovery import abandon
                abandon(store, lambda _: None)
        store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)',
                                 ('scope', '1', 'wd', 'markers', .6))
        store.connection.execute('PRAGMA user_version=18')
        store.connection.commit()
    assert db.update_database(path) == (18, LATEST_SCHEMA_VERSION)
    with StateStore(path, 'scope') as store:
        assert store.records('1', 'markers') == store.records('1', 'scene') == {}
        archived = archived_records(store, '1', 'markers')['wd']
        assert archived == [old['markers']]
        assert not store.connection.execute('SELECT * FROM result_thresholds').fetchall()
        assert set(store.connection.execute('SELECT model,kind FROM refresh_requests')) == {
            ('wd', 'scene'), ('wd', 'markers')}
        if location == 'recovery':
            from stash_ai.storage.recovery import publications
            result = publications(store, '1', 'markers')[0]
        else:
            result = store.snapshot('1', 'markers')[location == 'pending']
        assert result['owned_ids'] == ['native-marker']
        assert result['contributions'][0]['requires_rebase']
    assert db.check_database(path, 'scope')['ok']
