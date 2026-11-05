"""Real SQLite retention: reuse, identity safety, rollback and isolation."""
from contextlib import closing
from dataclasses import replace
import json
import sqlite3

import pytest

from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition import db_retention as retention


def settings(tmp_path):
    c = load(root=tmp_path)
    return replace(c, data=plain(c.data))


def successful(state, c, scan, scene='1', scope='scope'):
    run = state.start_run(scope, [scene], c.snapshot())
    state.progress(run, 'rebuild', {'selection': [{'scan': scan, 'scene': scene}]})
    state.finish_run(run, 'success', {})
    return run


def scans(state, c, count=4, *, scope='scope', scene='1', prefix='s'):
    db = state.connection
    result = []
    for i in range(count):
        identifier = prefix + str(i)
        with db:
            db.execute('INSERT INTO scans(id,scope,scene,namespace,input_sha,policy_sha,manifest,coverage,total,cursor,status) '
                       'VALUES (?,?,?,?,?,?,?,?,1,1,\'complete\')',
                       (identifier, scope, scene, namespace(c), identifier,
                        scan_identity(c, namespace(c)), '{}', '{}'))
            db.execute('INSERT INTO observations VALUES (?,?,0,0.,\'vtt\',\'[]\',\'[]\',\'{}\',NULL,1,?,?)',
                       ('face-' + identifier, identifier, b'vector', b'pixels' * 1000))
        successful(state, c, identifier, scene, scope)
        result.append(identifier)
    return result


def logical(path):
    with closing(sqlite3.connect(path)) as db:
        return {r[0]: db.execute('SELECT * FROM ' + r[0] + ' ORDER BY rowid').fetchall()
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def assert_only_retention_index_changed(before,after):
    from stash_ai.recognition.retention_index import TABLES,MARKER
    for name,rows in before.items():
        if name not in TABLES:
            observed=after[name]
            if name=='metadata':observed=[r for r in observed if r[0]!=MARKER]
            assert observed==rows


def test_preview_is_readonly_and_apply_keeps_active_and_one_previous(tmp_path):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
    before = c.database.read_bytes()
    preview = retention.prune_database(c)
    assert c.database.read_bytes() == before
    assert preview['scans'] == {'delete': 2, 'keep': 2}
    assert preview['actions'] == [dict(table='observations', action='delete', rows=2,
                                     logical_bytes=preview['reclaimable_payload_bytes'])]
    assert preview['reclaimable_payload_bytes'] > 12000
    applied = retention.prune_database(c, apply=True)
    assert applied['actions'] == preview['actions']
    with FaceDatabase(c.database) as state:
        assert [r[0] for r in state.connection.execute('SELECT id FROM scans ORDER BY rowid')] == ['s2', 's3']
        assert state.connection.execute('PRAGMA foreign_key_check').fetchall() == []
    assert retention.prune_database(c, apply=True)['actions'] == []


def test_identity_origins_keep_vectors_geometry_and_labels_but_release_old_pixels(tmp_path):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
        with state.connection as db:
            db.execute("INSERT INTO people VALUES ('person','scope',?,1,'77','token')", (namespace(c),))
            db.execute("INSERT INTO identity_origins VALUES ('physical','person','face-s0')")
            db.execute("INSERT INTO members VALUES ('face-s3','person')")
    before = logical(c.database)
    result = retention.prune_database(c, apply=True)
    after = logical(c.database)
    assert result['scans'] == {'archive': 1, 'delete': 1, 'keep': 2}
    for table in ('people', 'members', 'identity_origins', 'runs', 'progress', 'publications'):
        assert after[table] == before[table]
    original = next(r for r in before['observations'] if r[0] == 'face-s0')
    archived = next(r for r in after['observations'] if r[0] == 'face-s0')
    assert original[:-1] == archived[:-1] and archived[-1] is None
    assert dict(after['metadata'])[retention.FACE_CROPS + 's0'] == 'true'


@pytest.mark.parametrize('block', ['partial', 'failed', 'config', 'publication', 'queue'])
def test_unfinished_or_unpublished_replacement_never_prunes(tmp_path, block):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
        with state.connection as db:
            if block == 'partial':
                db.execute("UPDATE scans SET status='partial',cursor=0 WHERE id='s3'")
            elif block == 'failed':
                db.execute("UPDATE runs SET status='failed'")
            elif block == 'config':
                c.data['detection']['nms_threshold'] = .25
            elif block == 'publication':
                db.execute("INSERT INTO publications VALUES ('p','scope','1','person','link',NULL,'{}','{}','unknown',NULL)")
            else:
                from stash_ai.recognition.publication_queue import key
                db.execute('INSERT INTO metadata VALUES (?,?)', (key('scope', namespace(c), '1'), '{}'))
    before = logical(c.database)
    report = retention.prune_database(c, apply=True)
    assert report['actions'] == [] and report['scans'] == {'keep': 4}
    assert_only_retention_index_changed(before,logical(c.database))


def test_scope_scene_and_model_isolation(tmp_path):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
        scans(state, c, scope='other', prefix='other')
        scans(state, c, scene='2', prefix='scene2')
        other = replace(c, data=plain(c.data))
        other.data['models']['recognizer'] = 'auraface'
        scans(state, other, prefix='model2')
    retention.prune_database(c, apply=True, scope='scope', scenes=['1'])
    with FaceDatabase(c.database) as state:
        remaining = {r[0] for r in state.connection.execute('SELECT id FROM scans')}
    assert remaining == {'s2', 's3'} | {prefix + str(i) for prefix in ('other', 'scene2', 'model2') for i in range(4)}


def test_reactivated_old_head_keeps_creation_marker(tmp_path):
    from stash_ai.recognition.scan_heads import activate, predicate
    c = settings(tmp_path)
    c.data['storage']['retention']['keep_previous_scans'] = 0
    with FaceDatabase(c.database) as state:
        scans(state, c)
        with state.connection as db:
            activate(db, db.execute("SELECT * FROM scans WHERE id='s0'").fetchone())
    result = retention.prune_database(c, apply=True)
    assert result['protected_reasons'] == {'active_scan': 1, 'head_creation_marker': 1}
    with FaceDatabase(c.database) as state:
        assert state.connection.execute('SELECT id FROM scans s WHERE ' + predicate()).fetchone()[0] == 's0'


def test_nested_json_references_and_cross_scan_body_face_are_preserved(tmp_path):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c, count=5)
        with state.connection as db:
            for person in ('a', 'b'):
                db.execute('INSERT INTO people VALUES (?,\'scope\',?,1,NULL,?)', (person, namespace(c), person))
            db.execute("INSERT INTO identity_merges VALUES ('a','b',1,?,'local')", (encoded({'proof': {'witnesses': ['face-s0']}}),))
            db.execute("INSERT INTO candidates VALUES ('decision','s4','policy','a','accepted',?)",
                       (encoded({'historical': ['face-s1']}),))
            db.execute("INSERT INTO body_observations VALUES ('body','s4',0,0,'contract','face-s2','{}','{}',NULL,NULL,NULL)")
    result = retention.prune_database(c, apply=True)
    assert result['scans'] == {'archive': 3, 'keep': 2}
    with FaceDatabase(c.database) as state:
        assert state.connection.execute('PRAGMA foreign_key_check').fetchall() == []


def bodies(db, scan, contract, suffix, *, face=None):
    identifier = 'body-' + suffix
    db.execute('INSERT INTO body_observations VALUES (?,?,0,0,?,?,\'{}\',\'{}\',NULL,?,?)',
               (identifier, scan, contract, face, bytes(16384), b'body-pixels' * 100))
    db.execute('INSERT INTO body_cues VALUES (?,?,0)', (scan, contract))
    return identifier


def test_obsolete_body_contracts_keep_current_previous_and_pinned_numerical_evidence(tmp_path):
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.person_signal import contract as person_contract
    c = settings(tmp_path)
    c.data['sapiensid']['enabled'] = c.data['sapiensid']['collection']['enabled'] = True
    with FaceDatabase(c.database) as state:
        scans(state, c, count=1)
        with state.connection as db:
            db.execute("INSERT INTO people VALUES ('p','scope',?,1,'77','token')", (namespace(c),))
            db.execute("INSERT INTO members VALUES ('face-s0','p')")
            for i, value in enumerate(('old-pinned', 'old-unpinned', 'previous', contract(c))):
                body = bodies(db, 's0', value, str(i), face='face-s0')
                if i == 0:
                    db.execute("INSERT INTO body_origins VALUES (?,'p')", (body,))
            # A second unpinned row in a pinned variant must survive with its
            # complete body_cues certificate; it is still reusable evidence.
            db.execute("INSERT INTO body_observations SELECT 'extra',scan,cue,seconds,contract,face,geometry,quality,rejected,embedding,crop "
                       "FROM body_observations WHERE id='body-0'")
            for value in ('old-person', 'previous-person', person_contract(c)):
                db.execute("INSERT INTO person_evidence VALUES ('face-s0',?,'ready','{}',?,?)", (value, bytes(16384), b'person-pixels'))
    result = retention.prune_database(c, apply=True)
    assert result['obsolete_person_contracts'] == 3
    with FaceDatabase(c.database) as state:
        db = state.connection
        rows = {r['id']: r for r in db.execute('SELECT * FROM body_observations')}
        assert set(rows) == {'body-0', 'extra', 'body-2', 'body-3'}
        assert rows['body-0']['crop'] is rows['extra']['crop'] is None
        assert rows['body-0']['embedding'] == bytes(16384)
        assert rows['body-2']['crop'] and rows['body-3']['crop']
        assert {r[0] for r in db.execute('SELECT contract FROM body_cues')} == {'old-pinned', 'previous', contract(c)}
        assert db.execute("SELECT crop FROM person_evidence WHERE contract='old-person'").fetchone()[0] is None
        assert db.execute("SELECT COUNT(*) FROM person_evidence").fetchone()[0] == 3


def test_incomplete_current_person_contract_does_not_prune_old_variants(tmp_path):
    from stash_ai.recognition.body_observations import contract
    c = settings(tmp_path)
    c.data['sapiensid']['enabled'] = c.data['sapiensid']['collection']['enabled'] = True
    with FaceDatabase(c.database) as state:
        scans(state, c, count=1)
        with state.connection as db:
            for i in range(3):
                bodies(db, 's0', str(i), str(i))
            # New variant without a complete cue marker cannot supersede any.
            db.execute("INSERT INTO body_observations VALUES ('new','s0',0,0,?,NULL,'{}','{}',NULL,NULL,NULL)", (contract(c),))
    before = logical(c.database)
    assert retention.prune_database(c, apply=True)['actions'] == []
    assert_only_retention_index_changed(before,logical(c.database))


def test_writer_lock_and_failure_rollback(tmp_path, monkeypatch):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
        assert retention.prune_database(c)['scans']['delete'] == 2
        with pytest.raises(RuntimeError, match='recognition writer'):
            retention.prune_database(c, apply=True)
    before = logical(c.database)
    apply = retention._apply
    def fail(db):
        apply(db)
        raise RuntimeError('injected after deletes')
    monkeypatch.setattr(retention, '_apply', fail)
    with pytest.raises(RuntimeError, match='injected'):
        retention.prune_database(c, apply=True)
    assert logical(c.database) == before


def test_malformed_proof_refuses_pruning_without_changes(tmp_path):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
        with state.connection as db:
            db.execute("INSERT INTO candidates VALUES ('bad','s0','policy',NULL,'withheld','not-json')")
    before = logical(c.database)
    with pytest.raises(ValueError, match='malformed'):
        retention.prune_database(c, apply=True)
    assert logical(c.database) == before


def test_archived_face_crop_is_restored_on_reuse_without_models(tmp_path):
    from tests.test_recognition_scanning import config
    from tests.helpers.recognition import MediaClient, Backend
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    c = config(tmp_path)
    client = MediaClient()
    with FaceDatabase(c.database) as state, SpriteInput(client, client.scene, c) as source:
        scan, _ = scan_scene(state, 'scope', client.scene, c, 'namespace', source, Backend)
        before = [tuple(r) for r in state.connection.execute('SELECT * FROM observations ORDER BY id')]
        with state.connection as db:
            db.execute('UPDATE observations SET crop=NULL WHERE scan=?', (scan,))
            db.execute('INSERT INTO metadata VALUES (?,?)', (retention.FACE_CROPS + scan, 'true'))
        _, reused = scan_scene(state, 'scope', client.scene, c, 'namespace', source,
                               lambda: pytest.fail('Restoring pixels must not load models'))
        assert reused
        assert [tuple(r) for r in state.connection.execute('SELECT * FROM observations ORDER BY id')] == before
        assert not state.connection.execute('SELECT 1 FROM metadata WHERE key=?', (retention.FACE_CROPS + scan,)).fetchone()


@pytest.mark.parametrize('status', ['failed', 'partial', 'interrupted', 'running'])
def test_automatic_retention_requires_successful_finalized_history(tmp_path, status):
    c = settings(tmp_path)
    c.data['storage']['retention']['auto_prune'] = True
    with FaceDatabase(c.database) as state:
        scans(state, c)
        run = successful(state, c, 's3')
        with state.connection as db:
            db.execute('UPDATE runs SET status=? WHERE id=?', (status, run))
        assert retention.after_run(state, c, run) is None
        assert state.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 4


def test_successful_run_prunes_only_its_selection_and_records_result(tmp_path):
    c = settings(tmp_path)
    c.data['storage']['retention']['auto_prune'] = True
    with FaceDatabase(c.database) as state:
        scans(state, c)
        scans(state, c, scene='2', prefix='other')
        run = successful(state, c, 's3')
        result = retention.after_run(state, c, run)
        assert result['applied'] and result['scans']['delete'] == 2
        assert state.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 6
        stored = state.connection.execute("SELECT data FROM progress WHERE run=? AND phase='retention'", (run,)).fetchone()[0]
        assert json.loads(stored) == result


def test_offline_cli_preview_and_apply(tmp_path, capsys, monkeypatch):
    from stash_ai.recognition.cli import main
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
    monkeypatch.setenv('STASH_URL', 'not-a-url')
    before = c.database.read_bytes()
    args = ['db-prune', '--faces-db', str(c.database), '--json']
    # Omit the repository's shipped YAML so the seeded default model matches.
    monkeypatch.chdir(tmp_path)
    assert main(args) == 0
    assert not json.loads(capsys.readouterr().out)['applied']
    assert c.database.read_bytes() == before
    assert main(args + ['--apply']) == 0
    assert json.loads(capsys.readouterr().out)['applied']


@pytest.mark.parametrize('key,value', [('auto_prune', 1), ('prune_obsolete_crops', 'yes'),
                                      ('keep_previous_scans', -1), ('keep_previous_person_contracts', True)])
def test_invalid_retention_configuration_is_rejected(tmp_path, key, value):
    p = tmp_path / 'config.yml'
    p.write_text('version: 1\nstorage:\n  retention:\n    ' + key + ': ' + json.dumps(value) + '\n')
    with pytest.raises(ValueError, match='storage.retention'):
        load(p)


def test_end_to_end_retuning_prunes_after_success_and_can_reactivate_archived_scan(tmp_path, monkeypatch):
    from tests.test_recognition_run import setup, run
    from stash_ai.recognition.scan_heads import predicate
    c, client, services, backends = setup(tmp_path, monkeypatch)
    c = replace(c, data=plain(c.data))
    c.data['storage']['retention']['auto_prune'] = True
    for threshold in (.15, .20, .25, .30):
        c.data['detection']['nms_threshold'] = threshold
        assert run(c, services, '1') == 0
    writes = list(client.writes)
    assert len(backends) == 4
    with FaceDatabase(c.database) as state:
        db = state.connection
        assert db.execute("SELECT COUNT(*) FROM progress WHERE phase='retention'").fetchone()[0] == 4
        assert db.execute('SELECT COUNT(*) FROM scans').fetchone()[0] < 4
        assert db.execute('SELECT COUNT(*) FROM observations WHERE crop IS NULL AND rejected IS NULL').fetchone()[0] > 0
    c.data['detection']['nms_threshold'] = .15
    assert run(c, services, '1') == 0
    assert client.writes == writes and len(backends) == 4
    with FaceDatabase(c.database) as state:
        db = state.connection
        assert db.execute('SELECT COUNT(*) FROM observations o JOIN scans s ON s.id=o.scan WHERE o.crop IS NULL '
                          'AND o.rejected IS NULL AND ' + predicate()).fetchone()[0] == 0


def test_body_crops_pruned_by_retention_are_restored_without_inference(tmp_path):
    from tests.helpers.recognition import MediaClient
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.person_crop_backfill import backfill
    c = settings(tmp_path)
    c.data['sapiensid']['enabled'] = c.data['sapiensid']['collection']['enabled'] = True
    client = MediaClient()
    with FaceDatabase(c.database) as state, SpriteInput(client, client.scene, c) as source:
        scans(state, c, count=1)
        original = contract(c)
        with state.connection as db:
            db.execute("INSERT INTO people VALUES ('p','scope',?,1,'77','token')", (namespace(c),))
            body = bodies(db, 's0', original, 'original')
            db.execute('UPDATE body_observations SET quality=? WHERE id=?',
                       (encoded({'person': {'bounds': [0, 0, 64, 64]}}), body))
            db.execute("INSERT INTO body_origins VALUES (?,'p')", (body,))
        # Two later complete contracts supersede the original crop.
        for i, threshold in enumerate((.6, .7)):
            c.data['sapiensid']['pose']['score_threshold'] = threshold
            with state.connection as db:
                bodies(db, 's0', contract(c), str(i))
        state.connection.execute('BEGIN IMMEDIATE')
        retention.on_connection(state.connection, c, apply=True)
        state.connection.commit()
        assert state.connection.execute('SELECT crop FROM body_observations WHERE id=?', (body,)).fetchone()[0] is None
        c.data['sapiensid']['pose']['score_threshold'] = settings(tmp_path).data['sapiensid']['pose']['score_threshold']
        assert contract(c) == original
        before = state.connection.execute('SELECT embedding FROM body_observations WHERE id=?', (body,)).fetchone()[0]
        backfill(state.connection, 's0', source, c)
        row = state.connection.execute('SELECT embedding,crop FROM body_observations WHERE id=?', (body,)).fetchone()
        assert row['embedding'] == before and row['crop'].startswith(b'\x89PNG')


def test_unreferenced_matching_thumbnails_are_collected_without_removing_shared_pixels(tmp_path):
    c = settings(tmp_path)
    with FaceDatabase(c.database) as state:
        scans(state, c)
        with state.connection as db:
            db.executemany('INSERT INTO evidence_thumbnails VALUES (?,?)', [('shared', bytes(3072)), ('old', bytes(3072))])
            for face, thumbnail in (('face-s0', 'old'), ('face-s1', 'shared'), ('face-s3', 'shared')):
                db.execute('UPDATE observations SET quality=? WHERE id=?', (encoded({'evidence_thumbnail_id': thumbnail}), face))
    retention.prune_database(c, apply=True)
    with FaceDatabase(c.database) as state:
        assert [r[0] for r in state.connection.execute('SELECT id FROM evidence_thumbnails')] == ['shared']
