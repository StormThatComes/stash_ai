"""Durable cache invalidation follows evidence, including external SQL writes."""
import json
import sqlite3
import uuid
import pytest

from stash_ai.recognition.database import FaceDatabase, SCHEMA_V4, APPLICATION_ID, SCHEMA_VERSION
from stash_ai.recognition.dependencies import version
from stash_ai.recognition.rebuild_cache import input_context
from tests.test_recognition_rebuild_cache import assignment_fixture, setup, changed


def test_v4_migration_preserves_installation_and_installs_indexed_addresses(tmp_path):
    config=setup(tmp_path);installation=str(uuid.uuid4())
    with sqlite3.connect(config.database) as db:
        db.executescript(SCHEMA_V4)
        db.execute('INSERT INTO metadata VALUES (?,?)',('installation',installation))
        db.execute(f'PRAGMA application_id={APPLICATION_ID}');db.execute('PRAGMA user_version=4')
    with FaceDatabase(config.database) as state:
        assert state.installation==installation
        assert state.connection.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION
        from .test_recognition_clustering import scan
        import numpy as np
        sid,_=scan(state,config,'1',[np.array([1.,0.,0.,0.])]*3)
        from stash_ai.recognition.reference_roles import observation_sql
        plan=state.connection.execute('EXPLAIN QUERY PLAN SELECT o.id,(SELECT c.rowid FROM candidates c WHERE '+
            observation_sql('c')+'=o.id ORDER BY c.rowid DESC LIMIT 1) FROM observations o WHERE scan=?',(sid,))
        details=[r[3] for r in plan]
        assert any('SEARCH c USING' in p and 'candidates_reference_observation' in p for p in details),details
        assert not any('SCAN c' in p for p in details)


def test_changed_scene_rehashes_only_its_input_and_rollback_restores_cache(tmp_path):
    config=changed(setup(tmp_path),anchor_view_matching=True)
    with FaceDatabase(config.database) as state:
        selected=assignment_fixture(state)
        db=state.connection
        original=input_context(db,selected,'scope','ns',config)
        before=version(db)
        statements=[];db.set_trace_callback(statements.append)
        assert input_context(db,selected,'scope','ns',config)==original
        assert not any('o.dimensions,o.embedding' in s for s in statements)
        db.execute('UPDATE observations SET seconds=seconds+1 WHERE scan=?',(selected[0]['scan'],))
        statements.clear()
        assert input_context(db,selected,'scope','ns',config)!=original
        scans=[s for s in statements if 'o.dimensions,o.embedding' in s]
        assert len(scans)==1 and selected[0]['scan'] in scans[0]
        db.rollback()
        assert version(db)==before and input_context(db,selected,'scope','ns',config)==original


def test_external_replace_invalidates_both_actual_observation_addresses(tmp_path):
    config=changed(setup(tmp_path),anchor_view_matching=True)
    with FaceDatabase(config.database) as state:
        selected=assignment_fixture(state)
        db=state.connection
        before=input_context(db,selected,'scope','ns',config)
        source=db.execute("SELECT * FROM candidates WHERE json_valid(payload) AND json_extract(payload,'$.observation') "
            'IN (SELECT id FROM observations WHERE scan=?) ORDER BY rowid DESC LIMIT 1',(selected[0]['scan'],)).fetchone()
        target=db.execute('SELECT id FROM observations WHERE scan=? LIMIT 1',(selected[1]['scan'],)).fetchone()[0]
        old=version(db,'scan',selected[0]['scan']);new=version(db,'scan',selected[1]['scan'])
        values=list(source);values[-1]=json.dumps(dict(observation=target,reason='replacement'))
        # Default recursive_triggers=OFF is intentional: REPLACE must still
        # invalidate the address of the row it implicitly deleted.
        with sqlite3.connect(config.database) as writer:
            assert not writer.execute('PRAGMA recursive_triggers').fetchone()[0]
            writer.execute('INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?)',values)
        assert version(db,'scan',selected[0]['scan'])>old
        assert version(db,'scan',selected[1]['scan'])>new
        assert input_context(db,selected,'scope','ns',config)!=before


def test_moved_observation_invalidates_old_scan_but_crop_bytes_do_not(tmp_path):
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected=assignment_fixture(state)
        db=state.connection
        identifier=db.execute('SELECT id FROM observations WHERE scan=? LIMIT 1',(selected[0]['scan'],)).fetchone()[0]
        before=version(db);old=version(db,'scan',selected[0]['scan'])
        db.execute('UPDATE observations SET crop=? WHERE id=?',(b'jpeg',identifier))
        assert version(db)==before
        row=dict(db.execute('SELECT * FROM observations WHERE id=?',(identifier,)).fetchone())
        row['scan']=selected[1]['scan']
        db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',tuple(row.values()))
        assert version(db,'scan',selected[0]['scan'])>old


@pytest.mark.parametrize('change',['drop','replace'])
def test_missing_or_spoofed_dependency_trigger_fails_closed(tmp_path,change):
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        state.connection.execute('DROP TRIGGER recognition_version_observations_update')
        if change=='replace':
            state.connection.execute('CREATE TRIGGER recognition_version_observations_update AFTER UPDATE ON observations BEGIN SELECT 1; END')
        state.connection.commit()
    with pytest.raises(ValueError,match='trigger'):
        FaceDatabase(config.database)


def test_readonly_evaluation_does_not_write_digest_caches(tmp_path):
    config=changed(setup(tmp_path),anchor_view_matching=True)
    with FaceDatabase(config.database) as state:
        selected=assignment_fixture(state)
    before=config.database.read_bytes()
    with sqlite3.connect(config.database.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        assert input_context(db,selected,'scope','ns',config)
    assert config.database.read_bytes()==before
