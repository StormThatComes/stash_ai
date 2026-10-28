import json
import sqlite3
import pytest
from stash_ai.recognition.database import FaceDatabase, SCHEMA_V6, APPLICATION_ID
from stash_ai.recognition.pixel_storage import compact, read, copy_rows


def test_inline_thumbnail_becomes_verified_shared_blob(tmp_path):
    with FaceDatabase(tmp_path/'db') as state:
        db=state.connection;raw=json.dumps(dict(evidence_thumbnail=(bytes(range(256))*12).hex(),evidence_aspect=1))
        value=compact(db,raw);assert len(value)<150
        assert compact(db,raw)==value
        assert db.execute('SELECT COUNT(*) FROM evidence_thumbnails').fetchone()[0]==1
        assert read(db,json.loads(value))==read(db,json.loads(raw))
        db.execute('UPDATE evidence_thumbnails SET pixels=?',(b'0'*3072,))
        with pytest.raises(ValueError,match='changed'):read(db,json.loads(value))


def test_v6_writer_migration_keeps_old_inline_evidence(tmp_path):
    path=tmp_path/'old'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V6);db.execute(f'PRAGMA application_id={APPLICATION_ID}');db.execute('PRAGMA user_version=6')
        db.execute("INSERT INTO metadata VALUES ('installation','00000000-0000-0000-0000-000000000001')")
    with FaceDatabase(path) as state:
        from stash_ai.recognition.database import SCHEMA_VERSION
        assert state.connection.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION
        assert read(state.connection,dict(evidence_thumbnail=(b'0'*3072).hex()))==b'0'*3072


def test_hybrid_signature_cache_invalidates_direct_person_update(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.hybrid_merge import signature
    config,_=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        first=signature(db,selected,config)
        sql=[];db.set_trace_callback(sql.append)
        assert signature(db,selected,config)==first
        assert not any('SELECT o.rowid,o.id' in q or 'SELECT e.observation,e.contract' in q for q in sql)
        db.execute("UPDATE person_evidence SET metadata='{\"changed\":true}'")
        assert signature(db,selected,config)!=first


def test_retained_history_copies_thumbnail_blobs_and_missing_blob_error_is_preserved(tmp_path):
    from tests.test_recognition_clustering import setup, scan
    from stash_ai.recognition.database import SCHEMA
    from stash_ai.recognition.evaluation_scenarios import copy_history
    from stash_ai.recognition.evidence_pixels import canonicalize
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state, sqlite3.connect(':memory:') as target:
        sid, _ = scan(state, config, '1')
        source = state.connection
        quality = compact(source, json.dumps(dict(evidence_thumbnail=(bytes(range(256))*12).hex(),
            evidence_pixel_sha256='a'*64, evidence_aspect=1)))
        source.execute('UPDATE observations SET quality=? WHERE scan=?', (quality, sid))
        source.commit()
        target.row_factory = sqlite3.Row
        target.executescript(SCHEMA)
        record = source.execute('SELECT * FROM scans WHERE id=?', (sid,)).fetchone()
        copy_history(source, target, [], record['scope'], record['namespace'])
        assert read(target, json.loads(quality)) == read(source, json.loads(quality))
        target.execute('DELETE FROM evidence_thumbnails')
        with pytest.raises(ValueError, match='Missing or changed'):
            canonicalize(target)
        assert not target.execute("SELECT 1 FROM sqlite_temp_master WHERE name='pixel_witnesses'").fetchone()


def test_alias_map_reused_and_invalidated_by_thumbnail_write(tmp_path,monkeypatch):
    from stash_ai.recognition import evidence_pixels
    calls=[]
    def compute(db,scope=None,namespace=None,**kwargs):
        calls.append((scope,namespace))
        yield 'a','b'
    monkeypatch.setattr(evidence_pixels,'_aliases',compute)
    with FaceDatabase(tmp_path/'db') as state:
        db=state.connection
        assert dict(evidence_pixels.aliases(db,'scope','ns',persist=True))=={'a':'b'}
        db.commit()
        assert dict(evidence_pixels.aliases(db,'scope','ns',persist=True))=={'a':'b'}
        assert len(calls)==1
        db.execute("INSERT INTO evidence_versions VALUES ('raw-observations','scan',1)")
        db.execute("INSERT INTO scans VALUES ('scan','scope','1','ns','i','p','{}','{}',0,0,'complete')")
        assert dict(evidence_pixels.aliases(db,'scope','ns',persist=True))=={'a':'b'}
        assert len(calls)==2
        db.rollback()
        assert dict(evidence_pixels.aliases(db,'scope','ns',persist=True))=={'a':'b'}
        assert len(calls)==2
        dict(evidence_pixels.aliases(db,'scope','other'))
        assert len(calls)==3


def test_pixel_verification_edges_survive_map_invalidation(tmp_path,monkeypatch):
    import numpy as np
    from stash_ai.recognition import evidence_pixels
    from tests.test_recognition_clustering import setup,scan
    config=setup(tmp_path);calls=[];original=evidence_pixels._verified
    def counted(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(evidence_pixels,'_verified',counted)
    with FaceDatabase(config.database) as state:
        sid,_=scan(state,config,'1');db=state.connection
        pixels=np.random.default_rng(4).integers(10,245,3072,dtype='uint8')
        rows=list(db.execute('SELECT id,quality FROM observations WHERE scan=? ORDER BY id',(sid,)))
        for i,row in enumerate(rows):
            quality=json.loads(row['quality']);quality.update(evidence_thumbnail=(pixels+i).tobytes().hex(),
                evidence_pixel_sha256=f'{i:064x}',evidence_aspect=1.)
            db.execute('UPDATE observations SET quality=? WHERE id=?',(json.dumps(quality),row['id']))
        first=dict(evidence_pixels.aliases(db,persist=True));count=len(calls)
        assert count>0 and first
        db.execute("UPDATE observations SET quality=json_set(quality,'$.revision',1)")
        assert dict(evidence_pixels.aliases(db,persist=True))==first
        assert len(calls)==count
