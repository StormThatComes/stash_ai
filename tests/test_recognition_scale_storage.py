import hashlib
import json
import sqlite3

import numpy as np
import pytest

from stash_ai.recognition import database, person_signal
from stash_ai.recognition.shared_person_evidence import share


def seed(db):
    db.execute("INSERT INTO scans VALUES ('s','scope','1','ns','i','p','{}','{}',1,1,'complete')")
    db.execute("INSERT INTO observations VALUES ('f','s',0,0,'','[]','{}','{}',NULL,2,?,NULL)",
               (np.array([1,0],dtype='<f4').tobytes(),))
    vector=np.zeros(4096,dtype='<f4');vector[0]=1
    db.execute("INSERT INTO body_observations VALUES ('b','s',0,0,'bc','f','{}','{}',NULL,?,?)",(vector.tobytes(),b'image'))
    db.execute("INSERT INTO person_evidence VALUES ('f','pc','ready','{}',?,?)",(vector.tobytes(),b'image'))
    return vector.tobytes()


def test_shared_projection_survives_update_delete_and_rollback(tmp_path):
    with database.FaceDatabase(tmp_path/'db') as state:
        db=state.connection;vector=seed(db);share(db,'f','pc','b');db.commit()
        assert db.execute('SELECT length(embedding),length(crop) FROM person_evidence').fetchone()[:]==(6,6)
        assert person_signal.read(db,'f','pc')['embedding']==vector
        db.execute('UPDATE body_observations SET embedding=zeroblob(16384),crop=NULL')
        assert person_signal.read(db,'f','pc')['embedding']==vector
        assert person_signal.read(db,'f','pc')['crop']==b'image'
        db.rollback()
        assert db.execute('SELECT length(embedding) FROM person_evidence').fetchone()[0]==6
        db.execute('DELETE FROM body_observations')
        assert person_signal.read(db,'f','pc')['embedding']==vector
        assert person_signal.read(db,'f','pc')['crop']==b'image'


def test_v11_migration_compacts_exact_projection(tmp_path):
    path=tmp_path/'old'
    with sqlite3.connect(path) as db:
        db.executescript(database.SCHEMA_V11)
        db.execute(f'PRAGMA application_id={database.APPLICATION_ID}')
        db.execute('PRAGMA user_version=11')
        db.execute("INSERT INTO metadata VALUES ('installation','00000000-0000-0000-0000-000000000001')")
        vector=seed(db)
    with database.FaceDatabase(path) as state:
        db=state.connection
        assert db.execute('SELECT length(embedding) FROM person_evidence').fetchone()[0]==6
        assert person_signal.read(db,'f','pc')['embedding']==vector
        assert person_signal.read(db,'f','pc')['crop']==b'image'


def pixels(db,oid,delta):
    raw=(np.random.default_rng(7).integers(10,235,3072,dtype='uint8')+delta).tobytes()
    key=hashlib.sha256(raw).hexdigest()
    db.execute('INSERT OR IGNORE INTO evidence_thumbnails VALUES (?,?)',(key,raw))
    q=json.dumps(dict(evidence_pixel_sha256=oid*64,evidence_thumbnail_id=key,evidence_aspect=1))
    db.execute('INSERT INTO observations VALUES (?,?,0,0,\'\',\'[]\',\'{}\',?,NULL,2,?,NULL)',
               (oid,oid,q,np.array([1,0],dtype='<f4').tobytes()))


def test_incremental_pixels_recompute_dirty_bucket_and_preserve_nontransitivity(tmp_path,monkeypatch):
    from stash_ai.recognition.pixel_catalogue import prepare
    from stash_ai.recognition.evidence_pixels import aliases
    from stash_ai.recognition import pixel_storage
    with database.FaceDatabase(tmp_path/'db') as state:
        db=state.connection
        def append(oid,delta):
            db.execute('INSERT INTO scans VALUES (?,\'scope\',?,\'ns\',\'i\',\'p\',\'{}\',\'{}\',1,1,\'complete\')',(oid,str(ord(oid))))
            pixels(db,oid,delta)
        append('b',2);append('c',4)
        first=prepare(db,'scope','ns',disk_bytes=100000)
        assert dict(first.items())==dict(aliases(db,'scope','ns'))=={'c'*64:'b'*64}
        calls=[];original=pixel_storage.read
        monkeypatch.setattr(pixel_storage,'read',lambda *a:(calls.append(1),original(*a))[1])
        assert dict(prepare(db,'scope','ns',disk_bytes=100000).items())==dict(first.items())
        assert not calls
        append('a',0)
        assert dict(prepare(db,'scope','ns',disk_bytes=100000).items())==dict(aliases(db,'scope','ns'))=={'b'*64:'a'*64}
        db.commit()
        key=db.execute("SELECT json_extract(quality,'$.evidence_thumbnail_id') FROM observations WHERE id='a'").fetchone()[0]
        db.execute('UPDATE evidence_thumbnails SET pixels=zeroblob(3072) WHERE id=?',(key,))
        with pytest.raises(ValueError,match='changed evidence thumbnail'):prepare(db,'scope','ns',disk_bytes=100000)
        db.rollback()
        assert dict(prepare(db,'scope','ns',disk_bytes=100000).items())=={'b'*64:'a'*64}


def test_legacy_thumbnail_generation_reads_pixels_without_new_clock(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.evidence_pixels import _generation
    from stash_ai.recognition.dependency_schema import PIXEL_TRIGGERS
    from tests.test_recognition_clustering import setup,scan
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        scan(state,config,'1')
        import hashlib
        pixels=bytes(range(256))*12;key=hashlib.sha256(pixels).hexdigest()
        state.connection.execute('INSERT OR IGNORE INTO evidence_thumbnails VALUES (?,?)',(key,pixels))
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_thumbnail_id',?)",(key,))
        for name in PIXEL_TRIGGERS:state.connection.execute('DROP TRIGGER '+name)
        before=_generation(state.connection,'scope','ns')
        state.connection.execute('UPDATE evidence_thumbnails SET pixels=zeroblob(3072)')
        assert _generation(state.connection,'scope','ns')!=before
