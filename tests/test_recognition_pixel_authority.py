"""Global representative choice and warm evidence integrity regressions."""
import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from stash_ai.recognition.evidence_pixels import aliases, alias_scope, canonicalize


def pixel_db(items, dimensions=2):
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE observations(id TEXT PRIMARY KEY,quality TEXT,embedding BLOB,rejected TEXT)')
    raw=np.random.default_rng(7).integers(10,235,3072,dtype='uint8')
    vector=np.zeros(dimensions,dtype='<f4');vector[0]=1
    for name,delta in items:
        q=dict(evidence_pixel_sha256=name*64,evidence_thumbnail=(raw+delta).tobytes().hex(),evidence_aspect=1)
        db.execute('INSERT INTO observations VALUES (?,?,?,NULL)',(name,json.dumps(q),vector.tobytes()))
    db.commit()
    return db


def test_subset_does_not_compose_nontransitive_pixel_edges():
    with closing(pixel_db([('a',0),('b',2),('c',4)])) as source, closing(pixel_db([('b',2),('c',4)])) as replay:
        assert dict(aliases(source))=={'b'*64:'a'*64}
        assert dict(aliases(replay))=={'c'*64:'b'*64}
        with alias_scope(source):canonicalize(replay)
        assert {r['id']:json.loads(r['quality'])['evidence_pixel_sha256'] for r in replay.execute('SELECT * FROM observations')}=={'b':'a'*64,'c':'c'*64}


def test_enclosing_budget_used_once_and_index_stores_addresses():
    # A single 4096-dimensional witness exceeds this budget if copied into the index.
    with closing(pixel_db([('a',0),('b',1)],4096)) as db:
        with alias_scope(db,budget=8192):canonicalize(db)
        assert {json.loads(r[0])['evidence_pixel_sha256'] for r in db.execute('SELECT quality FROM observations')}=={'a'*64}
    from stash_ai.recognition.resource_errors import ResourceLimit
    with closing(pixel_db([('a',0)])) as db, pytest.raises(ResourceLimit,match='Pixel witness index'):
        with alias_scope(db,budget=64):canonicalize(db)


def test_warm_hybrid_rejects_corrupt_thumbnail(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.rebuild import rebuild_selected
    config,_=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        raw=np.random.default_rng(7).integers(10,235,3072,dtype='uint8').tobytes()
        key=hashlib.sha256(raw).hexdigest()
        db.execute('INSERT INTO evidence_thumbnails VALUES (?,?)',(key,raw))
        for (oid,) in db.execute('SELECT id FROM observations').fetchall():
            db.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_thumbnail_id',?,'$.evidence_pixel_sha256',?,'$.evidence_aspect',1) WHERE id=?",(key,hashlib.sha256(oid.encode()).hexdigest(),oid))
        db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans ORDER BY scene')]
        for _ in range(3):rebuild_selected(state,selected,config)
        assert state.hybrid_rebuild_mode=='warm'
        db.execute('UPDATE evidence_thumbnails SET pixels=zeroblob(3072) WHERE id=?',(key,));db.commit()
        with pytest.raises(ValueError,match='Missing or changed evidence thumbnail'):
            rebuild_selected(state,selected,config)
