import sqlite3
import uuid
from unittest.mock import patch
import numpy as np
from stash_ai.recognition.database import FaceDatabase, SCHEMA_V7, APPLICATION_ID
from stash_ai.recognition.dependencies import version
from stash_ai.recognition.hybrid_evidence import catalogue
from stash_ai.recognition.hybrid_jobs import copresence
from stash_ai.recognition.hybrid_query_index import QueryIndex
from stash_ai.recognition.run_cache import scope
from tests.test_recognition_hybrid_merge import prepared
from tests.test_recognition_rebuild_cache import setup,assignment_fixture


def test_v7_migration_installs_raw_stamps_without_changing_installation(tmp_path):
    config=setup(tmp_path);installation=str(uuid.uuid4())
    with sqlite3.connect(config.database) as db:
        db.executescript(SCHEMA_V7)
        db.execute('INSERT INTO metadata VALUES (?,?)',('installation',installation))
        db.execute(f'PRAGMA application_id={APPLICATION_ID}');db.execute('PRAGMA user_version=7')
    with FaceDatabase(config.database) as state:
        assert state.installation==installation
        selected=assignment_fixture(state);db=state.connection;sid=selected[0]['scan']
        original=version(db,'raw-observations',sid)
        db.execute('UPDATE candidates SET status=status WHERE scan=?',(sid,))
        assert version(db,'raw-observations',sid)==original
        db.execute('UPDATE observations SET seconds=seconds+1 WHERE scan=?',(sid,))
        assert version(db,'raw-observations',sid)>original
        db.rollback()
        assert version(db,'raw-observations',sid)==original


def test_catalogue_reuse_and_cooccurrence_parity(tmp_path):
    from stash_ai.recognition.global_views import _cooccurs_sql
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state,scope(config):
        db=state.connection;ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        first=catalogue(db,config,'scope',ns)
        with patch('stash_ai.recognition.hybrid_evidence._catalogue',side_effect=AssertionError('reread')):
            assert catalogue(db,config,'scope',ns) is first
        assert not first[0].faces.flags.writeable
        pairs=[tuple(people[:2])]
        assert copresence(db,pairs,config)=={p:_cooccurs_sql(db,*p,config) for p in pairs}
        db.execute('UPDATE members SET person=person')
        assert catalogue(db,config,'scope',ns) is not first


def test_query_shortlist_never_loses_exact_face_witness():
    rng=np.random.default_rng(42)
    banks=[(str(i),None,None,rng.normal(size=(4,16)).astype('float32')) for i in range(180)]
    banks=[(i,g,np.empty((0,4096)),f/np.linalg.norm(f,axis=1)[:,None]) for i,g,b,f in banks]
    index=QueryIndex(banks,10**7)
    for _ in range(10):
        query=rng.normal(size=16).astype('float32');query/=np.linalg.norm(query)
        for floor in [-1.,0.,.5,1.]:
            candidates={b[0] for b in index.candidates(query,floor)}
            exact={i for i,_,_,f in banks if (f.astype('float64') @ query.astype('float64')).min()>=floor}
            assert exact<=candidates
