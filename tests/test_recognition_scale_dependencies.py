from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.scene_artifacts import body_key
from stash_ai.recognition.dependencies import version
from stash_ai.recognition.local_dependencies import scene_key
from tests.test_recognition_body_identity import configuration,scene


def test_append_keeps_old_scene_fingerprints_without_rereading_features(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection
        a,ids=scene(db,c,'1');b,_=scene(db,c,'2')
        first=[body_key(db,x['scan'],c) for x in (a,b)];db.commit()
        scene(db,c,'3');statements=[];db.set_trace_callback(statements.append)
        assert [body_key(db,x['scan'],c) for x in (a,b)]==first
        assert not any(sql.startswith('SELECT b.id,b.cue,b.seconds') for sql in statements)
        db.set_trace_callback(None)
        before=version(db,'scene',scene_key(db,a['scan']))
        db.execute('UPDATE body_observations SET embedding=zeroblob(16384) WHERE id=?',(ids[0],))
        assert version(db,'scene',scene_key(db,a['scan']))>before
        assert body_key(db,a['scan'],c)!=first[0]
        db.rollback()
        assert body_key(db,a['scan'],c)==first[0]


def test_owner_role_and_head_edits_invalidate_local_clocks(tmp_path):
    from tests.test_recognition_clustering import setup,scan
    c=setup(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;sid,_=scan(state,c,'1');oid=db.execute('SELECT id FROM observations LIMIT 1').fetchone()[0]
        db.execute("INSERT INTO people VALUES ('owner','scope','ns',0,NULL,'token')")
        db.execute('INSERT INTO members VALUES (?,\'owner\')',(oid,));before=version(db,'owner','owner')
        db.execute('UPDATE observations SET quality=json_set(quality,\'$.pose_deviation\',.5) WHERE id=?',(oid,))
        assert version(db,'owner','owner')>before
        heads=version(db,'heads')
        db.execute("INSERT INTO metadata VALUES ('recognition-scan-head:synthetic','{}')")
        assert version(db,'heads')>heads

def test_index_moves_owner_and_rolls_back_with_membership(tmp_path):
    from tests.test_recognition_clustering import setup,scan
    c=setup(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;sid,_=scan(state,c,'1')
        ids=[row[0] for row in db.execute('SELECT id FROM observations LIMIT 2')]
        for owner in ('a','b'):
            db.execute("INSERT INTO people VALUES (?,'scope','ns',0,NULL,?)",(owner,owner))
        db.executemany("INSERT INTO members VALUES (?,'a')",[(x,) for x in ids]);db.commit()
        db.execute("INSERT OR REPLACE INTO members VALUES (?,'b')",(ids[0],))
        assert dict(db.execute('SELECT person,total FROM scene_owners WHERE scan=?',(sid,)))=={'a':1,'b':1}
        db.rollback()
        assert dict(db.execute('SELECT person,total FROM scene_owners WHERE scan=?',(sid,)))=={'a':2}
