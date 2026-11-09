import sqlite3
from types import SimpleNamespace

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rear_scene_changes import Directory
from tests.test_recognition_body_identity import configuration, scene


def test_directory_selects_changed_and_nominated_scans_across_rollback(tmp_path):
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        items=[scene(db,config,str(i))[0] for i in range(1,7)]
        index=SimpleNamespace(previous={},state={},prefix='test:',post=lambda *args:None)
        directory=Directory(db,'scope','ns',index)
        assert directory.select(set(),None,0) is None
        for item in items:
            directory.remember(item['scene'],item['scan'],[item['scan']])
        db.commit()
        index.previous=dict(index.state)
        added,_=scene(db,config,'7')
        db.commit()
        db.execute('BEGIN')
        assert directory.select(set(),{items[0]['scan']},0)=={items[0]['scan'],added['scan']}
        db.rollback()
        assert directory.select(set(),set(),0)=={added['scan']}
        db.commit()
        for table in ('body_cues','body_observations'):
            db.execute(f'DELETE FROM {table} WHERE scan=?',(items[-1]['scan'],))
        db.execute('DELETE FROM scans WHERE id=?',(items[-1]['scan'],))
        assert directory.select(set(),set(),0)==set()
        assert not db.execute('SELECT 1 FROM rear_indexed_scenes WHERE scan=?',(items[-1]['scan'],)).fetchone()
        assert directory.select(set(),set(),1) is None


def test_v15_migration_preserves_members_and_seeds_full_first_pass(tmp_path):
    from stash_ai.recognition.database import SCHEMA_V15, SCHEMA_VERSION, APPLICATION_ID
    path=tmp_path/'old.db'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V15)
        db.execute('PRAGMA user_version=15')
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute("INSERT INTO metadata VALUES ('installation','00000000-0000-0000-0000-000000000001')")
    with FaceDatabase(path) as state:
        assert state.connection.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION
        assert state.connection.execute('SELECT count(*) FROM rear_indexed_scenes').fetchone()[0]==0


def test_pending_reads_only_new_scene_header(tmp_path, monkeypatch):
    import numpy as np
    from stash_ai.recognition import profiling
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.rear_gallery import Gallery
    from stash_ai.recognition.rear_pending import Pending
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        stamp=policy(config)
        def add(i):
            item,ids=scene(db,config,str(i))
            db.executemany('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
                           [(oid,None,'auxiliary','withheld',stamp,'{}') for oid in ids])
            return item['scan']
        initial={add(i) for i in range(1,11)}
        v=np.zeros(4096,dtype='<f4')
        v[0]=1.
        gallery=Gallery([('A','external',[dict(id='a',roots=['l','r'],quality='{}',_vector=v)])])
        def replay():
            pending=Pending(db,gallery,SimpleNamespace(bank={},mean_versions={}),config.data['sapiensid']['identities'])
            return {s for s,_ in pending.candidates('scope','ns',set(),stamp)}
        assert replay()==initial
        scan=add(11)
        counts=[]
        monkeypatch.setattr(profiling,'count',lambda name,*args:counts.append(name))
        assert replay()=={scan}
        assert counts.count('rear_scene_headers')==1
        assert not replay()
