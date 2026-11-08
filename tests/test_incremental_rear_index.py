from contextlib import closing
import sqlite3
from types import SimpleNamespace

import numpy as np
from stash_ai.recognition.rear_reverse import ReverseIndex


def test_removed_rival_and_newly_reachable_queries_are_both_revisited():
    from stash_ai.recognition.rear_gallery import Gallery
    from stash_ai.recognition.rear_pending import _bank_versions
    axis = np.eye(4,dtype='<f4')
    def pending(entries):
        banks = [(p,'s',[dict(id=p,roots=[p+'a',p+'b'],quality='{}',_vector=v)]) for p,v in entries]
        gallery = Gallery(banks)
        return SimpleNamespace(banks=_bank_versions(gallery),gallery=gallery,radius=.1,floor=.8,
                               search=SimpleNamespace(mean_versions={}))
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value)')
        old = pending([('A',axis[0]),('B',-axis[0])])
        index = ReverseIndex(db,'scope','ns','v',old)
        from stash_ai.recognition.config import digest
        def post(scan,q,owners=()):
            index.post(scan,index.bins(q,0)|{'o'+digest(p) for p in owners})
            index.state['scans'][scan] = [scan]
        post('far',-axis[0])
        post('old-competitor',axis[0],['B'])
        post('new-query',axis[1])
        index.finish()
        new = pending([('A',axis[0]),('C',axis[1])])
        again = ReverseIndex(db,'scope','ns','v',new)
        assert again.nominate(new,[])=={'old-competitor','new-query'}
        assert again.nominate(new,None) is None


def test_disk_gallery_reuses_only_unchanged_owner_bank_bounds(tmp_path):
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition import derived_cache
    from stash_ai.recognition.rear_gallery import load
    from stash_ai.recognition.body_identity import rebuild
    from tests.test_recognition_body_global import configured,scene
    cfg=configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        front,back=[1.,0.],[.82,.5724]
        item,ids=scene(db,cfg,'1',[front,back,front,front,back,front],times=[0,1,2,5,6,7])
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'A'))
        for i,oid in enumerate(ids):
            rooted=i in (0,2,3,5)
            cue=[0,1,2,5,6,7][i]
            q=dict(pixel_digest=oid,keypoints=10,sharpness=100,orientation='front' if rooted else 'back',
                   frame_signature=(bytes((15 if i<3 else 180,15,15))*16).hex())
            db.execute('UPDATE body_observations SET quality=?,geometry=? WHERE id=?',
                       (encoded(dict(quality=q)),encoded(dict(bounds=[0,0,80,100])),oid))
            if rooted:
                fid='face-'+oid
                quality=encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (fid,item['scan'],cue,float(cue),'','[]','[]',quality,None,2,np.array(front,dtype='<f4').tobytes(),None))
                db.execute('INSERT INTO members VALUES (?,?)',(fid,'A'))
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,oid))
        rebuild(state,[item],cfg,{})
        cfg.data['execution']['derived_cache_bytes']=32*1024**2
        cfg.data['execution']['rebuild_cache_bytes']=32*1024**2
        def gallery():
            with derived_cache.scope(state,cfg),closing(sqlite3.connect(':memory:')) as pages:
                cached=load(db,cfg,'scope','ns',pages=pages)
                return cached.versions,derived_cache.current().hits
        versions,_=gallery()
        assert versions
        reused,hits=gallery()
        assert reused==versions
        assert {'body-appearance-rear-manifest','body-appearance-rear-bounds'}<=hits
        row=db.execute('SELECT embedding FROM body_observations WHERE id=?',(ids[1],)).fetchone()
        diff=np.frombuffer(row[0],dtype='<f4').copy()
        diff[:2]=[.8,.6]
        db.execute('UPDATE body_observations SET embedding=? WHERE id=?',(diff.tobytes(),ids[1]))
        revised,hits=gallery()
        assert revised!=versions
        with closing(sqlite3.connect(':memory:')) as pages:
            assert load(db,cfg,'scope','ns',pages=pages).versions==revised
