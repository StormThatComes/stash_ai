"""Later view evidence and complete components repair fragmented identities."""
import json
import math
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.global_views import merge_views
from tests.test_recognition_anchor_views import config_for,A
from tests.test_recognition_clustering import scan
from tests.test_recognition_global_views import seed,views
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('first_a',[True,False])
def test_later_confirmations_update_view_medoid(tmp_path,first_a):
    config=config_for(tmp_path)
    b=np.array([.84,math.sqrt(1-.84**2),0.,0.]);c=np.array([.5,math.sqrt(.75),0.,0.])
    vectors=([A]*8+[b]*12 if first_a else [b]*12+[A]*8)+[c]*3
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',vectors,count=len(vectors))
        rebuild_selected(db,[{'scene':'1','scan':sid}],config)
        rows=candidates(db,sid,config)
        assert all(r[0]=='accepted' and r[1]==rows[0][1] for r in rows)


def test_three_mutually_coherent_fragments_merge_as_one_component(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        for i in range(3):
            a=math.radians(i*12);seed(db,config,str(i+1),np.array([math.cos(a),math.sin(a),0.,0.]))
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==3
        merge_views(db,views(db,config),config,matching_policy(config))
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1
        proofs=[json.loads(r[0]) for r in db.connection.execute('SELECT evidence FROM identity_merges')]
        assert len(proofs)==2 and all(p['complete_component'] and p['component_size']==3 for p in proofs)


@pytest.mark.parametrize('signed',[.119,.121])
def test_pose_boundary_does_not_hide_a_strong_view(tmp_path,signed):
    config=config_for(tmp_path)
    b=np.array([.84,math.sqrt(1-.84**2),0.,0.]);c=np.array([.5,math.sqrt(.75),0.,0.])
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',[b]*12+[A]*8+[c]*3,count=23)
        for row in db.connection.execute('SELECT id,geometry FROM observations WHERE scan=? AND cue>=20',(sid,)).fetchall():
            g=json.loads(row['geometry']);distance=math.hypot(g[4]-g[6],g[5]-g[7]);g[8]=(g[4]+g[6])/2+signed*distance
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?',(json.dumps(g),row['id']))
        db.connection.commit();rebuild_selected(db,[{'scene':'1','scan':sid}],config)
        assert all(r[0]=='accepted' for r in candidates(db,sid,config)[20:])


def test_bounded_view_pool_retains_later_extreme_and_quality():
    from stash_ai.recognition.view_prototypes import _Cluster
    slot=_Cluster('other','first')
    def row(i,quality='{}'):
        return dict(id=str(i),scan='s',seconds=i,quality=quality)
    for i in range(8):
        slot.add(row(i),A)
    angle=math.radians(36);edge=np.array([math.cos(angle),math.sin(angle),0.,0.])
    assert slot.accepts(edge,.8)
    slot.add(row(8),edge)
    assert len(slot.examples)<=8
    opposite=np.array([math.cos(angle),-math.sin(angle),0.,0.])
    assert not slot.accepts(opposite,.8)
    slot.add(row(9,json.dumps(dict(pixels=512,sharpness=1000,pose_deviation=0))),A)
    assert slot.medoid()[1]=='9'
    slot.add(row(9),opposite)
    assert len(slot.cues)==10  # A duplicate timestamp cannot move the centroid.
