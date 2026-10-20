"""Cross-scene appearance recovery without reference feedback or chaining."""
from dataclasses import replace
import math

import numpy as np

from stash_ai.recognition.clustering import cluster_scene, matching_policy
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.global_views import catalogue, merge_views
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.reference_usage import ReferenceUsage
from tests.test_recognition_anchor_views import config_for, A, X, C
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates


def replay(db,config,scenes):
    selected=[]
    for scene,vectors in scenes.items():
        sid,_=scan(db,config,scene,vectors,count=len(vectors))
        selected.append({'scene':scene,'scan':sid})
    rebuild_selected(db,selected,config)
    return selected


def test_two_other_scenes_can_supply_independent_attachment_support(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,{'1':[A]*3+[X]*3,'2':[C],'3':[C]})
        rows=[candidates(db,item['scan'],config) for item in selected]
        person=rows[0][0][1]
        assert all(row[0]=='accepted' and row[1]==person for group in rows for row in group)
        assert rows[1][0][2]['reason']==rows[2][0][2]['reason']=='global_view_identity'
        assert rows[1][0][2]['reference_role']=='attachment'


def test_recovered_view_cannot_train_another_recovery(tmp_path):
    config=config_for(tmp_path)
    beyond=np.array([.1,math.sqrt(1-.1**2),0.,0.])
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,{'1':[A]*3+[X]*3,'2':[C]*2,'3':[beyond]*2})
        assert all(r[0]=='accepted' for r in candidates(db,selected[1]['scan'],config))
        assert all(r[0]=='withheld' for r in candidates(db,selected[2]['scan'],config))


def seed(db,config,scene,vector):
    data=plain(config.data)
    data['recognition']['match_threshold']=.99
    data['recognition']['ambiguity_margin']=.001
    strict=replace(config,data=data)
    sid,_=scan(db,config,scene,[vector]*3)
    person=cluster_scene(db,sid,strict)[0]['person']
    return sid,person


def views(db,config):
    return catalogue(db.connection,config,'scope','ns',
        usage=ReferenceUsage(config.data['clustering']['max_reference_bytes']))


def test_confirmed_matching_cohorts_can_merge_diverse_fragmented_people(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        _,owner=seed(db,config,'1',A)
        _,part=seed(db,config,'2',X)
        _,fragment=seed(db,config,'3',C)
        # A historical identity legitimately owns two distinct appearances.
        db.connection.execute('UPDATE members SET person=? WHERE person=?',(owner,part))
        db.connection.execute('UPDATE candidates SET person=? WHERE person=?',(owner,part))
        frozen=views(db,config)
        assert len(frozen)==3
        merge_views(db,frozen,config,matching_policy(config))
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==1
        proof=db.connection.execute('SELECT evidence FROM identity_merges').fetchone()[0]
        assert 'reciprocal' in proof and 'support_ids' in proof


def test_neighbour_chain_cannot_merge_distinct_endpoints(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        for scene,degrees in enumerate((0,20,40),1):
            angle=math.radians(degrees)
            seed(db,config,str(scene),np.array([math.cos(angle),math.sin(angle),0.,0.]))
        merge_views(db,views(db,config),config,matching_policy(config))
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==3


def test_co_present_people_need_stronger_reflection_evidence(tmp_path):
    config=config_for(tmp_path)
    left=np.array([1.,0.,0.,0.])
    right=np.array([.85,math.sqrt(1-.85**2),0.,0.])
    with FaceDatabase(config.database) as db:
        first,one=seed(db,config,'1',left)
        second,two=seed(db,config,'2',right)
        # Two distinct detections co-occur in the same source cue.
        row=db.connection.execute('SELECT id FROM observations WHERE scan=? LIMIT 1',(second,)).fetchone()
        db.connection.execute('UPDATE observations SET scan=?,cue=0,seconds=0 WHERE id=?',(first,row['id']))
        db.connection.execute('UPDATE candidates SET scan=? WHERE json_extract(payload,\'$.observation\')=?',(first,row['id']))
        merge_views(db,views(db,config),config,matching_policy(config))
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
