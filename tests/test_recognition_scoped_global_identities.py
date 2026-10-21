"""Selected replay reuses frozen global labels without rewriting other scenes."""
from dataclasses import replace
import math

import numpy as np

from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for, A, X, C
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates
from tests.test_recognition_rebuild import library, changed
from tests.test_recognition_run import run


def selected(db,config,scene,vectors):
    sid,_=scan(db,config,scene,vectors,count=len(vectors))
    rebuild_selected(db,[{'scene':scene,'scan':sid}],config)
    return sid


def snapshot(db,sid):
    return {table:[tuple(row) for row in db.connection.execute(sql,(sid,))] for table,sql in {
        'members':'SELECT m.* FROM members m JOIN observations o ON o.id=m.observation WHERE o.scan=? ORDER BY m.observation',
        'candidates':'SELECT * FROM candidates WHERE scan=? ORDER BY id',
        'observations':'SELECT * FROM observations WHERE scan=? ORDER BY id'}.items()}


def test_separate_and_combined_scene_runs_keep_one_identity(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3+[X]*3)
        owner=candidates(db,first,config)[0][1]
        before=snapshot(db,first)
        second=selected(db,config,'2',[C]*3)
        assert {r[1] for r in candidates(db,second,config)}=={owner}
        assert snapshot(db,first)==before
        assert all(r[2]['historical_reference']['mode']=='view' for r in candidates(db,second,config))
        rebuild_selected(db,[{'scene':'1','scan':first},{'scene':'2','scan':second}],config)
        assert {r[1] for r in candidates(db,second,config)}=={owner}


def test_existing_identity_can_recover_one_strictly_matching_selected_cue(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3)
        owner=candidates(db,first,config)[0][1]
        second=selected(db,config,'2',[A])
        row=candidates(db,second,config)[0]
        assert row[0]=='accepted' and row[1]==owner
        assert row[2]['reason']=='historical_identity'


def test_incompatible_scan_policy_cannot_supply_external_reference(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3)
        owner=candidates(db,first,config)[0][1]
        data=plain(config.data)
        data['detection']['min_face_pixels']=64
        changed_config=replace(config,data=data)
        second=selected(db,changed_config,'2',[A]*3)
        assert candidates(db,second,changed_config)[0][1]!=owner


def test_external_reference_change_invalidates_selected_warm_cache(tmp_path,monkeypatch):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3+[X]*3)
        owner=candidates(db,first,config)[0][1]
        second=selected(db,config,'2',[C]*3)
        assert candidates(db,second,config)[0][1]==owner
        # Remove the confirmed X cohort while leaving the frontal identity.
        db.connection.execute('DELETE FROM members WHERE observation IN '
                              '(SELECT id FROM observations WHERE scan=? AND cue>=3)',(first,))
        db.connection.commit()
        from stash_ai.recognition import historical_references
        calls=[]
        original=historical_references.reconcile
        def reconcile(*args,**kwargs):
            calls.append(True)
            return original(*args,**kwargs)
        monkeypatch.setattr(historical_references,'reconcile',reconcile)
        rebuild_selected(db,[{'scene':'2','scan':second}],config)
        assert calls
        assert 'historical_reference' not in candidates(db,second,config)[0][2]


def test_partial_run_reuses_stash_performer_without_creating_duplicate(tmp_path,monkeypatch):
    vectors={'1':[A]*3+[X]*3,'2':[C]*3}
    config,client,services,_=library(tmp_path,monkeypatch,vectors)
    config=changed(config,recognition={'min_identity_support':3},
        clustering={'max_anchor_pose_deviation':.2,'anchor_view_matching':True,'same_scene_match_bonus':0})
    assert run(config,services,'1')==0
    owner=client.links['1'][-1]
    client.profiles[owner]['name']='Keep my manual name'
    assert run(config,services,'2')==0
    assert client.links['2']==[owner]
    assert client.links['1']==['77',owner]
    assert client.profiles[owner]['name']=='Keep my manual name'
    assert sum(write[0]=='create' for write in client.writes)==1
    writes=list(client.writes)
    assert run(config,services,'2')==0
    assert client.writes==writes


def test_external_view_attachments_cannot_train_later_partial_runs(tmp_path):
    config=config_for(tmp_path)
    beyond=np.array([.1,math.sqrt(1-.1**2),0.,0.])
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3+[X]*3)
        owner=candidates(db,first,config)[0][1]
        second=selected(db,config,'2',[C]*3)
        assert all(r[1]==owner and r[2]['reference_role']=='attachment' for r in candidates(db,second,config))
        third=selected(db,config,'3',[beyond]*3)
        assert all(r[1]!=owner for r in candidates(db,third,config))


def multiple_views_fixture(db,config):
    plus=np.array([.75,math.sqrt(1-.75**2),0.,0.])
    minus=np.array([.75,-math.sqrt(1-.75**2),0.,0.])
    first=selected(db,config,'1',[A]*3+[plus]*3)
    owner=candidates(db,first,config)[0][1]
    other=selected(db,config,'9',[minus]*3)
    fragment=candidates(db,other,config)[0][1]
    # A verified historical identity has independently confirmed opposed views.
    db.connection.execute('UPDATE members SET person=? WHERE person=?',(owner,fragment))
    db.connection.execute('UPDATE candidates SET person=? WHERE person=?',(owner,fragment))
    db.connection.commit()
    return owner


def test_multiple_local_fragments_can_share_one_frozen_global_identity(tmp_path):
    config=config_for(tmp_path)
    right=np.array([.55,math.sqrt(1-.55**2),0.,0.])
    left=np.array([.55,-math.sqrt(1-.55**2),0.,0.])
    with FaceDatabase(config.database) as db:
        owner=multiple_views_fixture(db,config)
        sid=selected(db,config,'2',[right]*3+[left]*3)
        assert all(r[0]=='accepted' and r[1]==owner for r in candidates(db,sid,config))
        assert len({r[1] for r in candidates(db,sid,config)})==1


def test_co_present_local_fragments_cannot_adopt_same_global_identity(tmp_path):
    config=config_for(tmp_path)
    right=np.array([.55,math.sqrt(1-.55**2),0.,0.])
    left=np.array([.55,-math.sqrt(1-.55**2),0.,0.])
    with FaceDatabase(config.database) as db:
        owner=multiple_views_fixture(db,config)
        sid,_=scan(db,config,'2',[right]*3+[left]*3,count=6)
        db.connection.execute('UPDATE observations SET cue=0,seconds=0 WHERE scan=? AND cue=3',(sid,))
        db.connection.commit()
        rebuild_selected(db,[{'scene':'2','scan':sid}],config)
        assert len({r[1] for r in candidates(db,sid,config) if r[0]=='accepted'})==2
        assert any(r[1]!=owner for r in candidates(db,sid,config))


def test_co_present_withheld_views_cannot_recover_same_global_identity(tmp_path):
    config=config_for(tmp_path)
    right=np.array([.55,math.sqrt(1-.55**2),0.,0.])
    left=np.array([.55,-math.sqrt(1-.55**2),0.,0.])
    with FaceDatabase(config.database) as db:
        owner=multiple_views_fixture(db,config)
        sid,_=scan(db,config,'2',[right]*2+[left]*2,count=4)
        db.connection.execute('UPDATE observations SET cue=cue-2,seconds=seconds-2 '
                              'WHERE scan=? AND cue>=2',(sid,))
        db.connection.commit()
        rebuild_selected(db,[{'scene':'2','scan':sid}],config)
        assert not any(r[0]=='accepted' and r[1]==owner for r in candidates(db,sid,config))


def test_strict_historical_match_uses_strict_ambiguity_margin(tmp_path):
    config=config_for(tmp_path)
    rival=np.array([0.,1.,0.,0.])
    query=np.array([.70,.59,math.sqrt(1-.70**2-.59**2),0.])
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3)
        owner=candidates(db,first,config)[0][1]
        selected(db,config,'2',[rival]*3)
        third=selected(db,config,'3',[query])
        row=candidates(db,third,config)[0]
        assert row[0]=='accepted' and row[1]==owner
        assert row[2]['historical_reference']['ambiguity_margin']==config.data['recognition']['ambiguity_margin']
