"""Reference access is independent of selection scope and query pose role."""
import math
import pytest
from dataclasses import replace
from stash_ai.recognition.config import plain
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates
from tests.test_recognition_scoped_global_identities import selected,snapshot


@pytest.mark.parametrize("query",[A,C])
def test_auxiliary_only_selection_reuses_external_identity(tmp_path,query):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3+[X]*3)
        person=candidates(db,first,config)[0][1]
        before=snapshot(db,first)
        second,_=scan(db,config,'2',[query]*3,count=3)
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.3) WHERE scan=?",(second,))
        db.connection.commit()
        selection=[{'scene':'2','scan':second}]
        rebuild_selected(db,selection,config)
        rows=candidates(db,second,config)
        assert all(r[0]=='accepted' and r[1]==person and r[2]['reference_role']=='attachment' for r in rows)
        assert snapshot(db,first)==before
        rebuild_selected(db,selection,config)
        assert candidates(db,second,config)==rows


def test_final_strict_retry_recovers_earlier_singleton(tmp_path):
    config=config_for(tmp_path)
    def v(deg):
        angle=math.radians(deg);return np.array([math.cos(angle),math.sin(angle),0.,0.])
    with FaceDatabase(config.database) as db:
        first,_=scan(db,config,'1',[A]*3+[v(55)],count=4)
        second,_=scan(db,config,'2',[v(80)]*3+[v(55)]*2,count=5)
        rebuild_selected(db,[{'scene':'1','scan':first},{'scene':'2','scan':second}],config)
        row=candidates(db,first,config)[-1]
        assert row[0]=='accepted' and row[1]==candidates(db,second,config)[0][1]
        assert row[2]['reason']=='late_strict_identity'
        assert row[2]['reference_role']=='attachment'


def test_anchor_and_auxiliary_boundary_share_confirmed_view(tmp_path):
    results=[]
    for pose in (.199,.201):
        folder=tmp_path/str(pose);folder.mkdir()
        config=config_for(folder,view_prototypes_enabled=True)
        with FaceDatabase(config.database) as db:
            sid,_=scan(db,config,'1',[A]*3+[X]*3+[C]*3,count=9)
            db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.3) WHERE scan=? AND cue BETWEEN 3 AND 5",(sid,))
            db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',?) WHERE scan=? AND cue>=6",(pose,sid))
            db.connection.commit()
            rebuild_selected(db,[{'scene':'1','scan':sid}],config)
            rows=candidates(db,sid,config)
            assert all(r[0]=='accepted' for r in rows)
            assert all(r[2]['reference_role']=='attachment' for r in rows[6:])
            results.append(len({r[1] for r in rows}))
            data=plain(config.data);data['recognition']['min_merge_support']=4
            cold=replace(config,data=data)
            rebuild_selected(db,[{'scene':'1','scan':sid}],cold)
            assert len({r[1] for r in candidates(db,sid,cold) if r[0]=='accepted'})==1
    assert results==[1,1]
