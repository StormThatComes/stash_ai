"""Simultaneous proposals and global ambiguity cannot create identity claims."""
import math
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for
from tests.test_recognition_clustering import scan
from tests.test_recognition_global_views import seed
from tests.test_recognition_scene_context import candidates


def v(degrees):
    angle=math.radians(degrees)
    return np.array([math.cos(angle),math.sin(angle),0.,0.])


def test_distinct_simultaneous_local_views_cannot_claim_one_person(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',[v(-10)]*3+[v(39)]*3+[v(-36)]*2+[v(65)]*2,count=10)
        db.connection.execute('UPDATE observations SET cue=cue-2,seconds=seconds-2 WHERE scan=? AND cue>=8',(sid,))
        db.connection.commit()
        rebuild_selected(db,[{'scene':'1','scan':sid}],config)
        rows=candidates(db,sid,config)
        assert sum(r[0]=='accepted' for r in rows)==6
        assert all(r[2]['reason']=='co_present_identity_conflict' for r in rows if r[0]=='withheld')


def test_ambiguous_historical_fragments_do_not_create_fourth_person(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        people={seed(db,config,str(i+1),v(i*12))[1] for i in range(3)}
        assert len(people)==3
        sid,_=scan(db,config,'4',[v(12)]*3,count=3)
        selection=[{'scene':'4','scan':sid}]
        rebuild_selected(db,selection,config)
        rows=candidates(db,sid,config)
        assert all(r[0]=='withheld' and r[1] is None for r in rows)
        assert all(r[2]['reason']=='historical_ambiguous_identity' for r in rows)
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==3
        assert rebuild_selected(db,selection,config)
        assert candidates(db,sid,config)==rows


def test_exact_duplicate_and_strong_reflection_claims_are_not_conflicts():
    from stash_ai.recognition.decision_resolution import conflicting_claims
    def row(i,sample):
        return dict(id=str(i),person='p',scan='s',cue=0,geometry=str(i),dimensions=4,
                    embedding=sample.astype('<f4').tobytes())
    assert not list(conflicting_claims([row(1,v(0)),row(2,v(10))],.9))
    left=row(1,v(0));right=dict(left,id='2')
    assert not list(conflicting_claims([left,right],.9))
    assert len(list(conflicting_claims([row(1,v(0)),row(2,v(50))],.9)))==2
