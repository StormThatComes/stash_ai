"""One unsupported appearance cannot publish a duplicate for a supported cohort."""
import math,json
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates
from tests.test_recognition_scoped_global_identities import selected


def vector(degrees):
    angle=math.radians(degrees);return np.array([math.cos(angle),math.sin(angle),0.,0.])


@pytest.mark.parametrize('joint',[False,True])
def test_supported_cohort_reuses_identity_and_withholds_one_outlier(tmp_path,joint):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,_=scan(db,config,'1',[A]*9+[X]*3,count=12)
        if not joint:rebuild_selected(db,[{'scene':'1','scan':first}],config)
        second,_=scan(db,config,'2',[C]*8+[vector(84)],count=9)
        selection=[{'scene':'1','scan':first},{'scene':'2','scan':second}]
        rebuild_selected(db,selection if joint else selection[1:],config)
        owner=candidates(db,first,config)[0][1]
        rows=candidates(db,second,config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment' for r in rows[:8])
        assert rows[-1][0]=='withheld' and rows[-1][1] is None
        assert rows[-1][2]['reason']=='cohort_unresolved'
        assert all(r[2]['cohort_resolution']['frozen'] for r in rows[:8])
        rebuild_selected(db,selection if joint else selection[1:],config)
        assert candidates(db,second,config)==rows


@pytest.mark.parametrize('signed',[.119,.121])
@pytest.mark.parametrize('reverse',[False,True])
def test_training_pose_boundary_preserves_independent_view_support(tmp_path,signed,reverse):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,_=scan(db,config,'1',[A]*9+[X]*3,count=12)
        second,_=scan(db,config,'2',[C]*3,count=3)
        for row in db.connection.execute('SELECT id,cue,geometry FROM observations WHERE scan=? AND cue>=9',(first,)).fetchall():
            g=json.loads(row['geometry']);d=math.hypot(g[6]-g[4],g[7]-g[5])
            g[8]=(g[4]+g[6])/2+(.119 if row['cue']!=(9 if reverse else 11) else signed)*d
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?',(json.dumps(g),row['id']))
        db.connection.commit()
        rebuild_selected(db,[{'scene':'1','scan':first},{'scene':'2','scan':second}],config)
        owner=candidates(db,first,config)[0][1]
        assert all(r[0]=='accepted' and r[1]==owner for r in candidates(db,second,config))


def test_training_pose_cannot_bridge_opposite_directions():
    from stash_ai.recognition.view_prototypes import _Cluster
    slot=_Cluster('right','first')
    def geo(signed):
        return [16,16,128,128,49,60,111,60,80+62*signed,84,56,108,104,108,.99]
    for i,signed in enumerate((.121,.119)):
        assert slot.accepts_pose(geo(signed))
        slot.add(dict(scan='s',seconds=i,id=str(i),geometry=geo(signed)),A)
    assert not slot.accepts_pose(geo(-.121))


def test_one_view_query_is_not_independent_cohort_support(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=selected(db,config,'1',[A]*3+[X]*3)
        owner=candidates(db,first,config)[0][1]
        second=selected(db,config,'2',[C]+[vector(84)]*3)
        assert all(r[1]!=owner for r in candidates(db,second,config))
