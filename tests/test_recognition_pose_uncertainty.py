"""Pose hints may recover supported queries, never train another look."""
import json
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.view_prototypes import query_pose_matches, pose_matches
from tests.test_recognition_anchor_views import config_for, A, X, C
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('count,same_time,accepted', [(2,False,True),(1,False,False),(2,True,False)])
def test_boundary_recovery_requires_independent_queries(tmp_path,count,same_time,accepted):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,_=scan(db,config,'1',[A]*3+[X]*3,count=6)
        second,_=scan(db,config,'2',[C]*count,count=count)
        for row in db.connection.execute('SELECT id,geometry FROM observations WHERE scan=?',(second,)).fetchall():
            g=json.loads(row['geometry']);g[8]=80+.16*62
            assert not pose_matches(g,'other') and query_pose_matches(g,'other',relaxed=True)
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?',(json.dumps(g),row['id']))
        if same_time:db.connection.execute('UPDATE observations SET seconds=0 WHERE scan=?',(second,))
        db.connection.commit()
        selected=[dict(scene='1',scan=first),dict(scene='2',scan=second)]
        rebuild_selected(db,selected,config)
        rows=candidates(db,second,config)
        assert all((r[0]=='accepted')==accepted for r in rows)
        if accepted:
            assert all(r[2]['reason']=='pose_uncertain_identity' and r[2]['reference_role']=='attachment' for r in rows)
            assert all(r[2]['pose_recovery']['minimum_query_support']==2 for r in rows)
        before=[(r[0],r[1]) for r in rows]
        db.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild:%'")
        db.connection.commit()
        rebuild_selected(db,selected,config)
        assert [(r[0],r[1]) for r in candidates(db,second,config)]==before


def test_confident_opposite_sides_and_invalid_geometry_do_not_use_fallback():
    g=[16,16,128,128,49,60,111,60,80+62*.3,84,56,108,104,108,.99]
    assert not query_pose_matches(g,'left',relaxed=True)
    assert not query_pose_matches('invalid','left',relaxed=True)
    assert not query_pose_matches(g,'other')


def test_strong_competing_anchor_still_vetoes_pose_fallback(tmp_path):
    import numpy as np
    from stash_ai.recognition.clustering import Group
    from stash_ai.recognition.global_views import IdentityView
    from stash_ai.recognition.historical_references import History, rank
    config=config_for(tmp_path)
    g=[16,16,128,128,49,60,111,60,80+.16*62,84,56,108,104,108,.99]
    row=dict(id='q',geometry=g,dimensions=4,embedding=np.asarray(C,dtype='<f4').tobytes())
    view=IdentityView(Group('owner',(X,),True),'other','x',('x','y','z'),3,(X,))
    owner=Group('owner',(A,X),True)
    bank=History((owner,),(view,),{},'test')
    assert rank([row],bank,config) is None
    assert rank([row],bank,config,relaxed_pose=True)[0]=='owner'
    # A raw rival cannot disappear just because its whole core is incoherent
    # with the query or has no supported appearance of its own.
    rival=Group('rival',(C,np.array([0.,0.,1.,0.])),True)
    bank.groups+=(rival,)
    assert rank([row],bank,config,relaxed_pose=True) is None


def test_external_root_is_inserted_and_recovered_queries_never_train(tmp_path):
    from stash_ai.recognition.historical_references import capture
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,_=scan(db,config,'1',[A]*3+[X]*3,count=6)
        rebuild_selected(db,[dict(scene='1',scan=first)],config)
        second,_=scan(db,config,'2',[C]*2,count=2)
        for row in db.connection.execute('SELECT id,geometry FROM observations WHERE scan=?',(second,)).fetchall():
            g=json.loads(row['geometry']);g[8]=80+.16*62
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?',(json.dumps(g),row['id']))
        db.connection.commit()
        rebuild_selected(db,[dict(scene='2',scan=second)],config)
        rows=candidates(db,second,config)
        assert all(r[0]=='accepted' and r[2]['reason']=='pose_uncertain_identity' for r in rows)
        ids={r[0] for r in db.connection.execute('SELECT id FROM observations WHERE scan=?',(second,))}
        scope,namespace=db.connection.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
        history=capture(db.connection,[],config,scope,namespace,'test')
        assert all(not ids.intersection(v.support) for v in history.views)
        assert all(not ids.intersection(g.sample_ids) for g in history.groups)
