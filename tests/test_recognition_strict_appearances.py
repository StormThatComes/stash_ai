"""A strict appearance needs independent roots, queries, and complete witnesses."""
import json
import math
import numpy as np
import pytest

from stash_ai.recognition.clustering import Group, matching_policy
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.global_views import IdentityView
from stash_ai.recognition.historical_references import History, rank
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.reference_selection import refresh_local
from stash_ai.recognition.reference_usage import ReferenceUsage
from stash_ai.recognition.scanning import scan_identity
from tests.test_recognition_anchor_views import config_for, A, X
from tests.test_recognition_clustering import scan
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates

Q=np.array([math.cos(math.radians(85)),math.sin(math.radians(85)),0.,0.])


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('count',[2,3])
def test_supported_strict_appearance_recovers_queries_and_replays(tmp_path,enabled,count):
    config=config_for(tmp_path,strict_appearance_matching=enabled)
    with FaceDatabase(config.database) as db:
        first,_=scan(db,config,'1',[A]*3+[X]*3,count=6)
        second,_=scan(db,config,'2',[Q]*count,count=count)
        selected=[dict(scene='1',scan=first),dict(scene='2',scan=second)]
        rebuild_selected(db,selected,config)
        rows=candidates(db,second,config)
        owner=candidates(db,first,config)[0][1]
        assert all((r[0]=='accepted' and r[1]==owner)==enabled for r in rows)
        if enabled:
            assert all(r[2]['reference_role']=='attachment' for r in rows)
            assert all(r[2].get('strict_appearance',r[2].get('shared_reference'))['threshold']==config.data['recognition']['match_threshold'] for r in rows)
            before=[r[:2] for r in rows]
            db.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild:%'");db.connection.commit()
            rebuild_selected(db,selected,config)
            assert [r[:2] for r in candidates(db,second,config)]==before


@pytest.mark.parametrize('failure',['support','cohesion','competitor','authority'])
def test_best_medoid_alone_cannot_authorize_strict_appearance(tmp_path,failure):
    config=config_for(tmp_path,strict_appearance_matching=True)
    geometry=[16,16,128,128,49,60,111,60,80,84,56,108,104,108,.99]
    row=dict(id='q',geometry=geometry,dimensions=4,embedding=np.asarray(Q,dtype='<f4').tobytes())
    view=IdentityView(Group('owner',(X,),True),'other','x',('x','y','z'),
                      1 if failure=='support' else 3,
                      (X,A) if failure=='cohesion' else (X,),
                      'corroborated_view' if failure=='authority' else 'strict')
    groups=(Group('owner',(A,X),True),)
    if failure=='competitor':groups+=(Group('rival',(Q,np.array([0.,0.,1.,0.])),True),)
    assert rank([row],History(groups,(view,),{},'test'),config,strict_appearance=True) is None


def test_late_representative_replaces_duplicate_early_look_but_keeps_seed(tmp_path):
    config=config_for(tmp_path,strict_appearance_matching=True,max_samples_per_person=3)
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',[A]*5+[X],count=6)
        rows=list(db.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(sid,)))
        group=Group('p',[A],False,(rows[0]['id'],))
        usage=ReferenceUsage(1024);usage.add((A,),kind='seed')
        for row in rows[1:]:refresh_local(db.connection,group,row,3,usage)
        assert group.sample_ids[0]==rows[0]['id']
        assert rows[-1]['id'] in group.sample_ids
        assert len(group.samples)<=3 and usage.bytes==sum(v.nbytes for v in group.samples)
        assert not group._appearance_pending
        refresh_local(db.connection,group,rows[2],3,usage)
        assert group._appearance_pending
        refresh_local(db.connection,group,None,3,usage)
        assert not group._appearance_pending and group.sample_ids[0]==rows[0]['id']


def test_option_changes_matching_only_and_requires_appearance_bank(tmp_path):
    config=config_for(tmp_path,strict_appearance_matching=True)
    off=changed(config,clustering={'strict_appearance_matching':False})
    assert matching_policy(config)!=matching_policy(off)
    assert scan_identity(config,'ns')==scan_identity(off,'ns')
    from stash_ai.recognition.config import load
    for value in ('maybe','true'):
        p=tmp_path/'bad.yml';p.write_text('version: 1\nclustering: {strict_appearance_matching: '+value+'}\n')
        with pytest.raises(ValueError,match='strict_appearance_matching'):load(p)
