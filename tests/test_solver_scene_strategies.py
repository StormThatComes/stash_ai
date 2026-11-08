import json
import sqlite3
from contextlib import closing
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.scene_strategy_validation import omission_damage
from stash_ai.recognition.solver_validation import validate_winner
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.physical_goal import explicit_pairs,freeze
from tests.test_recognition_body_identity import configuration,scene


def test_damage_counts_each_physical_person_once():
    labels={'face':'A','body':'A','turn':'A','partner':'B'}
    coords={'face':('s',0),'body':('s',0),'turn':('s',1),'partner':('s',1)}
    def result(values):return {k:dict(accepted_identity=v) for k,v in zip(labels,values)}
    assert omission_damage(labels,coords,labels,result('AAAB'),result('AABB'))==dict(lost_correct_pairs=1,new_false_joins=1)


def test_six_strategies_share_retained_observations(tmp_path):
    config = configuration(tmp_path)
    ns = namespace(config)
    with FaceDatabase(config.database) as state:
        db=state.connection
        item,ids=scene(db,config)
        db.execute('UPDATE scans SET namespace=?,policy_sha=?',(ns,scan_identity(config,ns)))
        db.commit()
    manifest = tmp_path/'holdout.json'
    manifest.write_text(json.dumps(dict(scenes=['1'],observations={i:dict(truth='A',slices=[]) for i in ids},
                                        series=[['1',0,0]])))
    with closing(sqlite3.connect(':memory:')) as training:
        training.execute('CREATE TABLE scans(scope,namespace,scene)')
        result=validate_winner(config,training,dict(scope='scope',namespace=ns,best_result=dict(changes=[])),
                               manifest,compare_strategies=True)
    assert len(result['scene_strategies'])==6
    assert all(r['fragments_per_performer']=={'A':1} for r in result['scene_strategies'].values())
    assert all(set(w['damage'])=={'baseline','winner'} for w in result['omissions'])


def test_explicit_physical_vote_can_repair_unlinked_baseline(tmp_path):
    config = configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        item,ids=scene(db,config)
        db.execute('UPDATE scans SET namespace=?',(namespace(config),))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('face',item['scan'],0,0.,'','[]','[]','{}',None,2,np.array([1.,0.],dtype='<f4').tobytes(),None))
        pairs=explicit_pairs(db,config,'scope',['face:'+ids[0]])
        unit=dict(id='unit',observations=[dict(id='face',scene='1')])
        evidence=dict(core_units=[unit],soft_units=[],scenes={})
        member=dict(observations=[dict(id='face',scene='1')],identity_evidence=evidence)
        goals=dict(groups=[dict(members=[member],identity_evidence=evidence)])
        freeze(db,config,goals,pairs)
        assert len(evidence['core_units'])==1
        assert [r['id'] for r in unit['observations']]==['face',ids[0]]
