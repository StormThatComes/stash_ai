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
    assert len(result['scene_strategies'])==9
    assert all(r['fragments_per_performer']=={'A':1} for r in result['scene_strategies'].values())
    assert all(set(w['damage'])=={'baseline','winner',*result['stress_strategies']} for w in result['omissions'])


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


def test_unlabelled_face_preserves_a_turn_and_ordinary_false_joins_regress():
    from stash_ai.recognition.solver_validation import _metrics, holdout_verdict
    from stash_ai.recognition.reference_roles import STRICT_REASONS
    labels = {'a0':'A','a1':'A','a2':'A','face1':'A','b0':'B','b1':'B','b2':'B'}
    locations = {k:('1',int(k[-1])) for k in labels}
    samples = {k:dict(slices=['back'] if k in ('a1','face1') else []) for k in labels}
    def measure(join):
        decisions = {k:dict(accepted_identity=None if k=='face1' else 'A' if join else owner,
                            evidence_kind='person' if k=='a1' else 'face',
                            evidence=dict(reason=sorted(STRICT_REASONS)[0])) for k,owner in labels.items()}
        return dict(_metrics(labels,samples,locations,decisions),visible={})
    good,bad = measure(False),measure(True)
    assert good['sequences']['own_id_turns']==1
    verdict = holdout_verdict({'baseline/joint':good,'winner/joint':bad})
    assert verdict['status']=='regressed'
    assert verdict['measures']['physical_false_merge_pairs']['winner']==9
