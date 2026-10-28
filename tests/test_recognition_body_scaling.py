from copy import deepcopy
import math

import numpy as np
import pytest

from stash_ai.recognition.body_global import banks,report
from stash_ai.recognition.body_identity import rebuild,policy,vector
from stash_ai.recognition.database import FaceDatabase,encoded
from tests.test_recognition_body_identity import configuration,scene
from tests.helpers.body_global_reference import exhaustive


def fixture(tmp_path,centers,samples=3):
    c=configuration(tmp_path);c.data['sapiensid']['identities']['min_support']=2
    state=FaceDatabase(c.database);state.__enter__();db=state.connection
    selected=[]
    for i,v in enumerate(centers):
        item,_=scene(db,c,str(i+1),vectors=[v]*samples);selected.append(item)
    db.commit();rebuild(state,selected,c,{s['scan']:[] for s in selected})
    c.data['sapiensid']['global_matching']['enabled']=True
    db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
    return c,state


def hydrated(db,bank):
    result=deepcopy(bank)
    for b in result.values():
        for core in b['cores']:
            for row in core:row['_vector']=vector(db.execute('SELECT embedding FROM body_observations WHERE id=?',(row['id'],)).fetchone())
        b['faces']=[np.frombuffer(db.execute('SELECT embedding FROM observations WHERE id=?',(r['id'],)).fetchone()[0],dtype='<f4') for r in b['faces']]
    return result


@pytest.mark.parametrize('seed',range(8))
def test_sparse_planner_matches_exhaustive_decisions_and_pair_proofs(tmp_path,seed):
    rng=np.random.default_rng(seed)
    angles=rng.uniform(-1,1,12)
    c,state=fixture(tmp_path,[[math.cos(a),math.sin(a)] for a in angles])
    try:
        db=state.connection
        # Vary gates and retained guards independently of centroid nomination.
        c.data['sapiensid']['global_matching'].update(match_threshold=.85,cohesion_threshold=.8,ambiguity_margin=seed*.02)
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        bank=banks(db,c,'scope','ns');people=sorted(bank)
        bank[people[0]]['support']=1
        bank[people[1]]['missing_original']=True
        bank[people[2]]['cues'].update(bank[people[3]]['cues'])
        expected=exhaustive(hydrated(db,bank),c)
        from unittest.mock import patch
        with patch('stash_ai.recognition.body_global.banks',return_value=bank):
            actual=report(db,c,'scope','ns')
        assert actual['decisions']==expected['decisions']
        expected_pairs={tuple(p['people']):p for p in expected['pairs']}
        assert all(p==expected_pairs[tuple(p['people'])] for p in actual['pairs'])
        assert actual['search']['candidate_pairs']<=len(expected_pairs)
    finally:state.__exit__(None,None,None)


def test_32_scenes_no_longer_spend_budget_on_rejected_cross_products(tmp_path):
    c,state=fixture(tmp_path,np.eye(32).tolist(),samples=16)
    try:
        c.data['sapiensid']['execution'].update(max_candidate_pairs=1,max_witness_comparisons=1)
        result=report(state.connection,c,'scope','ns')
        assert result['search']['identity_pairs']==496
        assert result['search']['candidate_pairs']==0
        assert result['search']['body_witness_comparisons']==0
        assert result['pairs']==result['decisions']==[]
        assert result['search']['embedding_reads']==512
    finally:state.__exit__(None,None,None)


def test_duplicate_core_checks_share_scores_but_keep_proof_multiplicity(tmp_path):
    c,state=fixture(tmp_path,[[1,0],[1,0]],samples=16)
    try:
        db=state.connection;bank=banks(db,c,'scope','ns')
        for b in bank.values():b['cores'].extend([b['refs'],b['refs']])
        from unittest.mock import patch
        with patch('stash_ai.recognition.body_global.banks',return_value=bank):
            result=report(db,c,'scope','ns')
        assert result['search']['body_witness_comparisons']==16*16
        assert result['search']['score_cache_hits']>=9
        assert all(len(ids)==48 for ids in result['decisions'][0]['cores'].values())
    finally:state.__exit__(None,None,None)


def test_requested_subthreshold_pairs_and_all_goal_rivals_are_exact(tmp_path):
    c,state=fixture(tmp_path,[[math.cos(a),math.sin(a)] for a in (0,.52,1.0,2.7)])
    try:
        db=state.connection;bank=banks(db,c,'scope','ns');people=sorted(bank)
        focus={people[0]:0,people[1]:0,people[2]:1}
        actual=report(db,c,'scope','ns',focus=focus)
        expected=exhaustive(hydrated(db,bank),c)
        pairs={tuple(p['people']):p for p in expected['pairs']}
        assert all(pairs[tuple(p['people'])]==p for p in actual['pairs'])
        assert all(tuple(sorted((a,b))) in {tuple(p['people']) for p in actual['pairs']}
                   for a in focus for b in focus if a!=b)
        for rival in actual['nearest_competitors']:
            person=rival['identity']
            allowed=[p for p in expected['pairs'] if person in p['people'] and any(
                other not in focus or focus[other]!=focus[person] for other in p['people'])]
            assert rival['score']==max(p['score'] for p in allowed)
    finally:state.__exit__(None,None,None)


def test_cross_tile_candidates_rivals_and_small_working_budget_match_oracle(tmp_path):
    c,state=fixture(tmp_path,np.eye(131).tolist())
    try:
        db=state.connection;bank=banks(db,c,'scope','ns');people=sorted(bank)
        source=db.execute('SELECT embedding FROM body_observations WHERE id=?',(bank[people[0]]['refs'][0]['id'],)).fetchone()[0]
        for row in bank[people[-1]]['refs']:
            db.execute('UPDATE body_observations SET embedding=? WHERE id=?',(source,row['id']))
        db.commit()
        c.data['sapiensid']['execution']['reference_bytes']=8*1024**2
        expected=exhaustive(hydrated(db,bank),c)
        actual=report(db,c,'scope','ns',focus={people[13]:0,people[65]:0})
        assert actual['decisions']==expected['decisions']
        assert actual['search']['reference_working_bytes']<=8*1024**2
        pairs={tuple(p['people']):p for p in expected['pairs']}
        assert all(p==pairs[tuple(p['people'])] for p in actual['pairs'])
    finally:state.__exit__(None,None,None)


def test_reference_cap_applies_before_embedding_reads_and_keeps_full_support(tmp_path):
    c,state=fixture(tmp_path,np.eye(12).tolist(),samples=40)
    try:
        db=state.connection
        c.data['sapiensid']['global_matching']['max_references']=3
        db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
        bank=banks(db,c,'scope','ns')
        assert all(len(b['refs'])==3 and b['support']==40 for b in bank.values())
        assert all('embedding' not in r and '_vector' not in r for b in bank.values() for r in b['refs'])
        result=report(db,c,'scope','ns')
        assert result['search']['embedding_reads']==36
    finally:state.__exit__(None,None,None)


def test_original_core_veto_survives_nomination_and_score_reuse(tmp_path):
    c,state=fixture(tmp_path,[[1,0],[1,0],[-1,0]])
    try:
        from unittest.mock import patch
        db=state.connection;bank=banks(db,c,'scope','ns')
        directions={p:float(vector(db.execute('SELECT embedding FROM body_observations WHERE id=?',
            (b['refs'][0]['id'],)).fetchone())[0]) for p,b in bank.items()}
        positive=sorted(p for p,d in directions.items() if d>0);negative=next(p for p,d in directions.items() if d<0)
        bank[positive[0]]['cores'].append(bank[negative]['refs'])
        expected=exhaustive(hydrated(db,bank),c)
        with patch('stash_ai.recognition.body_global.banks',return_value=bank):actual=report(db,c,'scope','ns')
        assert actual['decisions']==expected['decisions']==[]
        assert actual['pairs'][0]['people']==positive
        assert 'body_cohesion' in actual['pairs'][0]['blockers']
        assert actual['pairs'][0]['minimum']==-1
    finally:state.__exit__(None,None,None)


def test_near_threshold_centroid_rounding_does_not_change_acceptance(tmp_path):
    c,state=fixture(tmp_path,[[1,0],[.9,math.sqrt(1-.9**2)]])
    try:
        db=state.connection;bank=banks(db,c,'scope','ns')
        exact=exhaustive(hydrated(db,bank),c)['pairs'][0]['score']
        for threshold in (math.nextafter(exact,0),exact,math.nextafter(exact,1)):
            c.data['sapiensid']['global_matching'].update(match_threshold=threshold,cohesion_threshold=.8,ambiguity_margin=0)
            db.execute('UPDATE body_members SET policy=?',(policy(c),));db.commit()
            actual=report(db,c,'scope','ns');expected=exhaustive(hydrated(db,bank),c)
            assert actual['pairs']==expected['pairs']
            assert actual['decisions']==expected['decisions']
    finally:state.__exit__(None,None,None)


@pytest.mark.parametrize('key,limit,match',[
    ('max_candidate_pairs',1,'candidate pairs'),('max_witness_comparisons',1,'witness comparisons'),
    ('reference_bytes',1,'reference metadata'),('plan_bytes',1,'body plan')])
def test_limits_fail_explicitly_without_source_changes_or_scratch_leaks(tmp_path,monkeypatch,key,limit,match):
    import tempfile
    c,state=fixture(tmp_path,[[1,0]]*3)
    try:
        state.connection.commit();before=c.database.read_bytes()
        monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
        c.data['sapiensid']['execution'][key]=limit
        with pytest.raises(ValueError,match=match):report(state.connection,c,'scope','ns')
        assert not list(tmp_path.glob('stash-body-means-*'))
        assert c.database.read_bytes()==before
    finally:state.__exit__(None,None,None)
