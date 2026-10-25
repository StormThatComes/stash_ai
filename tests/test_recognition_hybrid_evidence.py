import numpy as np
from types import SimpleNamespace
from stash_ai.recognition.hybrid_evidence import Identity,pair,plan
from stash_ai.recognition.config import defaults


def identity(name,angle=0,missing=()):
    face=np.array([[np.cos(angle),np.sin(angle)]],dtype='float32')
    return Identity(name,(name,),face,np.array([[1.,0.]]),((name,0),(name,1),(name,2)),missing)


def test_separate_encoders_and_unknown_evidence():
    settings=defaults()['sapiensid']['matching']
    assert not pair(identity('a'),identity('b',.8),settings)['blockers']
    assert 'face_contradiction' in pair(identity('a'),identity('b',2),settings)['blockers']
    assert 'missing_person_evidence' in pair(identity('a'),identity('b',missing=('b',)),settings)['blockers']
    assert 'copresence_conflict' in pair(identity('a'),identity('b',.8),settings,copresent=True,reflection_threshold=.9)['blockers']


def test_complete_component_and_rival(monkeypatch):
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    cfg=SimpleNamespace(data=defaults())
    # A-B and B-C are allowed, but A-C conflicts: never chain through B.
    result=plan(None,cfg,[identity('a'),identity('b',.8),identity('c',1.6)])
    assert result['decisions'][0]['blockers']==['incomplete_component']
    # Missing person evidence cannot join, but remains a face-compatible rival.
    c=Identity('c',('c',),np.array([[1.,0.]]),np.empty((0,2)),(),('c',))
    result=plan(None,cfg,[identity('a'),identity('b',.8),c])
    ab=next(d for d in result['decisions'] if len(d['people'])==2)
    assert ab['blockers']==['competing_identity']


def test_independent_votes_not_raw_sample_count():
    a=identity('a');b=identity('b')
    b=Identity(b.person,b.observations,b.faces,b.persons,(('scene',1),))
    assert 'insufficient_independent_support' in pair(a,b,defaults()['sapiensid']['matching'])['blockers']
