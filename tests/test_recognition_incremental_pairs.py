from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.derived_cache import scope
from stash_ai.recognition.config import defaults
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.hybrid_evidence import plan
from tests.test_recognition_hybrid_evidence import identity


def test_new_competitor_invalidates_old_merge_without_rescoring_its_proof(tmp_path,monkeypatch):
    from stash_ai.recognition import hybrid_jobs
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    config=SimpleNamespace(data=defaults())
    original=hybrid_jobs.exact
    calls=[]
    def measured(payload):
        calls.append(tuple(item.person for item in payload[:2]))
        return original(payload)
    monkeypatch.setattr(hybrid_jobs,'exact',measured)
    pair=[identity('a'),identity('b',.1)]
    path=tmp_path/'faces.sqlite'
    with FaceDatabase(path) as db,scope(db,config):
        result=plan(None,config,pair,compact=True)
        assert any(d['outcome']=='merge' for d in result['decisions'])
        db.connection.commit()
    rival=replace(identity('c'),persons=np.empty((0,2)),independent=())
    calls.clear()
    with FaceDatabase(path) as db,scope(db,config):
        result=plan(None,config,pair+[rival],compact=True)
        assert ('a','b') not in calls
        decision=next(d for d in result['decisions'] if d['people']==['a','b'])
        assert decision['blockers']==['competing_identity']
    assert result['decisions']==plan(None,config,pair+[rival])['decisions']

