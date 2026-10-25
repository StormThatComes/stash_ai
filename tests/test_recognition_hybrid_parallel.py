from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.config import defaults
from stash_ai.recognition.hybrid_evidence import plan
from stash_ai.recognition.hybrid_lookup import Lookup
from tests.test_recognition_hybrid_evidence import identity


def test_block_filter_matches_exhaustive_and_keeps_incident_rivals():
    bank=[identity(str(i),i*.1) for i in range(140)]
    lookup=Lookup(bank,10**7)
    full=list(lookup.pairs())
    filtered=list(lookup.pairs(face_floor=.9,person_floor=2))
    assert filtered==[p for p in full if p[2]+1e-6>=.9]
    assert list(lookup.pairs(incident={'1'}))==[p for p in full if '1' in p[:2]]
    assert list(lookup.pairs(incident=set()))==[]


def test_parallel_proofs_match_serial_complete_plan(monkeypatch):
    from stash_ai.recognition.compute import compute_scope as scope
    from stash_ai.recognition.run_cache import scope as cache_scope
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    config=SimpleNamespace(data=defaults());bank=[identity(str(i),i*.1) for i in range(12)]
    serial=plan(None,config,bank)
    config.data['execution']['rebuild_workers']=3
    with cache_scope(config),scope(config):parallel=plan(None,config,bank)
    assert parallel==serial
