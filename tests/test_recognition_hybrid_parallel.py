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


def test_only_changed_endpoint_proofs_recompute_but_rivals_remain_live(monkeypatch):
    from dataclasses import replace
    from stash_ai.recognition import hybrid_jobs
    from stash_ai.recognition.run_cache import scope
    monkeypatch.setattr('stash_ai.recognition.global_views._cooccurs',lambda *a:False)
    calls=[];original=hybrid_jobs.exact
    def counted(payload):calls.append(1);return original(payload)
    monkeypatch.setattr(hybrid_jobs,'exact',counted)
    config=SimpleNamespace(data=defaults());bank=[identity(str(i),0.) for i in range(4)]
    with scope(config):
        first=plan(None,config,bank);count=len(calls)
        assert count==6
        assert plan(None,config,bank)==first and len(calls)==count
        bank[0]=replace(bank[0],guard_missing=('lost-original',))
        changed=plan(None,config,bank)
        assert len(calls)==count+3 and changed!=first


def test_parallel_nomination_tiles_preserve_all_rivals_and_cache(monkeypatch):
    from stash_ai.recognition.compute import compute_scope
    from stash_ai.recognition.run_cache import scope
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus', lambda: 16)
    config = SimpleNamespace(data=defaults())
    config.data['execution']['rebuild_workers'] = 4
    bank = [identity(str(i), i*.037) for i in range(145)]
    expected = list(Lookup(bank, 10**7).pairs(incident={'1', '99'}, face_floor=.85))
    with scope(config), compute_scope(config) as pool:
        lookup = Lookup(bank, 10**7)
        assert list(lookup.pairs(incident={'1', '99'}, face_floor=.85)) == expected
        assert list(lookup.pairs(incident={'1', '99'}, face_floor=.85)) == expected
        assert pool.records['hybrid_nomination']['computed'] > 2
