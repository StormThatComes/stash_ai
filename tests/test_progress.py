"""Resume equivalence and bounded preparation; never connects to Stash/HF."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import threading

import numpy as np
import pytest

from stash_ai import cli as app
from stash_ai import merge
from stash_ai import progress
from stash_ai.taggers import ModelSettings, Vocabulary, WD, SOURCES
from tests.helpers.stash import Stash


from tests.helpers.inference import IndexedTagger
pytestmark = pytest.mark.usefixtures("reset_progress")


def test_resume_reuses_completed_pairs_without_loading_weights_or_rewriting_stash(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    p.stash.writes.clear()
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("completed model loaded"))
    assert app.main(p.args)==0
    assert not p.stash.writes


@pytest.mark.parametrize("aggregation,top_k",[("max",2),("topk_mean",2),("topk_mean",5)])
@pytest.mark.parametrize("prefetch",[0,1])
def test_interruption_resume_matches_full_inference_and_counts_each_frame_once(pipeline,monkeypatch,aggregation,top_k,prefetch):
    p=pipeline
    monkeypatch.setattr(app,"AGGREGATION",aggregation);monkeypatch.setattr(app,"TOP_K_FRAMES",top_k)
    p.fail["wd"]=4
    with pytest.raises(KeyboardInterrupt): app.main([*p.args,"--prefetch-batches",str(prefetch)])
    assert p.loaded[0].seen==[0,1,2,3] and p.loaded[0].closed
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert state.records("1","scene")=={}  # A partial model result is not published.
        assert state.connection.execute("SELECT processed FROM " + state.table('checkpoints')).fetchone()==(4,)
    p.fail.clear()
    assert app.main([*p.args,"--prefetch-batches",str(prefetch)])==0
    assert p.loaded[-1].seen==list(range(4,10))
    expected=IndexedTagger(p.loaded[-1].settings,p.scores["wd"],p.vocab.labels)
    predictions,frames=app.predict_scene(expected,p.stash,{},p.cues,raw_scene=True)
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        scene=state.records("1","scene")["wd"][0]
        markers=state.records("1","markers")["wd"][0]
        assert [(item["tag"],item["score"]) for item in scene["items"]]==predictions
        assert [(item["start"],item["tags"]) for item in markers["items"]]==[(f.cue.start,[list(t) for t in f.tags]) for f in frames]
        assert state.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]==0
        assert state.connection.execute("SELECT COUNT(*) FROM checkpoint_frames").fetchone()[0]==0


def test_restart_keeps_refresh_requests_for_unfinished_models(pipeline):
    p=pipeline;args=["--models","all",*p.args[2:]]
    assert app.main(args)==0
    p.fail["wd"]=4
    with pytest.raises(KeyboardInterrupt): app.main([*args,"--restart"])
    completed_count=len(p.loaded)
    p.fail.clear()
    assert app.main(args)==0
    resumed=p.loaded[completed_count:]
    assert [m.settings.source.key for m in resumed]==["wd","dtq","oppai"]
    assert resumed[0].seen==list(range(4,10))
    assert all(m.seen==list(range(10)) for m in resumed[1:])


def test_failed_restart_never_publishes_fresh_and_stale_model_mix(pipeline,monkeypatch):
    p=pipeline;args=["--models","wd","dtq",*p.args[2:]]
    assert app.main(args)==0
    before=deepcopy(p.stash.scene)
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        snapshots={kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}

    p.scores["wd"][:]=0;p.scores["dtq"][:]=0
    fail={"dtq":True}
    def load(settings,*_):
        assert all(model.closed for model in p.loaded)
        broken=fail.get(settings.source.key,False)
        model=IndexedTagger(settings,p.scores[settings.source.key],p.vocab.labels,
                            fail_at=0 if broken else None,error=RuntimeError)
        p.loaded.append(model)
        return model
    monkeypatch.setattr(app,"load_tagger",load)

    p.stash.writes.clear()
    assert app.main([*args,"--restart"])==1
    assert p.stash.scene==before and not p.stash.writes
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={
            "1":[("dtq","markers"),("dtq","scene")]}
        assert {kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}==snapshots
        assert state.records("1","scene")["wd"][0]["items"]==[]

    fail["dtq"]=False
    assert app.main(args)==0
    assert not p.stash.scene["tags"] and not p.stash.scene["scene_markers"]
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={}


def test_automatic_input_invalidation_never_publishes_fresh_and_stale_model_mix(pipeline,monkeypatch):
    p=pipeline;args=["--models","wd","dtq",*p.args[2:]]
    assert app.main(args)==0
    before=deepcopy(p.stash.scene)
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        snapshots={kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}

    # Same media URLs, but changed cue timing invalidates both models' inference signatures.
    p.cues=[app.Cue(c.start,c.end+.1,c.image,c.rect) for c in p.cues]
    p.scores["wd"][:]=0;p.scores["dtq"][:]=0
    fail={"dtq":True}
    def load(settings,*_):
        assert all(model.closed for model in p.loaded)
        broken=fail.get(settings.source.key,False)
        model=IndexedTagger(settings,p.scores[settings.source.key],p.vocab.labels,
                            fail_at=0 if broken else None,error=RuntimeError)
        p.loaded.append(model)
        return model
    monkeypatch.setattr(app,"load_tagger",load)

    p.stash.writes.clear()
    assert app.main(args)==1
    assert p.stash.scene==before and not p.stash.writes
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={
            "1":[("dtq","markers"),("dtq","scene")]}
        assert {kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}==snapshots
        assert state.records("1","scene")["wd"][0]["items"]==[]

    fail["dtq"]=False
    assert app.main(args)==0
    assert not p.stash.scene["tags"] and not p.stash.scene["scene_markers"]
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={}


def test_source_content_change_invalidates_checkpoint_even_with_same_url(pipeline,monkeypatch):
    p=pipeline;p.fail["wd"]=4
    with pytest.raises(KeyboardInterrupt): app.main(p.args)
    p.fail.clear();monkeypatch.setattr(p.stash,"get_media",lambda url:b"regenerated same URL")
    assert app.main(p.args)==0
    assert p.loaded[-1].seen==list(range(10))


def test_vtt_timing_change_invalidates_but_scene_threshold_reuses_frames(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    p.cues=[app.Cue(c.start,c.end+.1,c.image,c.rect) for c in p.cues]
    assert app.main(p.args)==0 and len(p.loaded)==2
    original=app.load_model_settings
    from dataclasses import replace
    monkeypatch.setattr(app,"load_model_settings",lambda *args:[replace(c,scene_threshold=.77) for c in original(*args)])
    assert app.main(p.args)==0 and len(p.loaded)==2


def test_policy_change_reuses_predictions_and_updates_outputs(pipeline,monkeypatch,tmp_path):
    p=pipeline
    assert app.main(p.args)==0
    policy=tmp_path/"scene.yml";policy.write_text("version: 1\nexcluded_scene_tags: [common]\n")
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("blacklist change reran inference"))
    assert app.main([*p.args,"--scene-tag-exclusions",str(policy)])==0
    assert "ai:common" not in {t["name"] for t in p.stash.scene["tags"]}
    assert not any(m["primary_tag"]["name"]=="ai:common" for m in p.stash.scene["scene_markers"])
    assert all(t['name']!='ai:common' for m in p.stash.scene['scene_markers'] for t in m['tags'])


def test_partial_output_can_reuse_a_completed_full_run(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("partial output reran inference"))
    before=deepcopy(p.stash.scene["tags"])
    assert app.main([*p.args,"--markers-only"])==0
    assert p.stash.scene["tags"]==before


def test_status_lookup_does_not_decompress_results(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        monkeypatch.setattr(merge,"decode",lambda *args:pytest.fail("status decompressed predictions"))
        assert merge.has_outputs(app,state,"1","wd")
        assert state.sources("1","markers")=={"wd"}


def test_dry_restart_does_not_change_results_checkpoint_or_refresh_requests(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    before=(p.path.read_bytes(),deepcopy(p.stash.scene))
    p.stash.dry_run=True
    assert app.main([*p.args,"--restart","--dry-run"])==0
    assert (p.path.read_bytes(),p.stash.scene)==before


def test_zero_predictions_are_successfully_completed(pipeline,monkeypatch):
    p=pipeline;p.scores["wd"][:]=0
    assert app.main(p.args)==0
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("empty successful result repeated"))
    assert app.main(p.args)==0


def test_api_failure_resumes_only_synchronization(pipeline,monkeypatch):
    p=pipeline;p.stash.fail_create=True
    assert app.main(p.args)==1
    p.stash.fail_create=False
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("API failure repeated inference"))
    assert app.main(p.args)==0
    assert p.stash.scene["scene_markers"]


def test_verified_media_ram_limit_and_input_consistency():
    stash=SimpleNamespace(get_media=lambda url:(url*8).encode())
    media=progress.VerifiedMedia(stash,limit=16)
    for url in ["a","b","c","d"]:media.get_media(url)
    assert media.size<=16 and len(media.cache)<=2
    stash.get_media=lambda url:b"changed"
    with pytest.raises(RuntimeError,match="changed"):media.get_media("a")


def test_prefetch_is_bounded_and_closes_producer(pipeline,monkeypatch):
    p=pipeline;config=ModelSettings(WD,Path("missing"),batch_size=2)
    model=IndexedTagger(config,p.scores["wd"],p.vocab.labels)
    prepared=threading.Event()
    original=model.prepare
    def prepare(image):
        out=original(image)
        if image==3: prepared.set()
        return out
    model.prepare=prepare
    iterator=progress.prepared_batches(app,model,p.stash,{},p.cues,True)
    batch,_=next(iterator)
    assert batch==[0,1]
    assert prepared.wait(2)
    assert model.prepared==[0,1,2,3]
    iterator.close()
    assert not any(t.name.startswith("prepare-frames") for t in threading.enumerate())


def test_v1_database_is_migrated_once_without_losing_sources(tmp_path,monkeypatch):
    path=tmp_path/"v1.sqlite3"
    segment={"mode":"scene","meta":{"origin":"inference"},"items":[{"tag":"common","score":.9}]}
    db=sqlite3.connect(path)
    db.executescript(f"PRAGMA application_id={merge.APPLICATION_ID}; PRAGMA user_version=1; "
                    "CREATE TABLE results(scope TEXT,scene TEXT,model TEXT,kind TEXT,data BLOB,PRIMARY KEY(scope,scene,model,kind));")
    db.execute("INSERT INTO results VALUES (?,?,?,?,?)",("fixture","1","wd","scene",merge.encode([segment])))
    db.commit();db.close()
    with merge.StateStore(path,"fixture") as state:
        assert state.connection.execute("PRAGMA user_version").fetchone()==(5,)
        assert state.records("1","scene")["wd"]==[segment]
        assert progress.completed(state,"1","wd",["scene"])
        assert not progress.completed(state,"1","wd",["scene"],"new-signature")
    monkeypatch.setattr(progress,"decode",lambda *args:pytest.fail("migration repeated"))
    with merge.StateStore(path,"fixture") as state:assert progress.completed(state,"1","wd",["scene"])


@pytest.mark.parametrize("failure",[False,True])
def test_pruning_follows_only_successful_merge(pipeline,monkeypatch,failure):
    from stash_ai import prune as stash_prune
    p=pipeline;p.stash.fail_create=failure
    calls=[]
    monkeypatch.setattr(stash_prune,"prune_tags",lambda *args,**kwargs:calls.append(args) or {})
    assert app.main([*p.args,"--prune-empty-tags"])==int(failure)
    assert len(calls)==2*int(not failure)  # Generic definitions and derived marker groups.
    p.stash.fail_create=False
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("merge-only loaded weights"))
    assert app.main([*p.args,"--merge-only","--prune-empty-tags"])==0
    assert len(calls)==2*int(not failure)+2


def test_ordered_vocabulary_changes_checkpoint_identity(pipeline):
    p=pipeline
    config=ModelSettings(WD,Path("missing"))
    original=progress.model_identity(app,config,p.vocab)
    reordered=Vocabulary(list(reversed(p.vocab.labels)),p.vocab.categories)
    assert progress.model_identity(app,config,reordered)!=original


@pytest.mark.parametrize("flag",["--resume","--restart"])
def test_resume_flags_are_explicitly_rejected_for_legacy_separate_mode(pipeline,flag):
    with pytest.raises(SystemExit) as exc:app.main([*pipeline.args,"--output-mode","separate",flag])
    assert exc.value.code==2 and not pipeline.loaded
