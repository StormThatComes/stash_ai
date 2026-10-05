"""Resume equivalence and bounded preparation; never connects to Stash/HF."""
from tests.helpers.evidence import current_record


from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import threading

import numpy as np
import pytest

from stash_ai.inference.identity import model_identity
from stash_ai.domain import Cue, FramePrediction
from stash_ai.inference import checkpoints, prediction, preparation
from stash_ai.storage import codec as state_codec
from tests.helpers.runtime import runtime as app
import stash_ai.inference.results as merge_results
import stash_ai.storage.store as merge_store
from stash_ai.storage.database import APPLICATION_ID
from stash_ai.storage import progress
from stash_ai.models import ModelSettings, Vocabulary, WD, SOURCES
from tests.helpers.stash import Stash


from tests.helpers.inference import IndexedTagger
pytestmark = pytest.mark.usefixtures("reset_progress")


def test_resume_reuses_completed_pairs_without_loading_weights_or_rewriting_stash(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    p.stash.writes.clear()
    monkeypatch.setattr(app.services,"load_tagger",lambda *args:pytest.fail("completed model loaded"))
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
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        assert state.records("1","scene")=={}  # A partial model result is not published.
        assert state.connection.execute("SELECT processed FROM " + state.table('checkpoints')).fetchone()==(4,)
    p.fail.clear()
    assert app.main([*p.args,"--prefetch-batches",str(prefetch)])==0
    assert p.loaded[-1].seen==list(range(4,10))
    expected=IndexedTagger(p.loaded[-1].settings,p.scores["wd"],p.vocab.labels)
    predictions,frames=prediction.predict_scene(app, expected,p.stash,{},p.cues)
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        scene=state.records("1","scene")["wd"]
        markers=state.records("1","markers")["wd"]
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
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
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
    monkeypatch.setattr(app.services,"load_tagger",load)

    p.stash.writes.clear()
    assert app.main([*args,"--restart"])==1
    assert p.stash.scene==before and not p.stash.writes
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={
            "1":[("dtq","markers"),("dtq","scene")]}
        assert {kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}==snapshots
        assert state.records("1","scene")["wd"]["items"]==[]

    fail["dtq"]=False
    assert app.main(args)==0
    assert not p.stash.scene["tags"] and not p.stash.scene["scene_markers"]
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={}


def test_automatic_input_invalidation_never_publishes_fresh_and_stale_model_mix(pipeline,monkeypatch):
    p=pipeline;args=["--models","wd","dtq",*p.args[2:]]
    assert app.main(args)==0
    before=deepcopy(p.stash.scene)
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        snapshots={kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}

    # Same media URLs, but changed cue timing invalidates both models' inference signatures.
    p.cues=[Cue(c.start,c.end+.1,c.image,c.rect) for c in p.cues]
    p.scores["wd"][:]=0;p.scores["dtq"][:]=0
    fail={"dtq":True}
    def load(settings,*_):
        assert all(model.closed for model in p.loaded)
        broken=fail.get(settings.source.key,False)
        model=IndexedTagger(settings,p.scores[settings.source.key],p.vocab.labels,
                            fail_at=0 if broken else None,error=RuntimeError)
        p.loaded.append(model)
        return model
    monkeypatch.setattr(app.services,"load_tagger",load)

    p.stash.writes.clear()
    assert app.main(args)==1
    assert p.stash.scene==before and not p.stash.writes
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={
            "1":[("dtq","markers"),("dtq","scene")]}
        assert {kind:state.snapshot("1",kind)[0] for kind in ("scene","markers")}==snapshots
        assert state.records("1","scene")["wd"]["items"]==[]

    fail["dtq"]=False
    assert app.main(args)==0
    assert not p.stash.scene["tags"] and not p.stash.scene["scene_markers"]
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        assert progress.pending_refreshes(state,["scene","markers"])=={}


def test_source_content_change_invalidates_checkpoint_even_with_same_url(pipeline,monkeypatch):
    p=pipeline;p.fail["wd"]=4
    with pytest.raises(KeyboardInterrupt): app.main(p.args)
    p.fail.clear();monkeypatch.setattr(p.stash,"get_media",lambda url:b"regenerated same URL")
    assert app.main(p.args)==0
    assert p.loaded[-1].seen==list(range(10))


def test_analysis_version_restarts_only_partial_model_and_preserves_completed_results(pipeline, monkeypatch):
    from stash_ai.evidence.temporal import statistics
    p = pipeline
    args = ['--models', 'dtq', 'wd', *p.args[2:]]
    p.fail['wd'] = 4
    with monkeypatch.context() as old:
        old.setattr(statistics, 'CHECKPOINT_FORMAT', 0)
        with pytest.raises(KeyboardInterrupt):
            app.main(args)
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        completed = state.connection.execute("SELECT kind,data FROM results WHERE model='dtq' ORDER BY kind").fetchall()
        assert completed
        blob, = state.connection.execute('SELECT data FROM checkpoint_analysis').fetchone()
        analysis = state_codec.decode(blob)
        analysis.pop('format')
        for item in analysis['items'].values():
            item.pop('peak_time')
        with state.connection:
            state.connection.execute('UPDATE checkpoint_analysis SET data=?', (state_codec.encode(analysis),))
    loaded = len(p.loaded)
    p.fail.clear()
    assert app.main(args) == 0
    assert len(p.loaded) == loaded + 1 and p.loaded[-1].seen == list(range(10))
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        assert state.connection.execute("SELECT kind,data FROM " + state.table('results') + " WHERE model='dtq' ORDER BY kind").fetchall() == completed
        for table in ('checkpoints', 'checkpoint_frames', 'checkpoint_analysis'):
            assert state.connection.execute('SELECT COUNT(*) FROM ' + table).fetchone() == (0,)


def test_vtt_timing_change_invalidates_but_scene_threshold_reuses_frames(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    p.cues=[Cue(c.start,c.end+.1,c.image,c.rect) for c in p.cues]
    assert app.main(p.args)==0 and len(p.loaded)==2
    original=app.services.load_model_settings
    from dataclasses import replace
    monkeypatch.setattr(app.services,"load_model_settings",lambda *args:[replace(c,scene_threshold=.77) for c in original(*args)])
    assert app.main(p.args)==0 and len(p.loaded)==2


def test_policy_change_reuses_predictions_and_updates_outputs(pipeline,monkeypatch,tmp_path):
    p=pipeline
    assert app.main(p.args)==0
    policy=tmp_path/"scene.yml";policy.write_text("version: 1\nexcluded_scene_tags: [common]\n")
    monkeypatch.setattr(app.services,"load_tagger",lambda *args:pytest.fail("blacklist change reran inference"))
    assert app.main([*p.args,"--scene-tag-exclusions",str(policy)])==0
    assert "ai:common" not in {t["name"] for t in p.stash.scene["tags"]}
    assert not any(m["primary_tag"]["name"]=="ai:common" for m in p.stash.scene["scene_markers"])
    assert all(t['name']!='ai:common' for m in p.stash.scene['scene_markers'] for t in m['tags'])


def test_partial_output_can_reuse_a_completed_full_run(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    monkeypatch.setattr(app.services,"load_tagger",lambda *args:pytest.fail("partial output reran inference"))
    before=deepcopy(p.stash.scene["tags"])
    assert app.main([*p.args,"--markers-only"])==0
    assert p.stash.scene["tags"]==before


def test_status_lookup_does_not_decompress_results(pipeline,monkeypatch):
    p=pipeline
    assert app.main(p.args)==0
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        monkeypatch.setattr(state,"unpack",lambda *args:pytest.fail("status decompressed predictions"))
        signature = state.connection.execute(
            "SELECT signature FROM " + state.table('result_index') + " WHERE scope=? AND scene='1' AND model='wd' AND kind='scene'",
            (state.scope,)).fetchone()[0]
        assert progress.completed(state,"1","wd",['scene','markers'],signature)
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
    monkeypatch.setattr(app.services,"load_tagger",lambda *args:pytest.fail("empty successful result repeated"))
    assert app.main(p.args)==0


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('cache_bytes', [0, 2**20])
def test_finish_rolls_back_results_checkpoint_and_refresh_together(tmp_path, monkeypatch, dry_run, cache_bytes):
    path = tmp_path / 'state.sqlite3'
    meta = dict(origin='inference', inference_signature='old')
    original = {
        'scene': current_record(dict(mode='scene', meta=meta, items=[dict(tag='core', score=.8)])),
        'markers': current_record(dict(mode='frames', meta=meta, items=[dict(start=0., end=1., tags=[['core', .8]])])),
    }
    fresh = deepcopy(original)
    for segment in fresh.values():
        segment['meta']['inference_signature'] = 'new'
        segment['items'] = []
    with merge_store.StateStore(path, 'fixture') as state:
        state.put('1', 'wd', original)
        checkpoint = checkpoints.Checkpoint(state, '1', 'wd', 'new', 2, 1, every=1)
        checkpoint.record(1, np.array([[.8]], dtype=np.float32),
                          [FramePrediction(Cue(0., 1., '', (0, 0, 1, 1)), (('core', .8),))],
                          analysis={'scene': {'frames': 1}})
        progress.request_refresh(state, ['1'], ['wd'], ['scene', 'markers'])

    tables = ('results', 'result_metadata', 'result_thresholds', 'result_index',
              'checkpoints', 'checkpoint_frames', 'checkpoint_analysis', 'refresh_requests')
    with merge_store.StateStore(path, 'fixture', dry_run, cache_bytes=cache_bytes) as state:
        def rows():
            return {table: state.connection.execute('SELECT * FROM ' + state.table(table)).fetchall()
                    for table in tables}
        before = rows()
        cached = {kind: state.records('1', kind) for kind in original}
        clear_checkpoint = checkpoints.clear_checkpoint

        def fail_after_checkpoint_removal(*args):
            clear_checkpoint(*args)
            raise RuntimeError('checkpoint cleanup failed')

        with monkeypatch.context() as patch:
            patch.setattr(checkpoints, 'clear_checkpoint', fail_after_checkpoint_removal)
            with pytest.raises(RuntimeError, match='checkpoint cleanup failed'):
                checkpoints.finish(state, '1', 'wd', fresh)
        assert rows() == before
        assert {kind: state.records('1', kind) for kind in original} == cached

        checkpoints.finish(state, '1', 'wd', fresh)
        for kind, segment in fresh.items():
            assert state.records('1', kind)['wd'] == segment
        for table in ('checkpoints', 'checkpoint_frames', 'checkpoint_analysis', 'refresh_requests'):
            assert state.connection.execute('SELECT COUNT(*) FROM ' + state.table(table)).fetchone() == (0,)
        assert progress.completed(state, '1', 'wd', ['scene', 'markers'], 'new')

    with merge_store.StateStore(path, 'fixture', True) as state:
        for kind, segment in (original if dry_run else fresh).items():
            assert state.records('1', kind)['wd'] == segment


def test_api_failure_resumes_only_synchronization(pipeline,monkeypatch):
    p=pipeline;p.stash.fail_create=True
    assert app.main(p.args)==1
    p.stash.fail_create=False
    monkeypatch.setattr(app.services,"load_tagger",lambda *args:pytest.fail("API failure repeated inference"))
    assert app.main(p.args)==0
    assert p.stash.scene["scene_markers"]


def test_prefetch_is_bounded_and_closes_producer(pipeline,monkeypatch):
    p=pipeline;config=ModelSettings(WD,Path("missing"),batch_size=2, scene_threshold=.4, marker_threshold=.65)
    model=IndexedTagger(config,p.scores["wd"],p.vocab.labels)
    prepared=threading.Event()
    original=model.prepare
    def prepare(image):
        out=original(image)
        if out==3: prepared.set()
        return out
    model.prepare=prepare
    iterator=preparation.prepared_batches(app,model,p.stash,{},p.cues,True)
    batch,_=next(iterator)
    assert batch==[0,1]
    assert prepared.wait(2)
    assert model.prepared==[0,1,2,3]
    iterator.close()
    assert not any(t.name.startswith("prepare-frames") for t in threading.enumerate())


def test_v1_database_is_migrated_once_without_losing_sources(tmp_path,monkeypatch):
    path=tmp_path/"v1.sqlite3"
    segment=current_record({"mode":"scene","meta":{"origin":"inference"},"items":[{"tag":"common","score":.9}]})
    db=sqlite3.connect(path)
    db.executescript(f"PRAGMA application_id={APPLICATION_ID}; PRAGMA user_version=1; "
                    "CREATE TABLE results(scope TEXT,scene TEXT,model TEXT,kind TEXT,data BLOB,PRIMARY KEY(scope,scene,model,kind));")
    db.execute("INSERT INTO results VALUES (?,?,?,?,?)",("fixture","1","wd","scene",state_codec.encode([segment])))
    db.commit();db.close()
    from stash_ai.db import update_database
    update_database(path)
    with merge_store.StateStore(path,"fixture") as state:
        assert state.connection.execute("PRAGMA user_version").fetchone()==(18,)
        assert state.records("1","scene")["wd"]==segment
        assert not progress.completed(state,"1","wd",["scene"],"new-signature")
    monkeypatch.setattr(state_codec,"decode",lambda *args:pytest.fail("migration repeated"))
    with merge_store.StateStore(path,"fixture") as state:
        assert not progress.completed(state,"1","wd",["scene"],"new-signature")


@pytest.mark.parametrize("failure",[False,True])
def test_pruning_follows_only_successful_merge(pipeline,monkeypatch,failure):
    from stash_ai.cleanup import prune as stash_prune
    p=pipeline;p.stash.fail_create=failure
    calls=[]
    monkeypatch.setattr(stash_prune,"prune_tags",lambda *args,**kwargs:calls.append(args) or {})
    assert app.main([*p.args,"--prune-empty-tags"])==int(failure)
    assert len(calls)==2*int(not failure)  # Generic definitions and derived marker groups.
    p.stash.fail_create=False
    monkeypatch.setattr(app.services,"load_tagger",lambda *args:pytest.fail("merge-only loaded weights"))
    assert app.main([*p.args,"--merge-only","--prune-empty-tags"])==0
    assert len(calls)==2*int(not failure)+2


def test_ordered_vocabulary_changes_checkpoint_identity(pipeline):
    p=pipeline
    config=ModelSettings(WD,Path("missing"), scene_threshold=.4, marker_threshold=.65, batch_size=4)
    original=model_identity(app,config,p.vocab)
    reordered=Vocabulary(list(reversed(p.vocab.labels)),p.vocab.categories)
    assert model_identity(app,config,reordered)!=original


@pytest.mark.parametrize("flag",["--resume","--restart"])
def test_resume_flags_are_explicitly_rejected_for_legacy_separate_mode(pipeline,flag):
    with pytest.raises(SystemExit) as exc:app.main([*pipeline.args,"--output-mode","separate",flag])
    assert exc.value.code==2 and not pipeline.loaded
