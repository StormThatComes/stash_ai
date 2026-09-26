"""Behavior tests for exact-label AI fusion; no live Stash or model downloads."""
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import stash_ai as app
import stash_merge as merge
import stash_progress as progress
from stash_taggers import SOURCES, JOYTAG, WD, DTQ, OPPAI, ModelSettings, Vocabulary
from tests.helpers.stash import Stash
from tests.helpers.inference import Tagger


from tests.helpers.evidence import span
from tests.helpers.evidence import hit
from tests.helpers.evidence import empty_policies
pytestmark = pytest.mark.usefixtures("reset_merge")


def store(tmp_path, scope="fixture", dry=False):
    return merge.StateStore(tmp_path/"sources.sqlite3",scope,dry)


def intervals(stash, label="common"):
    return sorted((marker["seconds"], marker.get("end_seconds")) for marker in stash.scene["scene_markers"]
                  if marker["primary_tag"]["name"]=="ai:"+label and app.owned_marker(marker,merge.AI))


def test_interval_union_retains_ends_separate_events_sources_and_extras():
    inputs=[span("joytag",10,30,extras=["extra"]),span("wd",18,42),
            span("oppai",14,35,score=.83,extras=["extra","another"]),span("dtq",70,90)]
    expected=[(10,42,["joytag","wd","oppai"]),(70,90,["dtq"])]
    for sequence in permutations(inputs):
        result=merge.union_intervals(sequence)
        assert [(m["start"],m["end"],m["sources"]) for m in result]==expected
        assert result[0]["extra_tags"]==["another","extra"]
        assert "score" not in result[0]
        assert sorted(c["score"] for c in result[0]["contributors"])==[.83,.9,.9]
    assert inputs[0]["start"]==10


@pytest.mark.parametrize("values,expected",[
    ([(0,10),(10,20)],[(0,20)]),
    ([(0,10),(10.001,20)],[(0,10),(10.001,20)]),
    ([(0,100),(20,30)],[(0,100)]),
    ([(0,10),(8,20),(19,30)],[(0,30)]),
    ([(10,None),(10,None)],[(10,None)]),
    ([(10,30),(20,None),(40,None)],[(10,30),(40,None)]),
])
def test_union_touching_containment_chains_gaps_and_unknown_ends(values,expected):
    merged=merge.union_intervals([span(key,start,end) for key,(start,end) in zip(SOURCES,values)])
    assert [(m["start"],m["end"]) for m in merged]==expected


def test_union_keeps_different_labels_and_case_separate():
    merged=merge.union_intervals([span("joytag",0,20,"blue_hair"),span("wd",0,20,"Blue_hair"),
                                   span("oppai",0,20,"bikini"),span("dtq",0,20,"swimsuit")])
    assert len(merged)==4


@pytest.mark.parametrize("start,end",[(-1,3),(3,1),(float("nan"),3),(0,float("inf")),(True,3)])
def test_bad_intervals_are_rejected_before_writes(start,end):
    with pytest.raises(ValueError): merge.union_intervals([span("wd",start,end)])


def test_state_is_durable_scoped_partial_and_dry_run_does_not_write(tmp_path):
    first={"mode":"scene","meta":{"origin":"inference"},"items":[dict(tag="common",score=.8)]}
    second={"mode":"spans","meta":{"origin":"legacy"},"items":[]}
    with store(tmp_path) as state:
        state.put("1","joytag",{"scene":first,"markers":second})
        state.put("1","wd",{"scene":first})
    path=tmp_path/"sources.sqlite3"
    before=(path.read_bytes(),path.stat().st_mtime_ns)
    with store(tmp_path,dry=True) as state:
        state.put("1","joytag",{"scene":{**first,"items":[]}})
        assert state.records("1","scene")["joytag"][0]["items"]==[]
    assert (path.read_bytes(),path.stat().st_mtime_ns)==before
    with store(tmp_path) as state:
        assert state.records("1","scene")["joytag"][0]==first
        state.put("1","joytag",{"scene":{**first,"items":[]}})
        assert state.records("1","markers")["joytag"][0]==second
        assert state.records("1","scene")["wd"][0]==first
    with store(tmp_path,scope="different-stash") as state:
        assert state.records("1","scene")=={}
    empty=tmp_path/"missing"/"dry.sqlite3"
    with merge.StateStore(empty,"fixture",True) as state:
        state.put("1","wd",{"scene":first})
    assert not empty.parent.exists()


def test_second_writer_is_rejected(tmp_path):
    with store(tmp_path):
        with pytest.raises(RuntimeError,match="Another merged-output process"):
            with store(tmp_path): pass


def test_sync_common_tag_union_idempotence_and_manual_ownership(tmp_path):
    stash=Stash()
    stash.scene["tags"]=[stash.tag("manual")]
    manual=stash.marker(JOYTAG,"common","manual",True)
    inputs=[span("joytag",10,30),span("wd",20,50),span("oppai",60,80)]
    with store(tmp_path) as state:
        merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),
                          [hit("joytag"),hit("wd"),hit("oppai","unique_op")],inputs)
        assert {t["name"] for t in stash.scene["tags"]}=={"manual","ai:common","ai:unique_op"}
        assert intervals(stash)==[(10,50),(60,80)]
        assert manual in stash.scene["scene_markers"]
        shared=next(m for m in stash.scene["scene_markers"] if m["seconds"]==10)
        assert shared["title"]=="[AI] common [jt+wd]"
        assert {t["name"] for t in shared["tags"]}=={merge.AI.owner_tag, "m:common"}
        stash.writes.clear()
        merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),
                          [hit("joytag"),hit("wd"),hit("oppai","unique_op")],inputs)
        assert not stash.writes


def test_untracked_ai_scene_association_remains_manual(tmp_path):
    stash=Stash();tag=stash.tag("ai:common");stash.scene["tags"]=[tag]
    with store(tmp_path) as state:
        for hits in ([hit("wd")],[]):
            merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),hits,[])
        assert stash.scene["tags"]==[tag]


def test_native_marker_id_survives_large_start_shift(tmp_path):
    stash=Stash()
    with store(tmp_path) as state:
        for inputs in ([span("wd",20,50)],[span("wd",20,50),span("joytag",10,30)]):
            merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),[],inputs)
            if len(inputs)==1: original=stash.scene["scene_markers"][0]["id"]
        assert len(stash.scene["scene_markers"])==1
        assert stash.scene["scene_markers"][0]["id"]==original and intervals(stash)==[(10,50)]


def test_legacy_import_migrates_all_sources_without_fabricating_scores(tmp_path):
    stash=Stash();manual_tag=stash.tag("manual extra")
    stash.scene["tags"]=[stash.tag("jt:common"),stash.tag("wd:common"),stash.tag("op:unique_op"),stash.tag("human")]
    for source,start,end in [(JOYTAG,10,30),(WD,20,50)]:
        marker=stash.marker(source,"common",source.key)
        marker.update(seconds=start,end_seconds=end)
        marker["tags"].append(manual_tag)
    manual=stash.marker(JOYTAG,"common","manual",True)
    with store(tmp_path) as state:
        merge.bootstrap_legacy(app,state,stash.find_scene("1"))
        hits,spans=merge.evidence(app,state,"1",empty_policies(),empty_policies())
        assert {s["source"] for s in hits}=={"joytag","wd","oppai"}
        assert all(s["score"] is None for s in hits+spans)
        assert {s["legacy_marker_id"] for s in spans}=={"joytag","wd"}
        merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),hits,spans)
        assert {t["name"] for t in stash.scene["tags"]}=={"ai:common","ai:unique_op","human"}
        assert intervals(stash)==[(10,50)] and manual in stash.scene["scene_markers"]
        assert all("manual extra" in {t["name"] for t in m["tags"]} for m in stash.scene["scene_markers"] if app.owned_marker(m,merge.AI))
        assert "jt:common" in stash.tags  # No global tag objects are deleted.


def test_failed_api_write_keeps_legacy_results_and_pending_merge_recovers(tmp_path):
    stash=Stash();stash.scene["tags"]=[stash.tag("jt:common")]
    old=stash.marker(JOYTAG,"common","legacy");old.update(seconds=10,end_seconds=30)
    stash.fail_create=True
    with store(tmp_path) as state:
        merge.bootstrap_legacy(app,state,stash.find_scene("1"))
        hits,spans=merge.evidence(app,state,"1",empty_policies(),empty_policies())
        with pytest.raises(RuntimeError,match="injected"):
            merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),hits,spans)
        assert old in stash.scene["scene_markers"] and stash.scene["tags"][0]["name"]=="jt:common"
        assert state.snapshot("1","markers")[1] is not None
        stash.fail_create=False
        merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),hits,spans)
        assert intervals(stash)==[(10,30)] and state.snapshot("1","markers")[1] is None


def test_main_combines_all_four_models_and_updates_only_one_contribution(merge_pipeline):
    p=merge_pipeline
    assert app.main(p.args)==0
    assert len(p.loaded)==4 and all(model.closed for model in p.loaded)
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:common","ai:unique_op"}
    assert intervals(p.stash)==[(10,50),(70,90)]
    p.scores["oppai"][:]=0
    assert app.main(["--models","oppai",*p.args[2:]])==0
    assert len(p.loaded)==5 and p.loaded[-1].settings.source==OPPAI
    assert intervals(p.stash)==[(10,40),(70,90)]
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:common"}


def test_inference_failure_retains_previous_contribution_and_other_models_continue(merge_pipeline):
    p=merge_pipeline
    assert app.main(p.args)==0
    before=deepcopy(p.stash.scene)
    p.scores["wd"][:]=np.nan
    assert app.main(p.args)==1
    assert intervals(p.stash)==[(10,50),(70,90)]
    assert {t["name"] for t in before["tags"]}=={t["name"] for t in p.stash.scene["tags"]}
    assert all(model.closed for model in p.loaded)


def test_merge_only_needs_no_model_assets_and_can_restore_unblocked_cached_tags(merge_pipeline,monkeypatch,tmp_path):
    p=merge_pipeline
    assert app.main(p.args)==0
    monkeypatch.setattr(app,"load_vocabulary",lambda *args:pytest.fail("merge-only loaded vocabulary"))
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("merge-only loaded weights"))
    policy=tmp_path/"scene_ban.yml"
    policy.write_text("version: 1\nexcluded_scene_tags: [unique_op]\n")
    args=p.args+["--merge-only","--offline","--scene-tag-exclusions",str(policy)]
    assert app.main(args)==0
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:common"}
    assert intervals(p.stash,"unique_op")==[]
    policy.write_text("version: 1\nexcluded_scene_tags: []\n")
    assert app.main(args)==0
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:common","ai:unique_op"}
    assert intervals(p.stash,'unique_op')==[(10,30)]


def test_source_blacklists_shrink_union_without_removing_other_sources(merge_pipeline,monkeypatch,tmp_path):
    p=merge_pipeline
    assert app.main(p.args)==0
    core=tmp_path/"core.yml";core.write_text("version: 1\nexcluded_core_tags: []\nby_model:\n  oppai: [common]\n")
    scene=tmp_path/"scene.yml";scene.write_text("version: 1\nexcluded_scene_tags: []\nby_model:\n  oppai: [common, unique_op]\n")
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("cleanup loaded weights"))
    monkeypatch.setattr(app,"load_vocabulary",lambda *args:pytest.fail("cleanup loaded vocabulary"))
    assert app.main([*p.args,"--cleanup-only","--marker-core-exclusions",str(core),"--scene-tag-exclusions",str(scene)])==0
    assert intervals(p.stash)==[(10,40),(70,90)]
    assert intervals(p.stash,"unique_op")==[]
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:common"}


def test_common_core_exclusion_does_not_remove_scene_tags_or_extras(merge_pipeline,tmp_path):
    p=merge_pipeline
    p.scores["oppai"][:2,2]=.94
    core=tmp_path/"core.yml";core.write_text("version: 1\nexcluded_core_tags: [extra]\n")
    assert app.main([*p.args,"--marker-core-exclusions",str(core)])==0
    assert "ai:extra" in {t["name"] for t in p.stash.scene["tags"]}
    assert not intervals(p.stash,"extra")
    assert "ai:extra" in {tag["name"] for m in p.stash.scene["scene_markers"] if m["primary_tag"]["name"]=="ai:unique_op" for tag in m["tags"]}


@pytest.mark.parametrize("mode,untouched",[("--markers-only","tags"),("--scene-tags-only","scene_markers")])
def test_partial_modes_preserve_other_output_and_cached_evidence(merge_pipeline,mode,untouched):
    p=merge_pipeline
    assert app.main(p.args)==0
    before=deepcopy(p.stash.scene[untouched])
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        old=state.records("1","scene" if untouched=="tags" else "markers")
    p.scores["oppai"][:]=0
    assert app.main(["--models","oppai",*p.args[2:],mode])==0
    assert p.stash.scene[untouched]==before
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert state.records("1","scene" if untouched=="tags" else "markers")==old


def test_dry_run_preserves_stash_and_state(merge_pipeline,monkeypatch):
    p=merge_pipeline
    assert app.main(p.args)==0
    before=(deepcopy(p.stash.scene),p.path.read_bytes(),p.path.stat().st_mtime_ns)
    p.stash.dry_run=True;p.stash.writes.clear();p.scores["oppai"][:]=0
    assert app.main(p.args+["--dry-run"])==0
    assert not p.stash.writes
    assert (p.stash.scene,p.path.read_bytes(),p.path.stat().st_mtime_ns)==before


def test_inspection_works_offline_without_stash_or_weights(merge_pipeline,monkeypatch,capsys):
    p=merge_pipeline
    assert app.main(p.args)==0
    capsys.readouterr()
    monkeypatch.setattr(app,"create_stash_client",lambda *args:pytest.fail("inspection contacted Stash"))
    monkeypatch.setattr(app,"load_vocabulary",lambda *args:pytest.fail("inspection loaded vocabulary"))
    assert app.main([*p.args,"--explain-tag","ai:common","--offline"])==0
    output=capsys.readouterr().out
    reports=json.loads(output[output.index("[\n"):])
    assert reports[0]["tag"]=="ai:common"
    assert {h["source"] for h in reports[0]["scene_contributors"]}==set(SOURCES)
    assert [(m["start"],m["end"]) for m in reports[0]["markers"]]==[(10,50),(70,90)]


def test_only_untagged_uses_successful_records_even_with_no_predictions(merge_pipeline,monkeypatch):
    p=merge_pipeline
    for scores in p.scores.values(): scores[:]=0
    assert app.main(p.args)==0
    monkeypatch.setattr(app,"SKIP_EXISTING_OUTPUTS",True)
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("already processed model loaded again"))
    assert app.main(p.args)==0


def test_combined_hierarchy_has_one_namespace_and_all_labels(tmp_path):
    root=Path(__file__).resolve().parents[1]
    args=SimpleNamespace(no_hierarchy=False,hierarchy=root/"joytag_hierarchy.yml",**{
        key+"_hierarchy":root/f"{key}_hierarchy.yml" for key in ("wd","dtq","oppai")})
    combined=merge.combined_hierarchy(app,args)
    expected=set()
    for key in SOURCES:
        data=yaml.safe_load((root/f"{key}_hierarchy.yml").read_text(encoding="utf-8"))
        expected.update(tag for tags in data["groups"].values() for tag in tags)
    assert set(combined.leaf_parents)==expected
    assert combined.leaf_parents["blue_hair"]==("ai:category:Appearance / Hair / Color",)
    assert {name for name,parents in combined.category_parents.items() if not parents}=={"ai:category:AI"}
    assert all(name.startswith("ai:category:") for name in combined.category_parents)
    stash=Stash();manual=stash.tag("manual parent")
    stash.tag("ai:blue_hair")["parents"]=[manual]
    app.TagResolver(stash,combined,source=merge.AI).resolve("ai:blue_hair")
    assert "manual parent" in {p["name"] for p in stash.tags["ai:blue_hair"]["parents"]}


def test_missing_source_state_does_not_destroy_existing_ai_markers(tmp_path):
    stash=Stash();marker=stash.marker(merge.AI,"common","owned")
    before=deepcopy(stash.scene)
    with store(tmp_path) as state:
        with pytest.raises(RuntimeError,match="source state"):
            merge.bootstrap_legacy(app,state,stash.find_scene("1"))
    assert stash.scene==before and not stash.writes


def test_no_frame_fusion_across_models(tmp_path):
    with store(tmp_path) as state:
        for source,indices in [("joytag",{0,2}),("wd",{1,3})]:
            state.put("1",source,{"markers":{"mode":"frames","meta":{"origin":"inference"},"items":[
                dict(start=i*5,end=(i+1)*5,tags=[["common",.9]] if i in indices else []) for i in range(4)]}})
        _,spans=merge.evidence(app,state,"1",empty_policies(),empty_policies())
        assert spans==[]  # Each model's MIN_HITS is still checked independently.


def test_manual_extras_survive_absorbed_ai_markers_and_later_split(tmp_path):
    stash=Stash()
    with store(tmp_path) as state:
        resolver=app.TagResolver(stash,source=merge.AI)
        base=[span("joytag",10,20),span("wd",30,40)]
        merge.sync_merged(app,stash,resolver,state,stash.find_scene("1"),[],base)
        for marker,name in zip(stash.scene["scene_markers"],["manual A","manual B"]):
            marker["tags"].append(stash.tag(name))
        merge.sync_merged(app,stash,resolver,state,stash.find_scene("1"),[],base+[span("oppai",20,30)])
        assert intervals(stash)==[(10,40)]
        assert {"manual A","manual B"} <= {tag["name"] for tag in stash.scene["scene_markers"][0]["tags"]}
        merge.sync_merged(app,stash,resolver,state,stash.find_scene("1"),[],base)
        assert intervals(stash)==[(10,20),(30,40)]
        assert all({"manual A","manual B"} <= {t["name"] for t in m["tags"]} for m in stash.scene["scene_markers"])


@pytest.mark.parametrize("failure",["update_scene_tags","delete_marker"])
def test_recovery_after_partial_api_success_preserves_ownership(tmp_path,monkeypatch,failure):
    stash=Stash();stash.scene["tags"]=[stash.tag("human"),stash.tag("wd:common")]
    old=stash.marker(WD,"common","legacy");old.update(seconds=10,end_seconds=30)
    actual=getattr(stash,failure)
    def interrupted(*args):
        actual(*args)  # Simulate server success followed by a lost response.
        raise RuntimeError("lost response")
    with store(tmp_path) as state:
        merge.bootstrap_legacy(app,state,stash.find_scene("1"))
        hits,spans=merge.evidence(app,state,"1",empty_policies(),empty_policies())
        monkeypatch.setattr(stash,failure,interrupted)
        with pytest.raises(RuntimeError,match="lost response"):
            merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),hits,spans)
        monkeypatch.setattr(stash,failure,actual)
        merge.bootstrap_legacy(app,state,stash.find_scene("1"))
        merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),hits,spans)
        assert intervals(stash)==[(10,30)]
        assert {t["name"] for t in stash.scene["tags"]}=={"human","ai:common"}
        assert all(state.snapshot("1",kind)[1] is None for kind in merge.KINDS)
        # It must remain managed after recovery, not accidentally become manual.
        merge.sync_merged(app,stash,app.TagResolver(stash,source=merge.AI),state,stash.find_scene("1"),[],[])
        assert {t["name"] for t in stash.scene["tags"]}=={"human"} and not intervals(stash)


def test_new_legacy_results_are_imported_alongside_existing_cache_once(merge_pipeline,monkeypatch):
    p=merge_pipeline
    assert app.main(p.args)==0
    p.stash.scene["tags"].append(p.stash.tag("wd:extra"))
    old=p.stash.marker(WD,"common","new-separate");old.update(seconds=90,end_seconds=100)
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("merge-only loaded weights"))
    assert app.main([*p.args,"--merge-only"])==0
    assert intervals(p.stash)==[(10,50),(70,100)]
    assert "ai:extra" in {t["name"] for t in p.stash.scene["tags"]}
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert len(state.records("1","markers")["wd"])==2
    assert app.main([*p.args,"--merge-only"])==0
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert len(state.records("1","markers")["wd"])==2


def test_retry_does_not_reimport_old_legacy_after_successful_new_inference(tmp_path):
    stash=Stash();old=stash.marker(WD,"common","legacy")
    with store(tmp_path) as state:
        merge.bootstrap_legacy(app,state,stash.find_scene("1"))
        state.put("1","wd",{"markers":{"mode":"frames","meta":{"origin":"inference"},"items":[]}})
        merge.bootstrap_legacy(app,state,stash.find_scene("1"))
        assert state.records("1","markers")["wd"][0]["items"]==[]
        assert len(state.records("1","markers")["wd"])==1


@pytest.mark.parametrize("mode,untouched",[("--markers-only","tags"),("--scene-tags-only","scene_markers")])
def test_merge_only_partial_modes_never_infer_or_edit_other_output(merge_pipeline,monkeypatch,tmp_path,mode,untouched):
    p=merge_pipeline
    assert app.main(p.args)==0
    before=deepcopy(p.stash.scene[untouched])
    core=tmp_path/"core.yml";core.write_text("version: 1\nexcluded_core_tags: [common]\n")
    scene=tmp_path/"scene.yml";scene.write_text("version: 1\nexcluded_scene_tags: [common]\n")
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("merge-only loaded weights"))
    assert app.main([*p.args,"--merge-only",mode,"--marker-core-exclusions",str(core),"--scene-tag-exclusions",str(scene)])==0
    assert p.stash.scene[untouched]==before
    if mode=="--markers-only": assert not intervals(p.stash)
    else: assert "ai:common" not in {t["name"] for t in p.stash.scene["tags"]}


def test_cleanup_selected_bridge_source_splits_union_and_keeps_manual_removals(merge_pipeline,tmp_path):
    p=merge_pipeline
    scope=merge.scope_for(app.STASH_URL)
    inputs=[span("joytag",10,20),span("oppai",20,30),span("wd",30,40),span("dtq",70,80,"extra")]
    with merge.StateStore(p.path,scope) as state:
        merge.sync_merged(app,p.stash,app.TagResolver(p.stash,source=merge.AI),state,p.stash.find_scene("1"),
                          [hit("joytag"),hit("oppai"),hit("dtq","extra")],inputs)
    assert intervals(p.stash)==[(10,40)]
    p.stash.scene["tags"]=[t for t in p.stash.scene["tags"] if t["name"]!="ai:extra"]
    p.stash.scene["scene_markers"]=[m for m in p.stash.scene["scene_markers"] if m["primary_tag"]["name"]!="ai:extra"]
    core=tmp_path/"core.yml";core.write_text("version: 1\nexcluded_core_tags: [common]\n")
    assert app.main(["--models","oppai",*p.args[2:],"--cleanup-only","--marker-core-exclusions",str(core)])==0
    assert intervals(p.stash)==[(10,20),(30,40)]  # joytag and wd survive a common ban scoped to oppai.
    assert not intervals(p.stash,"extra")
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:common"}


@pytest.mark.parametrize('kind', ['scene', 'markers'])
def test_enabling_replacement_collapses_reused_history_without_inference(merge_pipeline, monkeypatch, kind):
    p = merge_pipeline
    monkeypatch.setattr(app, 'REPLACE_SCENE_TAGS', False)
    monkeypatch.setattr(app, 'REPLACE_MARKERS', False)
    args = ['--models', 'oppai', *p.args[2:]]

    assert app.main(args) == 0
    p.scores['oppai'][:] = 0
    # Force a genuinely new input signature, then keep it stable for the reuse run.
    monkeypatch.setattr(p.stash, 'get_media', lambda url: b'changed-media')
    assert app.main(args) == 0

    config = app.load_model_settings(app.MODELS_CONFIG_FILE, ['oppai'], None)[0]
    identity = progress.model_identity(app, config, app.load_vocabulary(config))
    _, _, current_signature = progress.inspect_input(app, p.stash, p.stash.scene['paths'], identity)
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        scene_segments = state.records('1', 'scene')['oppai']
        marker_segments = state.records('1', 'markers')['oppai']
        assert len(scene_segments) == 2
        assert len(marker_segments) == 2
        assert scene_segments[-1]['meta']['inference_signature'] == current_signature
        assert marker_segments[-1]['meta']['inference_signature'] == current_signature
    assert 'ai:unique_op' in {t['name'] for t in p.stash.scene['tags']}
    assert intervals(p.stash, 'unique_op') == [(10, 30)]

    monkeypatch.setattr(app, 'REPLACE_SCENE_TAGS', kind == 'scene')
    monkeypatch.setattr(app, 'REPLACE_MARKERS', kind == 'markers')
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('replacement change reran inference'))
    p.stash.writes.clear()

    assert app.main(args) == 0

    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        expected = {'scene': 1 if kind == 'scene' else 2,
                    'markers': 1 if kind == 'markers' else 2}
        assert len(state.records('1', 'scene')['oppai']) == expected['scene']
        assert len(state.records('1', 'markers')['oppai']) == expected['markers']

    if kind == 'scene':
        assert 'ai:unique_op' not in {t['name'] for t in p.stash.scene['tags']}
        assert intervals(p.stash, 'unique_op') == [(10, 30)]
    else:
        assert 'ai:unique_op' in {t['name'] for t in p.stash.scene['tags']}
        assert intervals(p.stash, 'unique_op') == []


def test_replacement_disabled_keeps_evidence_but_blacklists_still_apply(merge_pipeline,monkeypatch,tmp_path):
    p=merge_pipeline
    assert app.main(p.args)==0
    monkeypatch.setattr(app,"REPLACE_SCENE_TAGS",False)
    monkeypatch.setattr(app,"REPLACE_MARKERS",False)
    p.scores["oppai"][:]=0
    assert app.main(["--models","oppai",*p.args[2:]])==0
    assert intervals(p.stash)==[(10,50),(70,90)]
    assert "ai:unique_op" in {t["name"] for t in p.stash.scene["tags"]}
    core=tmp_path/"core.yml";core.write_text("version: 1\nexcluded_core_tags: [unique_op]\n")
    scene=tmp_path/"scene.yml";scene.write_text("version: 1\nexcluded_scene_tags: [unique_op]\n")
    assert app.main([*p.args,"--merge-only","--marker-core-exclusions",str(core),"--scene-tag-exclusions",str(scene)])==0
    assert not intervals(p.stash,"unique_op")
    assert "ai:unique_op" not in {t["name"] for t in p.stash.scene["tags"]}


def test_caps_follow_blacklists_and_keep_all_qualified_evidence(merge_pipeline,monkeypatch,tmp_path):
    p=merge_pipeline
    p.scores["oppai"][:2,2]=.95
    monkeypatch.setattr(app,"MAX_TAGS",1)
    monkeypatch.setattr(app,"MARKER_MAX_TAGS_PER_FRAME",1)
    monkeypatch.setattr(app,"MAX_MARKERS_PER_SCENE",1)
    core=tmp_path/"core.yml";core.write_text("version: 1\nexcluded_core_tags: [extra]\n")
    scene=tmp_path/"scene.yml";scene.write_text("version: 1\nexcluded_scene_tags: [extra]\n")
    assert app.main(["--models","oppai",*p.args[2:],"--marker-core-exclusions",str(core),"--scene-tag-exclusions",str(scene)])==0
    assert {t["name"] for t in p.stash.scene["tags"]}=={"ai:unique_op"}
    assert intervals(p.stash,"unique_op")==[(10,30)]
    assert not intervals(p.stash) and not intervals(p.stash,"extra")
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        assert {item["tag"] for item in state.records("1","scene")["oppai"][0]["items"]}=={"common","unique_op","extra"}


def test_point_only_stash_retains_full_source_ranges(merge_pipeline,monkeypatch,capsys):
    p=merge_pipeline;p.stash.has_end_time=False
    assert app.main(p.args)==0
    assert intervals(p.stash)==[(10,None),(70,None)]
    assert all("end_seconds" not in write[1] for write in p.stash.writes if write[0] in {"create","update"})
    with merge.StateStore(p.path,merge.scope_for(app.STASH_URL),True) as state:
        snapshot,_=state.snapshot("1","markers")
        assert [(m["start"],m["end"]) for m in merge.union_intervals(snapshot["contributions"]) if m["tag"]=="common"]==[(10,50),(70,90)]


def test_merge_only_preserves_unknown_legacy_end_and_no_missing_tag_replacement(merge_pipeline,monkeypatch):
    p=merge_pipeline
    p.stash.scene["tags"]=[p.stash.tag("wd:common")]
    old=p.stash.marker(WD,"common","legacy");old.update(seconds=10,end_seconds=None)
    monkeypatch.setattr(app,"CREATE_MISSING",False)
    p.stash.tag(merge.AI.owner_tag)
    assert app.main([*p.args,"--merge-only"])==0
    assert p.stash.scene["tags"][0]["name"]=="wd:common"
    assert p.stash.scene["scene_markers"]==[old]
    monkeypatch.setattr(app,"CREATE_MISSING",True)
    assert app.main([*p.args,"--merge-only"])==0
    assert intervals(p.stash)==[(10,None)]


def test_scene_cap_does_not_drop_previous_runs_or_legacy_confidence(tmp_path,monkeypatch):
    monkeypatch.setattr(app,"MAX_TAGS",1)
    with store(tmp_path) as state:
        for origin,items in [("inference",[("old",.95),("low",.7)]),
                             ("inference",[("new",.9)]),("legacy",[("old",None),("legacy",None)])]:
            state.put("1","wd",{"scene":{"mode":"scene","meta":{"origin":origin},
                      "items":[dict(tag=tag,score=score) for tag,score in items]}},{"scene":False})
        hits,_=merge.evidence(app,state,"1",empty_policies(),empty_policies())
        assert {h["tag"] for h in hits}=={"old","new","legacy"}
        assert {h["score"] for h in hits if h["tag"]=="old"}=={.95,None}


@pytest.mark.parametrize("has_end,use_end",[(False,True),(True,False)])
def test_cleanup_point_output_uses_recorded_extent_when_start_moves(merge_pipeline,monkeypatch,tmp_path,has_end,use_end):
    p=merge_pipeline;p.stash.has_end_time=has_end
    monkeypatch.setattr(app,"MARKER_USE_END_TIME",use_end)
    assert app.main(p.args)==0
    first=next(m for m in p.stash.scene["scene_markers"] if m["seconds"]==10 and m["primary_tag"]["name"]=="ai:common")
    first["tags"].append(p.stash.tag("manual extra"))
    core=tmp_path/"core.yml"
    core.write_text("version: 1\nexcluded_core_tags: []\nby_model:\n  joytag: [common]\n  wd: [common]\n")
    assert app.main([*p.args,"--cleanup-markers-only","--marker-core-exclusions",str(core)])==0
    assert intervals(p.stash)==[(30,None),(70,None)]
    moved=next(m for m in p.stash.scene["scene_markers"] if m["id"]==first["id"])
    assert moved["seconds"]==30 and "manual extra" in {t["name"] for t in moved["tags"]}
    p.stash.writes.clear()
    assert app.main([*p.args,"--cleanup-markers-only","--marker-core-exclusions",str(core)])==0
    assert not p.stash.writes
