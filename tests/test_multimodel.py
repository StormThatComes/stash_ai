"""Offline behavior tests. Run: python -m pytest -q tests/test_multimodel.py"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest
import yaml
from stash_ai import inference
from stash_ai import cli as app
from stash_ai import taggers as adapters
from stash_ai.taggers import JOYTAG, WD, DTQ, OPPAI, SOURCES, ModelSettings, Vocabulary, load_vocabulary, select_models, validate_scores, prepare_wd, prepare_dtq, prepare_oppai, OnnxTagger


from tests.helpers.stash import Stash
from tests.helpers.stash import sync
from tests.helpers.inference import Tagger
pytestmark = pytest.mark.usefixtures("reset_multimodel")


@pytest.mark.parametrize("source", list(SOURCES.values()))
def test_source_isolation_and_idempotence(source):
    stash=Stash()
    stash.scene["tags"].append(stash.tag("manual"))
    for item in SOURCES.values():
        stash.scene["tags"].append(stash.tag(item.prefix+"old"))
        stash.marker(item,"old",item.key)
    manual=stash.marker(source,"old","manual",True)
    other=deepcopy([m for m in stash.scene["scene_markers"] if m["id"]!=source.key])
    spec=app.MarkerSpec("new",10,12,.9,2,("extra",))
    sync(stash,source,[("new",.9)],[spec])
    assert {t["name"] for t in stash.scene["tags"]} == {"manual",source.prefix+"new"}|{s.prefix+"old" for s in SOURCES.values() if s!=source}
    assert all(m in stash.scene["scene_markers"] for m in other)
    assert manual in stash.scene["scene_markers"]
    stash.writes.clear()
    result=sync(stash,source,[("new",.9)],[spec])
    assert not stash.writes and result["markers_unchanged"]==1


@pytest.mark.parametrize("source",list(SOURCES.values()))
def test_global_blacklist_covers_markers_but_core_only_tags_remain_extras(source,monkeypatch):
    monkeypatch.setattr(app,"REPLACE_SCENE_TAGS",False)
    monkeypatch.setattr(app,"REPLACE_MARKERS",False)
    stash=Stash()
    stash.scene["tags"]=[stash.tag(source.prefix+"scene_ban"),stash.tag("scene_ban"),stash.tag(source.prefix+"keep")]
    stash.marker(source,"core_ban","remove")
    kept=stash.marker(source,"old","keep")
    manual=stash.marker(source,"core_ban","manual",True)
    specs=[app.MarkerSpec("core_ban",0,2,.9,2),app.MarkerSpec("scene_ban",4,6,.9,2,("core_ban",)),
           app.MarkerSpec('allowed',8,10,.9,2,('core_ban','scene_ban'))]
    result=sync(stash,source,[("core_ban",.9),("scene_ban",.9)],specs,frozenset({"core_ban"}),frozenset({"scene_ban"}))
    assert {t["name"] for t in stash.scene["tags"]}=={"scene_ban",source.prefix+"keep",source.prefix+"core_ban"}
    assert kept in stash.scene["scene_markers"] and manual in stash.scene["scene_markers"]
    assert not any(m['primary_tag']['name']==source.prefix+'scene_ban' for m in stash.scene['scene_markers'])
    generated=next(m for m in stash.scene["scene_markers"] if m["primary_tag"]["name"]==source.prefix+"allowed")
    assert source.prefix+"core_ban" in {t["name"] for t in generated["tags"]}
    assert source.prefix+'scene_ban' not in {t['name'] for t in generated['tags']}
    assert result["markers_core_cleaned"]==result["scene_tags_cleaned"]==1


@pytest.mark.parametrize("source",list(SOURCES.values()))
def test_dry_run_and_failure_preserve_existing_markers(source):
    stash=Stash(True)
    old=stash.marker(source,"old","old")
    specs=[app.MarkerSpec("new",0,2,.9,2)]
    result=sync(stash,source,[("new",.9)],specs)
    assert not stash.writes and result["markers_created"]==result["markers_deleted"]==1
    stash.dry_run,stash.fail_create=False,True
    with pytest.raises(RuntimeError,match="injected"):
        sync(stash,source,[("new",.9)],specs)
    assert old in stash.scene["scene_markers"]
    assert not any(w[0] in {"scene","delete"} for w in stash.writes)


@pytest.mark.parametrize("dry",[False,True])
def test_cleanup_all_refreshes_between_sources(tmp_path,monkeypatch,dry):
    stash=Stash(dry)
    for source in SOURCES.values():
        stash.scene["tags"].append(stash.tag(source.prefix+"blocked"))
        stash.marker(source,"blocked",source.key)
    stash.scene["tags"].append(stash.tag("blocked"))
    stash.marker(WD,"blocked","manual",True)
    before=deepcopy(stash.scene)
    monkeypatch.setattr(app,"create_stash_client",lambda _:stash)
    from tests.helpers.cleanup import options, install
    install(monkeypatch, stash)
    command = [*options(tmp_path, ['blocked'], ['blocked'], output='separate'),
               '--cleanup-strategy', 'scenes', '--scene-id', '1']
    assert app.main(command + (['--dry-run'] if dry else [])) == 0
    if dry: assert stash.scene==before and not stash.writes
    else:
        assert [t["name"] for t in stash.scene["tags"]]==["blocked"]
        assert stash.scene["scene_markers"]==[]


@pytest.mark.parametrize("source",[WD,OPPAI])
@pytest.mark.parametrize("mode",["markers","scenes"])
def test_output_modes(monkeypatch,mode,source):
    stash=Stash();stash.scene["tags"].append(stash.tag(source.prefix+"old"));stash.marker(source,"old","old")
    before=deepcopy(stash.scene)
    monkeypatch.setattr(app,"APPLY_SCENE_TAGS",mode=="scenes")
    monkeypatch.setattr(app,"CREATE_MARKERS",mode=="markers")
    sync(stash,source,[("new",.9)],[app.MarkerSpec("new",0,2,.9,2)])
    key="tags" if mode=="markers" else "scene_markers"
    assert stash.scene[key]==before[key]


@pytest.mark.parametrize("fail_wd",[False,True])
def test_main_all_releases_models_and_preserves_failed_source(monkeypatch,fail_wd):
    stash=Stash();stash.scene["tags"].append(stash.tag("wd:old"));old=stash.marker(WD,"old","old")
    vocab=Vocabulary(["tag"],{"tag":"General"})
    cues=[app.Cue(i,i+1,"x",(0,0,1,1)) for i in range(2)]
    monkeypatch.setattr(app,"create_stash_client",lambda _:stash)
    monkeypatch.setattr(app,"load_vocabulary",lambda _:vocab)
    monkeypatch.setattr(inference,"fetch_cues",lambda *args:(cues,2))
    monkeypatch.setattr(inference,"iter_sprite_frames",lambda *args:((cue,None) for cue in cues))
    loaded=[]
    def load(settings,vocabulary,prepare):
        assert all(t.closed for t in loaded)
        model=Tagger(settings,vocabulary.labels,np.full((2,1),np.nan if fail_wd and settings.source==WD else .9,np.float32))
        loaded.append(model);return model
    monkeypatch.setattr(app,"load_tagger",load)
    assert app.main(["--models","all","--no-hierarchy"])==int(fail_wd)
    assert all(t.closed for t in loaded) and len(loaded)==len(SOURCES)
    assert {t["name"] for t in stash.scene["tags"]}=={"jt:tag","dtq:tag","op:tag","wd:old" if fail_wd else "wd:tag"}
    assert (old in stash.scene["scene_markers"])==fail_wd


@pytest.mark.parametrize("aggregation",["max","topk_mean"])
@pytest.mark.parametrize("batch",[1,2,4])
def test_every_label_and_streaming_aggregation(monkeypatch,aggregation,batch):
    labels=["core_ban","scene_ban"]+[f"tag{i}" for i in range(40)]
    scores=np.random.default_rng(7).uniform(.7,.99,(5,len(labels))).astype(np.float32)
    cues=[app.Cue(i,i+1,"x",(0,0,1,1)) for i in range(5)]
    monkeypatch.setattr(app,"AGGREGATION",aggregation)
    monkeypatch.setattr(inference,"iter_sprite_frames",lambda *args:((c,None) for c in cues))
    predictions,frames=inference.predict_scene(app, Tagger(ModelSettings(WD,Path("."),batch_size=batch),labels,scores),Stash(),{},cues,frozenset({"scene_ban"}))
    expected=scores.max(axis=0) if aggregation=="max" else np.sort(scores,axis=0)[-2:].mean(axis=0)
    assert len(predictions)==len(labels)-1
    for tag,score in predictions: assert score==pytest.approx(expected[labels.index(tag)])
    markers,_=app.build_markers(frames,frozenset({"core_ban"}))
    assert len(markers)==len(labels)-2 and "core_ban" in markers[0].extra_tags
    assert all('scene_ban' not in m.extra_tags for m in markers)
    assert all(m.start==0 and m.end==5 and m.hits==5 for m in markers)


def test_core_limit_and_gap_timing(monkeypatch):
    monkeypatch.setattr(app,"MARKER_MAX_TAGS_PER_FRAME",1)
    frames=[app.FramePrediction(app.Cue(i,i+1,"x",(0,0,1,1)),(("blocked",.99),("valid",.8))) for i in [0,1,4,5]]
    markers,total=app.build_markers(frames,frozenset({"blocked"}))
    assert total==2 and [(m.start,m.end) for m in markers]==[(0,2),(4,6)]
    assert all(m.extra_tags==("blocked",) for m in markers)
    monkeypatch.setattr(app,"MARKER_MERGE_GAP_SECONDS",2)
    markers,_=app.build_markers(frames,frozenset({"blocked"}))
    assert len(markers)==1 and markers[0].hits==4


def test_hierarchy_loader_requires_explicit_source():
    import inspect

    parameter = inspect.signature(app.load_tag_hierarchy).parameters["source"]
    assert parameter.default is inspect.Parameter.empty


def test_original_hierarchy_and_model_settings():
    root=Path(__file__).resolve().parents[1]
    data=yaml.safe_load((root/"joytag_hierarchy.yml").read_text())
    labels=sorted({t for group in data["groups"].values() for t in group})
    hierarchy=app.load_tag_hierarchy(root/"joytag_hierarchy.yml",labels,JOYTAG)
    assert len(hierarchy.leaf_parents)==5813
    assert hierarchy.description_prefix==JOYTAG.hierarchy_description
    assert [s.source for s in app.load_model_settings(root/"tagger_models.yml",["all"],None)]==[JOYTAG,WD,DTQ,OPPAI]
    assert select_models(["dtq,wd","wd"])==["dtq","wd"]


def test_model_hierarchies_include_all_categories_and_keep_manual_parents():
    stash=Stash();manual=stash.tag("manual parent");other=stash.tag("jt:category:Other")
    other["description"]=JOYTAG.hierarchy_description
    old=stash.tag("wd:category:Old");old["description"]=WD.hierarchy_description
    stash.tag("wd:tag")["parents"]=[manual,other,old]
    vocab=Vocabulary(["tag","character","unknown"],{"tag":"General","character":"Character"})
    hierarchy=app.metadata_hierarchy(vocab,WD)
    resolver=app.TagResolver(stash,hierarchy,source=WD)
    for name in vocab.labels: resolver.resolve("wd:"+name)
    assert {p["name"] for p in stash.tags["wd:tag"]["parents"]}=={"manual parent","jt:category:Other","wd:category:General"}
    assert all("wd:"+name in stash.tags for name in vocab.labels)


@pytest.mark.parametrize("body",['version: 1\nexcluded_core_tags: ["wd:tag"]','version: 1\nexcluded_core_tags: [a,a]',
    'version: 1\nexcluded_core_tags: [true]','version: 1\nexcluded_core_tags: []\nby_model: {wrong: []}',
    'version: 1\nexcluded_core_tags: []\nexcluded_core_tags: []'])
def test_bad_policies(tmp_path,body):
    path=tmp_path/"policy.yml";path.write_text(body)
    with pytest.raises(ValueError): app.load_exclusion_policy(path,"excluded_core_tags")


def test_scoped_policies_and_cleanup_without_model_files(tmp_path,monkeypatch):
    path=tmp_path/"policy.yml";path.write_text('version: 1\nexcluded_core_tags: [common]\nby_model: {wd: [special]}')
    policies=app.load_exclusion_policy(path,"excluded_core_tags")
    assert policies["wd"]==frozenset({"common","special"}) and policies["dtq"]==frozenset({"common"})
    stash=Stash();stash.marker(WD,"watermark","delete")
    monkeypatch.setattr(app,"create_stash_client",lambda _:stash)
    monkeypatch.setattr(app,"load_vocabulary",lambda _:pytest.fail("cleanup read vocabulary"))
    assert app.main(["--models","wd","--cleanup-markers-only","--offline"])==0
    assert not stash.scene["scene_markers"]


def test_vocabulary_order_and_categories(tmp_path):
    (tmp_path/"selected_tags.csv").write_text('tag_id,name,category\n999,rating,9\n10,tag,0\n1,character,4\n')
    assert load_vocabulary(ModelSettings(WD,tmp_path,download_missing=False)).labels==["rating","tag","character"]
    (tmp_path/"tag_to_id.json").write_text('{"copyright":2,"general":0,"character":1}')
    (tmp_path/"config.json").write_text('{"image_size":448,"num_classes":3}')
    (tmp_path/"tag_category.json").write_text('{"copyright":3,"general":0,"character":4}')
    assert load_vocabulary(ModelSettings(DTQ,tmp_path,download_missing=False)).labels==["general","character","copyright"]
    (tmp_path/"tag_to_id.json").write_text('{"a":0,"b":0,"c":2}')
    with pytest.raises(ValueError): load_vocabulary(ModelSettings(DTQ,tmp_path,download_missing=False))


@pytest.mark.parametrize("values",[np.array([[np.nan]]),np.array([[np.inf]]),np.array([[1.1]]),np.zeros((2,1))])
def test_bad_scores(values):
    with pytest.raises(RuntimeError): validate_scores(values,1,["tag"],"test")


def test_preprocessing_and_only_untagged(monkeypatch):
    from PIL import Image
    image=Image.new("RGB",(8,4),(255,0,0))
    wd,dtq=prepare_wd(image,8),prepare_dtq(image,8)
    np.testing.assert_array_equal(wd[4,4],[0,0,255])
    np.testing.assert_array_equal(wd[0,0],[255,255,255])
    np.testing.assert_allclose(dtq[:,0,0],-np.array([.485,.456,.406])/np.array([.229,.224,.225]),atol=1e-6)
    stash=Stash();stash.scene["tags"].append(stash.tag("jt:seen"));stash.marker(JOYTAG,"seen","jt")
    assert app.outputs_already_present(stash.scene,JOYTAG) and not app.outputs_already_present(stash.scene,WD)
    stash.scene["tags"].append(stash.tag("wd:seen"))
    assert not app.outputs_already_present(stash.scene,WD)
    monkeypatch.setattr(app,"CREATE_MARKERS",False)
    assert app.outputs_already_present(stash.scene,WD)


def test_dry_run_api_guard():
    client=app.create_stash_client(True)
    try:
        with pytest.raises(RuntimeError,match="mutation attempted"): client.gql("mutation Test { x }")
    finally: client.close()


@pytest.fixture
def oppai_files(tmp_path):
    labels=["<PAD>","<UNK>","watermark","rare_tag","<o>_<o>","rating:explicit"]
    mapping={name:i for i,name in enumerate(labels)}
    vocab={"tag_to_index":dict(reversed(list(mapping.items()))),
           "index_to_tag":{str(i):name for i,name in enumerate(labels)}}
    (tmp_path/"vocabulary.json").write_text(json.dumps(vocab))
    (tmp_path/"config.json").write_text(json.dumps({"format":"onnx","num_labels":len(labels),"image_size":448}))
    preprocessing={"image_size":448,"patch_size":16,"num_channels":3,"color_order":"RGB",
        "resize_mode":"letterbox","pad_color_rgb":[114]*3,"normalize_mean":[.5]*3,
        "normalize_std":[.5]*3,"input_dtype":"float32","input_layout":"BCHW"}
    (tmp_path/"preprocessing.json").write_text(json.dumps(preprocessing))
    (tmp_path/"selected_tags.csv").write_text("tag_id,name,category\n"+"".join(f"{i},{name},0\n" for i,name in enumerate(labels)))
    return tmp_path


def test_oppai_vocabulary_preserves_indices_and_real_tags(oppai_files):
    vocab=load_vocabulary(ModelSettings(OPPAI,oppai_files,download_missing=False))
    assert vocab.output_count==6 and vocab.output_indices==(2,3,4,5)
    assert vocab.labels==["watermark","rare_tag","<o>_<o>","rating:explicit"]
    assert set(vocab.categories)==set(vocab.labels)


@pytest.mark.parametrize("damage",["index","csv","count","resolution","normalization"])
def test_oppai_rejects_mismatched_assets(oppai_files,damage):
    if damage=="csv":
        (oppai_files/"selected_tags.csv").write_text("tag_id,name,category\n0,wrong,0\n")
    else:
        name={"index":"vocabulary.json","count":"config.json","resolution":"config.json",
              "normalization":"preprocessing.json"}[damage]
        data=json.loads((oppai_files/name).read_text())
        if damage=="index": data["tag_to_index"]["watermark"]=5
        elif damage=="count": data["num_labels"]=7
        elif damage=="resolution": data["image_size"]=320
        else: data["normalize_mean"]=[.485,.456,.406]
        (oppai_files/name).write_text(json.dumps(data))
    with pytest.raises(ValueError): load_vocabulary(ModelSettings(OPPAI,oppai_files,download_missing=False))


@pytest.mark.parametrize("shape,valid_shape",[((896,448),(448,224)),((448,448),(448,448)),
    ((64,32),(64,32)),((9,5),(9,5)),((1,1000),(1,448)),((1000,1),(448,1))])
def test_oppai_letterbox_mask_and_color(shape,valid_shape):
    from PIL import Image
    with Image.new("RGB",shape,(255,0,0)) as image:
        pixels,mask=prepare_oppai(image)
        assert image.getpixel((0,0))==(255,0,0)  # caller retains ownership
    width,height=valid_shape
    assert pixels.shape==(3,448,448) and pixels.dtype==np.float32
    assert mask.shape==(448,448) and mask.dtype==np.bool_
    assert np.count_nonzero(~mask)==width*height
    top,left=(448-height)//2,(448-width)//2
    assert not mask[top:top+height,left:left+width].any()
    np.testing.assert_array_equal(pixels[:,~mask],np.broadcast_to(np.array([1,-1,-1])[:,None],(3,width*height)))
    if mask.any(): np.testing.assert_allclose(pixels[:,mask],114/255*2-1,atol=1e-7)


def test_oppai_transparency_and_exif():
    from PIL import Image
    with Image.new("RGBA",(448,448),(255,0,0,0)) as image:
        pixels,mask=prepare_oppai(image)
        assert not mask.any()  # transparent source pixels are still image pixels
        np.testing.assert_allclose(pixels,114/255*2-1,atol=1e-7)
    with Image.new("RGB",(12,6),(0,0,255)) as image:
        image.getexif()[274]=6
        _,mask=prepare_oppai(image)
        y,x=np.where(~mask)
        assert (x.max()-x.min()+1,y.max()-y.min()+1)==(6,12)


@pytest.mark.parametrize("source",[WD,DTQ,OPPAI])
def test_onnx_batches_activation_and_output_mapping(source,tmp_path,monkeypatch):
    import sys
    from PIL import Image
    captures=[]
    count=6 if source==OPPAI else 4
    labels=["a","b","c","d"]
    vocabulary=Vocabulary(labels,{},448,count,(2,3,4,5) if source==OPPAI else None)
    fixed_batch=3
    def value(name,shape,dtype="tensor(float)"):
        return SimpleNamespace(name=name,shape=shape,type=dtype)
    class Session:
        def __init__(self,path,**kwargs): captures.append(("options",kwargs))
        def get_providers(self): return ["CPUExecutionProvider"]
        def get_inputs(self):
            if source==OPPAI:
                return [value("padding_mask",[fixed_batch,448,448],"tensor(bool)"),
                        value("pixel_values",[fixed_batch,3,448,448])]
            return [value("image",[fixed_batch,448,448,3] if source==WD else [fixed_batch,3,448,448])]
        def get_outputs(self): return [value("probabilities" if source==OPPAI else "output",[fixed_batch,count])]
        def run(self,names,feed):
            captures.append(("feed",feed))
            row=np.array([.99,.98,.1,.3,.7,.9] if source==OPPAI else [-2,0,1,2] if source==DTQ else [.1,.3,.7,.9],np.float32)
            return [np.broadcast_to(row,(fixed_batch,count)).copy()]
    ort=SimpleNamespace(InferenceSession=Session,get_available_providers=lambda:["CPUExecutionProvider"],
        SessionOptions=SimpleNamespace,GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1,ORT_DISABLE_ALL=0))
    monkeypatch.setitem(sys.modules,"onnxruntime",ort)
    (tmp_path/"model.onnx").touch()
    tagger=OnnxTagger(ModelSettings(source,tmp_path,batch_size=4,device="cpu"),vocabulary)
    with Image.new("RGB",(64,32),"red") as a, Image.new("RGB",(32,64),"blue") as b:
        result=tagger.predict_batch([tagger.prepare(a),tagger.prepare(b)])
    assert result.shape==(2,4)
    expected=1/(1+np.exp(-np.array([-2,0,1,2]))) if source==DTQ else [.1,.3,.7,.9]
    np.testing.assert_allclose(result,np.broadcast_to(expected,(2,4)),atol=1e-7)
    feed=captures[-1][1]
    assert all(len(item)==fixed_batch for item in feed.values())
    for item in feed.values(): np.testing.assert_array_equal(item[-1],item[-2])
    if source==OPPAI:
        assert set(feed)=={"pixel_values","padding_mask"}
        assert feed["padding_mask"].dtype==np.bool_
        assert not np.array_equal(feed["padding_mask"][0],feed["padding_mask"][1])
        assert captures[0][1]["sess_options"].graph_optimization_level==1
    tagger.close()


def test_oppai_modes_and_old_config_compatibility(tmp_path,oppai_files,monkeypatch):
    old=tmp_path/"models.yml"
    old.write_text("version: 1\nenabled_models: [joytag]\n")
    monkeypatch.setenv("OPPAI_MODEL_PATH",str(oppai_files))
    settings=app.load_model_settings(old,["oppai"],None,True)[0]
    assert settings.source.prefix=="op:" and settings.model_dir==oppai_files
    assert settings.scene_threshold==.80 and settings.marker_threshold==.85
    monkeypatch.setattr(app,"create_stash_client",lambda _:pytest.fail("model-only mode contacted Stash"))
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("model-only mode loaded weights"))
    hierarchy_path=tmp_path/"fixture_hierarchy.yml"
    hierarchy_path.write_text(yaml.safe_dump({"version":1,"root":OPPAI.title,
        "category_prefix":"op:category:","groups":{"Fixture":load_vocabulary(settings).labels}}))
    assert app.main(["--model-config",str(old),"--models","oppai","--validate-hierarchy",
                     "--oppai-hierarchy",str(hierarchy_path),"--offline"])==0
    downloads=[]
    monkeypatch.setattr(app,"ensure_files",lambda config,names:downloads.append((config.source,names)))
    assert app.main(["--model-config",str(old),"--models","oppai","--download-models"])==0
    assert downloads==[(OPPAI,("model.onnx",))]


def test_oppai_scoped_blacklist_and_cleanup_without_assets(tmp_path,monkeypatch):
    policy=tmp_path/"cores.yml"
    policy.write_text("version: 1\nexcluded_core_tags: [common]\nby_model:\n  oppai: [special]\n")
    policies=app.load_exclusion_policy(policy,"excluded_core_tags")
    assert policies["oppai"]==frozenset({"common","special"})
    assert policies["wd"]==frozenset({"common"})
    stash=Stash();stash.marker(OPPAI,"special","delete");stash.marker(OPPAI,"keep","keep")
    stash.marker(WD,"special","other");stash.marker(OPPAI,"special","manual",True)
    monkeypatch.setattr(app,"create_stash_client",lambda _:stash)
    monkeypatch.setattr(app,"load_vocabulary",lambda _:pytest.fail("cleanup accessed model files"))
    assert app.main(["--models","oppai","--cleanup-markers-only","--marker-core-exclusions",str(policy),"--offline"])==0
    assert {m["id"] for m in stash.scene["scene_markers"]}=={"keep","other","manual"}
    policy.write_text('version: 1\nexcluded_core_tags: ["op:special"]\n')
    with pytest.raises(ValueError): app.load_exclusion_policy(policy,"excluded_core_tags")


# Independently measured from the pinned upstream vocabularies, excluding only
# Oppai's PAD/UNK slots. The default files must contain those exact label sets.
BUNDLED_VOCABULARIES = {
    "joytag": (5813,"ad82c7dd5746bffdb3de832c44dc348af74cbc240a0c17960d068b4eccb8b8f3"),
    "wd": (10861,"b350f44fd5f5f96414af4596210fd669b3018f89a98deb87776fcfb19800ce7a"),
    "dtq": (11424,"6993dd6fe83114e3c652fdfd76a6264810e4bf0b25c67d1e21e5674b9869fb6d"),
    "oppai": (19292,"f8a73952c61f079e7972aec6612ce182c791e2fba99d2e9afaa9e0ebabf9c980"),
}


@pytest.fixture(scope="module")
def bundled_catalogs():
    root=Path(__file__).resolve().parents[1]
    result={}
    for key,(count,digest) in BUNDLED_VOCABULARIES.items():
        path=root/f"{key}_hierarchy.yml"
        data=yaml.safe_load(path.read_text(encoding="utf-8"))
        labels=sorted({label for group in data["groups"].values() for label in group})
        assert len(labels)==count and app.vocabulary_digest(labels)==digest
        hierarchy=app.load_tag_hierarchy(path,labels,SOURCES[key])
        assert hierarchy.category_prefix==SOURCES[key].prefix+"category:"
        result[key]=(Vocabulary(labels,{}),hierarchy)
    return result


def test_default_hierarchies_cover_pinned_vocabularies_and_validate_without_weights(bundled_catalogs,monkeypatch,tmp_path):
    # No explicit hierarchy flags: verify that all four CLI defaults load from
    # beside the script even when launched in another working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app,"load_vocabulary",lambda settings:bundled_catalogs[settings.source.key][0])
    monkeypatch.setattr(app,"create_stash_client",lambda _:pytest.fail("validation contacted Stash"))
    monkeypatch.setattr(app,"load_tagger",lambda *args:pytest.fail("validation loaded weights"))
    assert app.main(["--models","all","--validate-hierarchy","--offline"])==0
    joy= bundled_catalogs["joytag"][1]
    for source in (WD,DTQ,OPPAI):
        hierarchy=bundled_catalogs[source.key][1]
        for label in set(joy.leaf_parents)&set(hierarchy.leaf_parents):
            assert hierarchy.leaf_parents[label]==tuple(
                source.prefix+parent[len(JOYTAG.prefix):] for parent in joy.leaf_parents[label])
        # The fallback is an ordinary parent: its leaves remain in the vocabulary.
        assert any(parent.endswith("Other / Unclassified")
                   for parents in hierarchy.leaf_parents.values() for parent in parents)


@pytest.mark.parametrize("source",[WD,DTQ,OPPAI])
def test_bundled_hierarchy_migrates_old_metadata_parents_idempotently(bundled_catalogs,source):
    stash=Stash()
    manual=stash.tag("my manual group")
    foreign=stash.tag("jt:category:General");foreign["description"]=JOYTAG.hierarchy_description
    old=stash.tag(source.prefix+"category:General");old["description"]=source.hierarchy_description
    leaf=stash.tag(source.prefix+"blue_hair");leaf["parents"]=[manual,foreign,old]
    hierarchy=bundled_catalogs[source.key][1]
    resolver=app.TagResolver(stash,hierarchy,source=source)
    resolver.resolve(leaf["name"])
    assert {parent["name"] for parent in leaf["parents"]}=={
        manual["name"],foreign["name"],source.prefix+"category:Appearance / Hair / Color"}
    assert old["name"] in stash.tags
    before=deepcopy(stash.writes)
    app.TagResolver(stash,hierarchy,source=source).resolve(leaf["name"])
    assert stash.writes==before
