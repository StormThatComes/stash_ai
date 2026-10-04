"""Shared fixtures, loaded lazily so updater tests need only pytest.

Reset profiles are requested by each test module through usefixtures markers.
They intentionally preserve the former per-module autouse behavior.
"""

import pytest


@pytest.fixture
def reset_multimodel(monkeypatch):
    from tests.helpers.runtime import runtime as app
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    monkeypatch.setattr(app,"DEFAULT_TEMPORAL",False)
    monkeypatch.setattr(app,"TEMPORAL_POLICY",TemporalPolicy(enabled=False))
    # Preserve regression coverage of the explicitly available separate mode.
    monkeypatch.setattr(app,"DEFAULT_OUTPUT_MODE","separate")
    monkeypatch.setattr(app,"DEFAULT_RESUME",False)
    monkeypatch.setattr(app,"PREFETCH_BATCHES",0)
    for key, value in dict(DRY_RUN=False, APPLY_SCENE_TAGS=True, CREATE_MARKERS=True,
        CREATE_MISSING=True, MAX_TAGS=0, MAX_MARKERS_PER_SCENE=0,
        MARKER_MAX_TAGS_PER_FRAME=0, MARKER_MIN_HITS=2, MARKER_MERGE_GAP_SECONDS=0.0,
        AGGREGATION="topk_mean", TOP_K_FRAMES=2).items():
        monkeypatch.setattr(app,key,value)


@pytest.fixture
def reset_merge(monkeypatch, tmp_path):
    from tests.helpers.runtime import runtime as app
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    monkeypatch.setattr(app,"DEFAULT_TEMPORAL",False)
    monkeypatch.setattr(app,"TEMPORAL_POLICY",TemporalPolicy(enabled=False))
    for key, value in dict(DEFAULT_OUTPUT_MODE="merged", DEFAULT_RESUME=False, PREFETCH_BATCHES=0, MERGE_STATE_FILE=tmp_path/"sources.sqlite3",
        DRY_RUN=False, APPLY_SCENE_TAGS=True, CREATE_MARKERS=True, CREATE_MISSING=True,
        MAX_TAGS=0, MAX_MARKERS_PER_SCENE=0, MARKER_MAX_TAGS_PER_FRAME=0, MARKER_MIN_HITS=2,
        MARKER_MERGE_GAP_SECONDS=0.0, AGGREGATION="topk_mean", TOP_K_FRAMES=2).items():
        monkeypatch.setattr(app,key,value)


@pytest.fixture
def reset_progress(monkeypatch,tmp_path):
    from tests.helpers.runtime import runtime as app
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    monkeypatch.setattr(app,"DEFAULT_TEMPORAL",False)
    monkeypatch.setattr(app,"TEMPORAL_POLICY",TemporalPolicy(enabled=False))
    for key,value in dict(DEFAULT_OUTPUT_MODE="merged",DEFAULT_RESUME=True,PREFETCH_BATCHES=0,
        MERGE_STATE_FILE=tmp_path/"state.sqlite3",DRY_RUN=False,APPLY_SCENE_TAGS=True,CREATE_MARKERS=True,
        CREATE_MISSING=True,MAX_TAGS=0,MAX_MARKERS_PER_SCENE=0,MARKER_MAX_TAGS_PER_FRAME=0,MARKER_MIN_HITS=2,
        MARKER_MERGE_GAP_SECONDS=0.0,AGGREGATION="topk_mean",TOP_K_FRAMES=2).items(): monkeypatch.setattr(app,key,value)


@pytest.fixture
def reset_temporal(monkeypatch, tmp_path):
    from tests.helpers.runtime import runtime as app
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    for key, value in dict(DEFAULT_OUTPUT_MODE="merged", DEFAULT_RESUME=True, DEFAULT_TEMPORAL=True,
        TEMPORAL_POLICY=TemporalPolicy(), PREFETCH_BATCHES=0,
        MERGE_STATE_FILE=tmp_path/"state.sqlite3", DRY_RUN=False, APPLY_SCENE_TAGS=True, CREATE_MARKERS=True,
        CREATE_MISSING=True, MAX_TAGS=0, MAX_MARKERS_PER_SCENE=0, MARKER_MAX_TAGS_PER_FRAME=0, MARKER_MIN_HITS=2,
        MARKER_MERGE_GAP_SECONDS=0.0, AGGREGATION="topk_mean", TOP_K_FRAMES=2).items():
        monkeypatch.setattr(app, key, value)


@pytest.fixture
def pipeline(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from tests.helpers.runtime import runtime as app
    from stash_ai.inference import sprites
    from stash_ai.domain import Cue
    from stash_ai.models import SOURCES, Vocabulary
    from tests.helpers.stash import Stash
    from tests.helpers.inference import IndexedTagger, sprite_frames
    stash=Stash()
    cues=[Cue(i,i+1,"sprite",(0,0,1,1)) for i in range(10)]
    vocab=Vocabulary(["common","extra"],{"common":"General","extra":"General"})
    values=np.array([[.8,.1],[.9,.3],[.2,.8],[.8,.9],[.9,.7],[.3,.9],[.9,.9],[.8,.2],[.9,.8],[.4,.9]],np.float32)
    context=SimpleNamespace(stash=stash,cues=cues,vocab=vocab,scores={key:values.copy() for key in SOURCES},
                            loaded=[],fail={},path=tmp_path/"state.sqlite3")
    monkeypatch.setattr(app.services,"create_stash_client",lambda _:stash)
    monkeypatch.setattr(app.services,"load_vocabulary",lambda _:vocab)
    monkeypatch.setattr(sprites,"fetch_cues",lambda *args:(context.cues,len(context.cues)))
    monkeypatch.setattr(sprites,"iter_sprite_frames",lambda stash,paths,cues:sprite_frames(cues))
    def load(settings,*args):
        assert all(model.closed for model in context.loaded)
        model=IndexedTagger(settings,context.scores[settings.source.key],vocab.labels,context.fail.get(settings.source.key))
        context.loaded.append(model)
        return model
    monkeypatch.setattr(app.services,"load_tagger",load)
    context.args=["--models","wd","--no-hierarchy","--scene-id","1","--state-db",str(context.path)]
    return context


@pytest.fixture
def merge_pipeline(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from tests.helpers.runtime import runtime as app
    from stash_ai.inference import sprites
    from stash_ai.domain import Cue
    from stash_ai.models import SOURCES, Vocabulary
    from tests.helpers.stash import Stash
    from tests.helpers.inference import Tagger, sprite_frames
    stash=Stash();cues=[Cue(i,i+10,"sprite",(0,0,1,1)) for i in [10,20,30,40,70,80]]
    labels=["common","unique_op","extra"]
    vocabulary=Vocabulary(labels,{tag:"General" for tag in labels})
    scores={key:np.zeros((6,3),np.float32) for key in SOURCES}
    for key,indices in {"joytag":[0,1],"wd":[1,2],"oppai":[2,3],"dtq":[4,5]}.items():
        scores[key][indices,0]=.9
    # Keep Oppai common safely above its 0.85 marker threshold but below
    # unique_op (.92), preserving tests that exercise scene-tag ranking caps.
    scores["oppai"][2:4,0]=.91
    scores["oppai"][:2,1]=.92
    monkeypatch.setattr(app.services,"create_stash_client",lambda _:stash)
    monkeypatch.setattr(app.services,"load_vocabulary",lambda _:vocabulary)
    monkeypatch.setattr(sprites,"fetch_cues",lambda *args:(cues,len(cues)))
    monkeypatch.setattr(sprites,"iter_sprite_frames",lambda *args:sprite_frames(cues))
    loaded=[]
    def load(settings,*args):
        assert all(model.closed for model in loaded)
        model=Tagger(settings,labels,scores[settings.source.key]);loaded.append(model);return model
    monkeypatch.setattr(app.services,"load_tagger",load)
    args=["--models","all","--no-hierarchy","--state-db",str(tmp_path/"sources.sqlite3"),"--scene-id","1"]
    return SimpleNamespace(stash=stash,scores=scores,loaded=loaded,args=args,path=tmp_path/"sources.sqlite3")


@pytest.fixture
def supported_stash_version(monkeypatch):
    """Allow real clients to initialize without network access in transport tests."""
    from types import SimpleNamespace
    import requests

    def post(self, url, **kwargs):
        assert kwargs['json']['query'] == 'query StashAIVersion { version { version } }'
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'data': {'version': {'version': 'v0.31.1'}}})

    monkeypatch.setattr(requests.Session, 'post', post)

