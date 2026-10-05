"""Temporal qualification, exclusions, source votes and exact checkpoint resume."""
from tests.helpers.temporal import threshold_policy


from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stash_ai.domain import Cue, FramePrediction
from stash_ai.inference import prediction
from stash_ai.cleanup import evidence as cleanup_evidence
from stash_ai.publication import merged as merge_sync
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
import stash_ai.evidence.source_markers as merge_source_markers
import stash_ai.storage.store as merge_store
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.evidence.temporal.statistics import SceneAccumulator, scene_reason
from stash_ai.evidence.temporal.markers import consensus_seeds
from stash_ai.models import SOURCES
from tests.helpers.inference import IndexedTagger


from tests.helpers.evidence import frames
from tests.helpers.evidence import segment
pytestmark = pytest.mark.usefixtures("reset_temporal")


def test_extras_recur_across_entire_marker_and_first_frame_noise_is_removed():
    data=frames([[('core',.9),('brief',.9)], [('core',.9),('later',.9)],
                 [('core',.9),('later',.9)], [('core',.9)]])
    markers,_=app.build_markers(data,frozenset({'brief','later'}),threshold=.8)
    assert len(markers)==1 and markers[0].extra_tags==('later',)
    assert markers[0].extra_evidence==(('later',2,.5),)


def test_core_exclusion_remains_eligible_as_extra():
    data=frames([[('core',.9),('excluded',.9)]]*3)
    markers,_=app.build_markers(data,frozenset({'excluded'}),threshold=.8)
    assert [m.tag for m in markers]==['core'] and markers[0].extra_tags==('excluded',)


def test_extra_coverage_is_duration_weighted_and_overlaps_not_double_counted(monkeypatch):
    data=[FramePrediction(Cue(0,10,'',(0,0,1,1)),(('core',.9),)),
          FramePrediction(Cue(10,11,'',(0,0,1,1)),(('core',.9),('brief',.9))),
          FramePrediction(Cue(10.5,11.5,'',(0,0,1,1)),(('core',.9),('brief',.9)))]
    markers,_=app.build_markers(data,frozenset({'brief'}),threshold=.8)
    assert markers[0].extra_tags==()
    monkeypatch.setattr(app,'TEMPORAL_POLICY',replace(app.TEMPORAL_POLICY,extra_min_fraction=0))
    markers,_=app.build_markers(data,frozenset({'brief'}),threshold=.8)
    assert markers[0].extra_evidence[0][2]==pytest.approx(1.5/11.5)


def test_extras_include_observations_inside_a_permitted_primary_gap(monkeypatch):
    monkeypatch.setattr(app,'MARKER_MERGE_GAP_SECONDS',2)
    data=frames([[('core',.9)],[('extra',.9)],[('extra',.9)],[('core',.9)]])
    markers,_=app.build_markers(data,frozenset({'extra'}),threshold=.8)
    assert len(markers)==1 and markers[0].extra_evidence==(('extra',2,.5),)


def test_merged_extra_coverage_uses_the_whole_union_duration():
    records={'wd':[segment('wd',[[('core',.9),('extra',.9)]]*2)],
             'dtq':[segment('dtq',[[('core',.9)]]*20)]}
    spans=merge_source_markers.marker_contributions(app,records,{m:frozenset({'extra'}) for m in SOURCES})
    result=merge_intervals.union_intervals(spans)[0]
    assert result['end']==20 and result['extra_tags']==[]
    assert result['extra_evidence']['extra']['coverage_fraction']==.1


@pytest.mark.parametrize('same_time',[False,True])
def test_merged_extra_hits_combine_distinct_times_but_not_duplicate_model_votes(same_time):
    records={'wd':[segment('wd',[[('core',.9),('extra',.9)],[('core',.9)]])],
             'dtq':[segment('dtq',[[('core',.9)],[('core',.9),('extra',.9)]])]} 
    if same_time:records['dtq']=[segment('dtq',[[('core',.9),('extra',.9)],[('core',.9)]])]
    spans=merge_source_markers.marker_contributions(app,records,{m:frozenset({'extra'}) for m in SOURCES})
    assert all(span['extra_tags']==([] if same_time else ['extra']) for span in spans)
    result=merge_intervals.union_intervals(spans)[0]
    assert result['extra_tags']==([] if same_time else ['extra'])
    assert result['extra_evidence']['extra']['support_seconds']==(1 if same_time else 2)


def test_unproved_extra_tags_do_not_join_temporal_evidence():
    records={'wd':[segment('wd',[[('core',.9)]]*2)],'joytag':[
        dict(mode='spans',meta={'origin':'legacy'},items=[dict(tag='core',start=0,end=2,extra_tags=['old_extra'])])]}
    spans=merge_source_markers.marker_contributions(app,records,{m:frozenset() for m in SOURCES})
    assert merge_intervals.union_intervals(spans)[0]['extra_tags']==[]


def test_hysteresis_needs_a_seed_and_caps_weak_tail(monkeypatch):
    monkeypatch.setattr(app,'TEMPORAL_POLICY',replace(app.TEMPORAL_POLICY,marker_max_weak_seconds=2.5))
    data=frames([[('core',.7)]]*2+[[('core',.9)]]+[[('core',.7)]]*8)
    markers,_=app.build_markers(data,frozenset(),threshold=.8)
    assert len(markers)==1
    assert (markers[0].start,markers[0].end)==(2,5.5)
    assert markers[0].strong_hits==1


def test_hysteresis_does_not_bridge_missing_detections_without_allowed_gap():
    data=frames([[('core',.9)],[('core',.7)],[],[('core',.9)],[('core',.7)]])
    markers,_=app.build_markers(data,frozenset(),threshold=.8)
    assert [(m.start,m.end) for m in markers]==[(0,2),(3,5)]


def test_new_strong_hit_resets_weak_tail_limit(monkeypatch):
    monkeypatch.setattr(app,'TEMPORAL_POLICY',replace(app.TEMPORAL_POLICY,marker_max_weak_seconds=1))
    data=frames([[('core',s)] for s in [.9,.7,.9,.7,.7]])
    markers,_=app.build_markers(data,frozenset(),threshold=.8)
    assert [(m.start,m.end) for m in markers]==[(0,4)]


def test_consensus_promotes_repeated_weak_detections_and_preserves_source_scores():
    records={m:[segment(m,[[('core',.7)]]*3)] for m in ('wd','dtq')}
    policies={m:frozenset() for m in SOURCES}
    spans=merge_source_markers.marker_contributions(app,records,policies)
    result=merge_intervals.union_intervals(spans)
    assert len(result)==1 and result[0]['sources']==['wd','dtq']
    assert (result[0]['start'],result[0]['end'])==(0,3)
    assert all(c['score']==.7 for c in result[0]['contributors'])
    assert 'score' not in result[0]


@pytest.mark.parametrize('difference',['input','time','same_model','exclusion'])
def test_consensus_rejects_incompatible_or_duplicate_votes(difference):
    rows=[[('core',.7)]]*3
    records={m:[segment(m,rows)] for m in ('wd','dtq')}
    policies={m:frozenset() for m in SOURCES}
    if difference=='input':records['dtq'][0]['meta']['frame_input_signature']='different'
    if difference=='time':
        for f in records['dtq'][0]['items']:f['start']+=10;f['end']+=10
    if difference=='same_model':records={'wd':[records['wd'][0],deepcopy(records['wd'][0])]}
    if difference=='exclusion':policies['dtq']=frozenset({'core'})
    assert not merge_source_markers.marker_contributions(app,records,policies)


def test_single_model_unique_strong_tag_does_not_need_consensus():
    records={'oppai':[segment('oppai',[[('unique',.9)]]*2)]}
    spans=merge_source_markers.marker_contributions(app,records,{m:frozenset() for m in SOURCES})
    assert len(spans)==1 and spans[0]['tag']=='unique' and spans[0]['support_sources']==[]


@pytest.mark.parametrize('values,expected',[
    ([.95,0,0,0,0], 'strong_peak'),
    ([.65,.65,0,0,0], 'repeated_high'),
    ([.55,.55,.55,.55,0], 'persistent_support'),
    ([.55,0,0,0,.55], None),
])
def test_scene_rules_accept_short_strong_or_repeated_support(values,expected):
    policy=app.TEMPORAL_POLICY
    data=frames([[]]*len(values))
    accumulator=SceneAccumulator(.6,policy)
    accumulator.add(np.array(values,np.float32).reshape(-1,1),[f.cue for f in data])
    stats=accumulator.statistics(['tag']).get('tag')
    score=float(np.mean(sorted(values,reverse=True)[:2]))
    assert scene_reason(score,stats,.6,policy)==expected


def test_scene_coverage_and_counter_snapshot_do_not_double_count_or_mutate():
    accumulator=SceneAccumulator(.5,app.TEMPORAL_POLICY)
    accumulator.add(np.array([[.8],[.8]],np.float32),[Cue(0,2,'',(0,0,1,1)),Cue(1,3,'',(0,0,1,1))])
    state=accumulator.state()
    accumulator.add(np.array([[.8]],np.float32),[Cue(3,4,'',(0,0,1,1))])
    resumed=SceneAccumulator(.5,app.TEMPORAL_POLICY,state)
    stats=resumed.statistics(['tag'])['tag']
    assert stats['hits']==2 and stats['sampled_seconds']==stats['support_seconds']==3
    assert stats['coverage_fraction']==1


def test_scene_run_breaks_on_a_negative_frame_even_with_overlapping_cues():
    accumulator=SceneAccumulator(.5,app.TEMPORAL_POLICY)
    cues=[Cue(0,10,'',(0,0,1,1)),Cue(1,2,'',(0,0,1,1)),Cue(2,3,'',(0,0,1,1))]
    accumulator.add(np.array([[.8],[0],[.8]],np.float32),cues)
    assert accumulator.statistics(['tag'])['tag']['longest_hits']==1


@pytest.mark.parametrize('aggregation,top_k',[('max',2),('topk_mean',2),('topk_mean',5)])
@pytest.mark.parametrize('prefetch',[0,1])
def test_temporal_resume_preserves_statistics_and_all_sparse_scores(pipeline,monkeypatch,aggregation,top_k,prefetch):
    p=pipeline
    monkeypatch.setattr(app,'AGGREGATION',aggregation);monkeypatch.setattr(app,'TOP_K_FRAMES',top_k)
    p.fail['wd']=4
    with pytest.raises(KeyboardInterrupt):app.main([*p.args,'--prefetch-batches',str(prefetch)])
    p.fail.clear()
    assert app.main(p.args)==0
    assert p.loaded[-1].seen==list(range(4,10))
    expected=IndexedTagger(p.loaded[-1].settings,p.scores['wd'],p.vocab.labels)
    analysis={}
    predictions,output=prediction.predict_scene(app, expected,p.stash,{},p.cues,analysis=analysis)
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        scene=state.records('1','scene')['wd'][-1]
        markers=state.records('1','markers')['wd'][-1]
        assert [(r['tag'],r['score']) for r in scene['items']]==predictions
        assert {r['tag']:r['statistics'] for r in scene['items']}==analysis['scene_statistics']
        assert [r['tags'] for r in markers['items']]==[[list(pair) for pair in f.tags] for f in output]
        assert state.connection.execute('SELECT COUNT(*) FROM checkpoint_analysis').fetchone()[0]==0


@pytest.mark.parametrize('mode',['--markers-only','--scene-tags-only'])
def test_temporal_partial_resume_reuses_completed_results(pipeline,monkeypatch,mode):
    p=pipeline
    assert app.main([*p.args,mode])==0
    monkeypatch.setattr(app.services,'load_tagger',lambda *args:pytest.fail('completed partial model loaded'))
    assert app.main([*p.args,mode])==0


def test_temporal_dry_run_preserves_database(pipeline):
    p=pipeline
    assert app.main(p.args)==0
    before=(p.path.read_bytes(),deepcopy(p.stash.scene))
    p.stash.dry_run=True
    assert app.main([*p.args,'--restart','--dry-run'])==0
    assert before==(p.path.read_bytes(),p.stash.scene)


def test_cleanup_revokes_consensus_votes_without_restoring_manual_deletions(tmp_path):
    from tests.helpers.stash import Stash
    stash=Stash();policies={m:frozenset() for m in SOURCES}
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for model in ('wd','dtq','oppai'):
            state.put('1',model,{'markers':segment(model,[[('core',.7)]]*3)})
        hits,spans=merge.evidence(app,state,'1',policies,policies)
        merge_sync.sync_merged(app,stash,app.services.TagResolver(stash,source=merge_labels.AI),state,stash.find_scene('1'),hits,spans)
        policies['wd']=frozenset({'core'})
        kept=[s for s in spans if s['source']!='wd']
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',kept,{'wd'},policies)
        assert merge_intervals.union_intervals(cleaned)[0]['sources']==['dtq','oppai']
        policies['dtq']=frozenset({'core'})
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',[s for s in cleaned if s['source']!='dtq'],{'dtq'},policies)
        assert cleaned==[]


@pytest.mark.parametrize('change',['new_model','scores'])
def test_cleanup_uses_published_evidence_and_does_not_adopt_new_votes(tmp_path,change):
    policies={m:frozenset() for m in SOURCES}
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for model in ('wd','dtq'):
            state.put('1',model,{'markers':segment(model,[[('core',.7)]]*3)})
        _,spans=merge.evidence(app,state,'1',policies,policies)
        policies['wd']=frozenset({'core'})
        kept=[span for span in spans if span['source']=='dtq']
        if change=='new_model':
            state.put('1','oppai',{'markers':segment('oppai',[[('core',.7)]]*3)})
            assert cleanup_evidence.cleanup_support(app,state,'1',kept,{'wd'},policies)==[]
        else:
            altered=segment('wd',[[('core',.7)]]*3)
            altered['items'][0]['tags']=[('core',.71)]
            state.put('1','wd',{'markers':altered})
            with pytest.raises(RuntimeError,match='merge-only'):
                cleanup_evidence.cleanup_support(app,state,'1',kept,{'wd'},policies)


@pytest.mark.parametrize('field,value',[('extra_min_hits',True),('marker_continue_ratio',0),
    ('consensus_min_models',1),('marker_max_weak_seconds',float('nan')),('scene_repeat_fraction',2)])
def test_temporal_policy_rejects_invalid_settings(field,value):
    with pytest.raises(ValueError):replace(app.TEMPORAL_POLICY,**{field:value}).validate()


def test_old_cached_results_still_merge_without_new_evidence(tmp_path):
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        state.put('1','wd',{'scene':dict(mode='scene',meta={'origin':'legacy'},items=[{'tag':'legacy','score':None}]),
                           'markers':dict(mode='spans',meta={'origin':'legacy'},items=[dict(tag='legacy',start=0,end=10,extra_tags=[])])})
        hits,spans=merge.evidence(app,state,'1',{m:frozenset() for m in SOURCES},{m:frozenset() for m in SOURCES})
        assert hits[0]['tag']=='legacy' and spans[0]['end']==10
