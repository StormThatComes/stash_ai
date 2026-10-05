"""Shared VTT evidence, extra-only models, published-source cleanup and cache reuse."""
from stash_ai.db.snapshot_migration import complete_policy

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from stash_ai.inference import sprites
from stash_ai.cleanup import evidence as cleanup_evidence
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.source_markers as merge_source_markers
import stash_ai.storage.store as merge_store
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.models import SOURCES
from tests.helpers.evidence import segment


pytestmark = pytest.mark.usefixtures("reset_temporal")


def records_for(values, tag='core'):
    return {source: [segment(source, [[(tag, value)] if value else [] for value in scores])]
            for source, scores in values.items()}


def policies(excluded=()):
    return {source: frozenset(excluded) for source in SOURCES}


def fused(records, exclusions=None):
    spans=merge_source_markers.marker_contributions(app, records, exclusions or policies())
    return spans, merge_intervals.union_intervals(spans)


def test_alternating_strong_hits_form_one_marker_with_original_source_scores():
    spans, markers=fused(records_for({'wd':[.91,0,.91,0],'dtq':[0,.95,0,.95]}))
    assert [(m['start'],m['end'],m['sources']) for m in markers]==[(0,4,['wd','dtq'])]
    shared=spans[0]
    assert shared['derived']=='shared_frames' and shared['shared_hits']==4
    assert shared['source_evidence']['wd']['score']==.91 and shared['source_evidence']['dtq']['score']==.95
    assert shared['source_evidence']['wd']['intervals']==[[0,1],[2,3]]
    assert shared['source_evidence']['dtq']['hits']==2
    assert 'score' not in markers[0]


@pytest.mark.parametrize('difference',['identity','timeline','duplicate_passes','weak_only'])
def test_shared_evidence_does_not_invent_compatible_frames_or_strong_hits(difference):
    records=records_for({'wd':[.9,0],'dtq':[0,.9]})
    if difference=='identity':records['dtq'][0]['meta']['frame_input_signature']='other'
    if difference=='timeline':
        for row in records['dtq'][0]['items']:row['start']+=10;row['end']+=10
    if difference=='duplicate_passes':
        records=records_for({'wd':[.9],'dtq':[.9]})
        for source in records:records[source].append(deepcopy(records[source][0]))
    if difference=='weak_only':records=records_for({'wd':[.7,0,.7,0],'dtq':[0,.7,0,.7]})
    assert fused(records)[1]==[]


def test_each_source_uses_its_own_threshold():
    records=records_for({'wd':[.6,0],'dtq':[0,.96]})
    records['wd'][0]['meta']['marker_threshold']=.5
    records['dtq'][0]['meta']['marker_threshold']=.95
    assert fused(records)[1][0]['end']==2


def test_joint_hysteresis_can_continue_in_another_model_with_a_bounded_tail(monkeypatch):
    monkeypatch.setattr(app,'TEMPORAL_POLICY',replace(app.TEMPORAL_POLICY,marker_max_weak_seconds=1.5))
    spans,markers=fused(records_for({'wd':[.9,0,0,0,0],'dtq':[0,.7,.7,.7,.7]}))
    assert [(m['start'],m['end']) for m in markers]==[(0,2.5)]
    assert spans[0]['source_evidence']['dtq']['intervals']==[[1,2.5]]


@pytest.mark.parametrize('gap,expected',[(0,[(0,2),(3,5)]),(1,[(0,5)])])
def test_joint_gaps_follow_the_existing_setting(monkeypatch,gap,expected):
    monkeypatch.setattr(app,'MARKER_MERGE_GAP_SECONDS',gap)
    _,markers=fused(records_for({'wd':[.9,0,0,.9,0],'dtq':[0,.9,0,0,.9]}))
    assert [(m['start'],m['end']) for m in markers]==expected


def test_joint_extension_preserves_original_qualified_ranges():
    spans,markers=fused(records_for({'wd':[.9,.9,.9,0],'dtq':[0,0,0,.9]}))
    assert [(m['start'],m['end']) for m in markers]==[(0,4)]
    assert any(s['source']=='wd' and not s.get('derived') and (s['start'],s['end'])==(0,3) for s in spans)


def test_no_redundant_shared_range_when_original_union_already_covers_it():
    spans,markers=fused(records_for({'wd':[.9,.9,0,0],'dtq':[0,0,.9,.9]}))
    assert len(spans)==2 and all(not s.get('derived') for s in spans)
    assert [(m['start'],m['end']) for m in markers]==[(0,4)]


def test_shared_candidate_limit_is_bounded_and_deterministic(monkeypatch):
    monkeypatch.setattr(app,'MAX_MARKERS_PER_SCENE',1)
    records={source:[segment(source,[[(tag,.9) for tag in ('b','a')] if i==offset else [] for i in range(2)])]
             for source,offset in [('wd',0),('dtq',1)]}
    assert [m['tag'] for m in fused(records)[1]]==['a']


@pytest.mark.parametrize('excluded_source',['wd','dtq'])
def test_core_exclusion_revokes_a_required_source(excluded_source):
    excluded=policies();excluded[excluded_source]=frozenset({'core'})
    assert fused(records_for({'wd':[.9,0],'dtq':[0,.9]}),excluded)[1]==[]


def test_old_cache_without_frame_identity_does_not_gain_shared_ranges():
    records=records_for({'wd':[.9,0],'dtq':[0,.9]})
    for segments in records.values():segments[0]['meta'].pop('frame_input_signature')
    with pytest.raises(ValueError, match='signatures'):
        fused(records)


def extra_records(values=(.9,.9,.9,.9)):
    return {'wd':[segment('wd',[[('core',.9)]]*len(values))],
            'dtq':[segment('dtq',[[('detail',score)] if score else [] for score in values])]}


def test_extra_only_model_adds_detail_with_separate_provenance():
    _,markers=fused(extra_records(),policies({'detail'}))
    marker=markers[0]
    assert marker['extra_tags']==['detail'] and marker['sources']==['wd']
    evidence=marker['extra_evidence']['detail']['source_evidence']
    assert set(evidence)=={'dtq'} and evidence['dtq']['score']==.9
    assert evidence['dtq']['support_seconds']==4


@pytest.mark.parametrize('difference',['identity','timeline','low_score','single_time'])
def test_extra_only_evidence_obeys_input_threshold_and_unique_hit_requirements(difference):
    records=extra_records()
    if difference=='identity':records['dtq'][0]['meta']['frame_input_signature']='other'
    if difference=='timeline':records['dtq'][0]['items'][0]['end']=.5
    if difference=='low_score':records=extra_records((.7,.7,.7,.7))
    if difference=='single_time':
        records=extra_records((.9,0,0,0));records['oppai']=[deepcopy(records['dtq'][0])]
    assert fused(records,policies({'detail'}))[1][0]['extra_tags']==[]


def test_extra_hits_can_come_from_different_models_at_different_times():
    records=extra_records((.9,0));records['oppai']=[segment('oppai',[[],[('detail',.97)]])]
    marker=fused(records,policies({'detail'}))[1][0]
    assert marker['extra_tags']==['detail']
    assert set(marker['extra_evidence']['detail']['source_evidence'])=={'dtq','oppai'}


def test_extra_coverage_is_checked_over_the_final_longest_union():
    records=extra_records((.9,.9)+((0,)*18))
    records['oppai']=[segment('oppai',[[('core',.9)]]*20)]
    records['wd'][0]['items'][2:]=[dict(start=i,end=i+1,tags=[]) for i in range(2,20)]
    marker=fused(records,policies({'detail'}))[1][0]
    assert marker['end']==20 and marker['extra_tags']==[]
    assert marker['extra_evidence']['detail']['coverage_fraction']==.1


def test_extra_model_can_supply_observations_inside_a_permitted_core_gap(monkeypatch):
    monkeypatch.setattr(app,'MARKER_MERGE_GAP_SECONDS',2)
    records=extra_records((0,.9,.9,0))
    records['wd'][0]['items'][1]['tags']=[];records['wd'][0]['items'][2]['tags']=[]
    assert fused(records,policies({'detail'}))[1][0]['extra_tags']==['detail']


def test_independent_switches_and_legacy_snapshot_defaults(monkeypatch):
    monkeypatch.setattr(app,'TEMPORAL_POLICY',replace(app.TEMPORAL_POLICY,shared_marker_frames=False))
    assert fused(records_for({'wd':[.9,0],'dtq':[0,.9]}))[1]==[]
    assert fused(extra_records(),policies({'detail'}))[1][0]['extra_tags']==['detail']
    monkeypatch.setattr(app,'TEMPORAL_POLICY',replace(app.TEMPORAL_POLICY,shared_marker_extras=False))
    assert fused(extra_records(),policies({'detail'}))[1][0]['extra_tags']==[]
    old=TemporalPolicy.from_snapshot(complete_policy({'enabled':True}))
    assert not old.shared_marker_frames and not old.shared_marker_extras


def test_cleanup_rebuilds_a_shared_range_when_its_anchor_source_is_excluded(tmp_path):
    records=records_for({'wd':[.9,0,0],'dtq':[0,.91,0],'oppai':[0,0,.92]})
    excluded=policies()
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for source,segments in records.items():state.put('1',source,{'markers':segments[0]})
        _,spans=merge.evidence(app,state,'1',excluded,policies())
        assert spans[0]['source']=='wd'
        excluded['wd']=frozenset({'core'})
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',spans,{'wd'},excluded)
        marker=merge_intervals.union_intervals(cleaned)[0]
        assert (marker['start'],marker['end'],marker['sources'])==(1,3,['dtq','oppai'])
        excluded['dtq']=frozenset({'core'})
        assert cleanup_evidence.cleanup_support(app,state,'1',cleaned,{'dtq'},excluded)==[]


def test_cleanup_does_not_adopt_new_unpublished_votes(tmp_path):
    records=records_for({'wd':[.9,0,0],'dtq':[0,.9,0]});excluded=policies()
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for source,segments in records.items():state.put('1',source,{'markers':segments[0]})
        _,spans=merge.evidence(app,state,'1',excluded,policies())
        state.put('1','oppai',{'markers':segment('oppai',[[],[],[('core',.9)]])})
        excluded['wd']=frozenset({'core'})
        assert cleanup_evidence.cleanup_support(app,state,'1',spans,{'wd'},excluded)==[]


def test_cleanup_clips_source_scores_and_extra_proofs_to_published_bounds(tmp_path):
    records=records_for({'wd':[.9,0,0,0,0], 'dtq':[0,.99,0,.91,0], 'oppai':[0,0,.92,0,.99]})
    records['joytag']=[segment('joytag',[[],[],[('detail',.81)],[('detail',.82)],[('detail',.99)]])]
    excluded=policies({'detail'})
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for source,segments in records.items():state.put('1',source,{'markers':segments[0]})
        _,spans=merge.evidence(app,state,'1',excluded,policies())
        # An already-published window bounds cleanup, even if rebuilding would
        # otherwise return a longer interval from the same frozen inputs.
        spans[0].update(start=2,end=4)
        excluded['wd']=frozenset({'core','detail'})
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',spans,{'wd'},excluded)
        assert len(cleaned)==1
        proof=cleaned[0]['source_evidence']
        assert (proof['dtq']['score'],proof['oppai']['score'])==(.91,.92)
        assert proof['dtq']['intervals']==[[3,4]] and proof['oppai']['intervals']==[[2,3]]
        assert proof['dtq']['hits']==proof['oppai']['hits']==1 and cleaned[0]['shared_hits']==2
        extra=merge_intervals.union_intervals(cleaned)[0]['extra_evidence']['detail']['source_evidence']['joytag']
        assert extra['score']==.82 and extra['support_seconds']==2 and extra['hits_at_least']==2


def test_cleanup_refuses_changed_required_frame_records(tmp_path):
    records=records_for({'wd':[.9,0,0], 'dtq':[0,.9,0], 'oppai':[0,0,.9]})
    excluded=policies()
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for source,segments in records.items():state.put('1',source,{'markers':segments[0]})
        _,spans=merge.evidence(app,state,'1',excluded,policies())
        changed=deepcopy(records['dtq'][0]);changed['items'][1]['tags']=[('core',.99)]
        state.put('1','dtq',{'markers':changed})
        excluded['wd']=frozenset({'core'})
        with pytest.raises(RuntimeError,match='sources changed'):
            cleanup_evidence.cleanup_support(app,state,'1',spans,{'wd'},excluded)


def test_cleanup_command_rebuilds_shared_marker_and_then_removes_last_support(pipeline,monkeypatch,tmp_path):
    p=pipeline
    for values in p.scores.values():values[:]=0
    for source,index in [('wd',0),('dtq',1),('oppai',2)]:p.scores[source][index,0]=.95
    args=['--models','wd','dtq','oppai',*p.args[2:]]
    assert app.main(args)==0
    assert [(m['seconds'],m['end_seconds']) for m in p.stash.scene['scene_markers']]==[(0,3)]
    monkeypatch.setattr(app.services,'load_tagger',lambda *args:pytest.fail('cleanup loaded a model'))
    core=tmp_path/'core.yml'
    core.write_text('version: 1\nexcluded_core_tags: []\nby_model:\n  wd: [common]\n')
    cleanup=[*args,'--cleanup-only','--marker-core-exclusions',str(core)]
    assert app.main(cleanup)==0
    assert [(m['seconds'],m['end_seconds']) for m in p.stash.scene['scene_markers']]==[(1,3)]
    assert '[dtq+op]' in p.stash.scene['scene_markers'][0]['title']
    p.stash.writes.clear()
    assert app.main(cleanup)==0 and not p.stash.writes
    core.write_text('version: 1\nexcluded_core_tags: []\nby_model:\n  wd: [common]\n  dtq: [common]\n')
    assert app.main(cleanup)==0 and not p.stash.scene['scene_markers']


def test_cleanup_requires_published_evidence_and_preserves_extras_when_only_core_is_excluded(tmp_path):
    excluded=policies({'detail'})
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for source,segments in extra_records().items():state.put('1',source,{'markers':segments[0]})
        _,spans=merge.evidence(app,state,'1',excluded,policies())
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',spans,{'dtq'},excluded)
        assert merge_intervals.union_intervals(cleaned)[0]['extra_tags']==['detail']
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',spans,{'wd','dtq'},excluded,policies({'detail'}))
        assert merge_intervals.union_intervals(cleaned)[0]['extra_tags']==[]


def test_merge_only_applies_shared_rules_without_inference_or_changing_source_results(pipeline,monkeypatch,tmp_path):
    p=pipeline
    p.scores['wd'][:]=0;p.scores['dtq'][:]=0
    p.scores['wd'][0:4:2,0]=.95;p.scores['dtq'][1:4:2,0]=.95
    args=['--models','wd','dtq',*p.args[2:]]
    old=tmp_path/'old.yml';old.write_text('version: 1\nshared_marker_frames: false\nshared_marker_extras: false\noccurrence_markers: false\n')
    assert app.main([*args,'--temporal-policy',str(old)])==0
    assert not p.stash.scene['scene_markers']
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        before=state.connection.execute('SELECT model,kind,data FROM results ORDER BY model,kind').fetchall()
    monkeypatch.setattr(app.services,'load_tagger',lambda *args:pytest.fail('shared merge loaded a model'))
    monkeypatch.setattr(sprites,'fetch_cues',lambda *args:pytest.fail('shared merge fetched images'))
    assert app.main([*args,'--merge-only'])==0
    assert [(m['seconds'],m['end_seconds']) for m in p.stash.scene['scene_markers']]==[(0,4)]
    with merge_store.StateStore(p.path,merge_store.scope_for(app.STASH_URL),True) as state:
        assert before==state.connection.execute('SELECT model,kind,data FROM results ORDER BY model,kind').fetchall()
    p.stash.writes.clear()
    assert app.main([*args,'--merge-only'])==0 and not p.stash.writes


def test_shared_merge_dry_run_and_extra_explanation(pipeline,monkeypatch,tmp_path,capsys):
    p=pipeline
    p.scores['wd'][:]=[.95,0];p.scores['dtq'][:]=[0,.96]
    args=['--models','wd','dtq',*p.args[2:]]
    core=tmp_path/'core.yml';core.write_text('version: 1\nexcluded_core_tags: [extra]\n')
    args.extend(['--marker-core-exclusions',str(core)])
    assert app.main(args)==0
    expected=(p.path.read_bytes(),deepcopy(p.stash.scene))
    p.stash.dry_run=True
    assert app.main([*args,'--merge-only','--dry-run'])==0
    assert (p.path.read_bytes(),p.stash.scene)==expected
    capsys.readouterr()
    assert app.main([*args,'--explain-tag','ai:extra'])==0
    output=capsys.readouterr().out
    report=json.loads(output[output.index('[\n'):])[0]
    assert report['as_marker_extra'][0]['tag']=='ai:common'
    assert set(report['as_marker_extra'][0]['evidence']['source_evidence'])=={'dtq'}



@pytest.mark.parametrize('fail_collection', [False, True])
def test_clipped_replay_closes_its_observation_cache(monkeypatch, fail_collection):
    from stash_ai.evidence import shared_markers
    from stash_ai.evidence.observations import ObservationIndex

    records = {'wd': [segment('wd', [[('core', .95), ('detail', .9)]] * 4)]}
    spans, _ = fused(records)
    span = deepcopy(spans[0])
    span.update(start=1, end=3)
    original = deepcopy(records)
    indexes = []

    class TrackedIndex(ObservationIndex):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            indexes.append(self)

    monkeypatch.setattr(shared_markers, 'ObservationIndex', TrackedIndex)
    if fail_collection:
        def fail(app, span, index, allowed=None):
            index.support(span['start'], span['end'], app.TEMPORAL_POLICY)
            raise RuntimeError('collection interrupted')
        monkeypatch.setattr(shared_markers, 'collect_extras', fail)
        with pytest.raises(RuntimeError, match='collection interrupted'):
            shared_markers.refresh_clipped_evidence(app, span, records, policies())
    else:
        assert shared_markers.refresh_clipped_evidence(app, span, records, policies())
        assert span['extra_tags'] == ['detail']
        assert span['extra_support']['detail']['sources']['wd']['intervals'] == [[1, 3]]
    assert len(indexes) == 1
    assert indexes[0].cache_keys == set() and indexes[0].timelines == {}
    assert records == original
