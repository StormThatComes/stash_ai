"""Temporal qualification uses capped frame votes and bounded temporal coverage."""
from dataclasses import replace
import pytest
from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import segment, paired_records, empty_policies
from stash_ai.storage.store import StateStore
from stash_ai.evidence import pipeline
from stash_ai.evidence.temporal.policy import TemporalPolicy

pytestmark = pytest.mark.usefixtures('reset_progress')


def test_reserved_ownership_tag_is_rejected(tmp_path):
    from stash_ai.config import load_exclusion_policy
    from stash_ai.exclusions import excluded_names
    p = tmp_path/'bans.yml'
    p.write_text('version: 1\nexcluded_scene_tags: ["marker:stash-taggers"]\n')
    with pytest.raises(ValueError, match='reserved'):
        load_exclusion_policy(p, 'excluded_scene_tags')


def test_consensus_does_not_count_tag_removed_by_frame_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'MARKER_MAX_TAGS_PER_FRAME', 1)
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY,marker_continue_ratio=.85))
    with StateStore(tmp_path/'s.db', 's') as state:
        state.put('1','wd',{'markers':segment('wd', [[('a',.75),('b',.74)]]*2)})
        state.put('1','dtq',{'markers':segment('dtq', [[('z',.95),('a',.70)]]*2)})
        _, spans = pipeline.evidence(app,state,'1',empty_policies(),empty_policies())
        assert not any(s['tag']=='a' for s in spans)



def test_extra_interval_query_clips_leading_coverage_without_inventing_sample():
    from stash_ai.evidence.observations import ObservationIndex
    events=[{'start':0,'end':10,'extras':{'x':{'wd':.9}}},{'start':10,'end':20,'extras':{'x':{'wd':.9}}}]
    index=ObservationIndex(events,{'wd':'s'},'group')
    try:
        proof=index.support(5,15,app.TEMPORAL_POLICY)['x']
        assert proof['intervals']==[[5,15]] and proof['hits']==1
        assert proof['sample_times']==[10]
        assert proof['sources']['wd']['intervals']==[[5,15]]
    finally:
        index.close()


def test_other_model_range_does_not_suppress_an_occurrence_for_another_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(TemporalPolicy(), shared_marker_frames=False,
                                                    shared_scene_tags=False))
    scenes, markers = paired_records({'dtq': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    with StateStore(tmp_path/'s.db', 's') as state:
        state.put('1', 'dtq', {'scene': scenes['dtq'], 'markers': markers['dtq']})
        hits, alone = pipeline.evidence(app, state, '1', empty_policies(), empty_policies())
        assert len(alone) == 1 and alone[0]['derived'] == 'occurrence'
        state.put('1', 'wd', {'markers': segment('wd', [[('different', .95)]]*2)})
        hits, together = pipeline.evidence(app, state, '1', empty_policies(), empty_policies())
        assert any(h['tag'] == 'core' for h in hits)
        assert {(s['source'], s['tag']) for s in together} == {('wd', 'different'), ('dtq', 'core')}


@pytest.mark.parametrize('key', ['excluded_scene_tags', 'excluded_core_tags'])
@pytest.mark.parametrize('label', ['marker:stash-taggers', 'marker:occurrence', 'category:AI'])
@pytest.mark.parametrize('per_model', [False, True])
def test_all_exclusion_sections_reject_technical_labels(tmp_path, key, label, per_model):
    from stash_ai.config import load_exclusion_policy
    import yaml
    data = {'version': 1, key: [] if per_model else [label]}
    if per_model:
        data['by_model'] = {'wd': [label]}
    path = tmp_path/'exclusions.yml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='reserved'):
        load_exclusion_policy(path, key)


def test_occurrence_agreement_keeps_its_anchor_when_that_model_has_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(TemporalPolicy(), shared_marker_frames=False,
                                                    shared_scene_tags=False))
    scenes, markers = paired_records({'wd': [0,.95,0], 'dtq': [0,.95,0]}, threshold=.35, marker_threshold=.65)
    # An unrelated WD range does not change the point's evidence or anchor.
    from stash_ai.evidence.occurrences import occurrence_contributions
    hits = [{**scenes[s]['items'][0], 'source': s, 'meta': scenes[s]['meta']} for s in scenes]
    existing = [{'source':'wd', 'tag':'other', 'start':0, 'end':2}]
    points = occurrence_contributions(app, hits, markers, existing, empty_policies(), empty_policies())
    assert len(points) == 1 and points[0]['source'] == 'wd'
    assert points[0]['support_sources'] == ['dtq']


def test_all_qualified_ranges_are_retained_for_each_model(tmp_path):
    with StateStore(tmp_path/'s.db', 's') as state:
        for source, label in [('wd','a'), ('dtq','b')]:
            state.put('1', source, {'markers': segment(source, [[(label,.95), (label+'-extra',.9)]]*2)})
        _, spans = pipeline.evidence(app, state, '1', empty_policies(), empty_policies())
    assert {(s['source'],s['tag']) for s in spans} == {('wd','a'), ('wd','a-extra'), ('dtq','b'), ('dtq','b-extra')}
