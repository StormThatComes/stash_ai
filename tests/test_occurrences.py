"""Occurrence publication, old caches, revocation and exact sampled positions."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from stash_ai import inference
from stash_ai import merge_cleanup
from stash_ai import merge_sync
from stash_ai import cli as app
from stash_ai import merge
from stash_ai import progress
from stash_ai.occurrences import OCCURRENCE_TAG, occurrence_contributions
from stash_ai.temporal import SceneAccumulator, TemporalPolicy
from tests.helpers.stash import Stash
from tests.helpers.evidence import paired_records
from tests.helpers.evidence import policies


pytestmark = pytest.mark.usefixtures("reset_temporal")


def put(state, scenes, markers, scene='1'):
    for source in set(scenes) | set(markers):
        for kind, records in (('scene', scenes), ('markers', markers)):
            for segment in records.get(source, []):
                state.put(scene, source, {kind: segment}, {kind: False})


def test_peak_time_is_compact_earliest_max_and_survives_resume():
    policy = TemporalPolicy()
    acc = SceneAccumulator(.2, policy)
    cues = [app.Cue(i * 10, (i + 1) * 10, '', (0, 0, 1, 1)) for i in range(6)]
    acc.add(np.array([[.25], [.35], [.30]]), cues[:3])
    saved = acc.state()
    resumed = SceneAccumulator(.2, policy, saved)
    resumed.add(np.array([[.35], [.34], [.32]]), cues[3:])
    assert resumed.statistics(['core'])['core']['peak_time'] == 10
    assert resumed.statistics(['core'])['core']['max_score'] == .35
    assert saved['items'][0]['hits'] == 3
    assert len(saved['items'][0]) == len(resumed.state()['items'][0])


def test_old_checkpoint_never_assigns_a_new_time_to_an_unseen_old_maximum():
    acc = SceneAccumulator(.2, TemporalPolicy())
    acc.add(np.array([[.9]]), [app.Cue(0, 10, '', (0, 0, 1, 1))])
    saved = acc.state()
    saved['items'][0].pop('peak_time')
    resumed = SceneAccumulator(.2, TemporalPolicy(), saved)
    resumed.add(np.array([[.4]]), [app.Cue(10, 20, '', (0, 0, 1, 1))])
    assert 'peak_time' not in resumed.statistics(['core'])['core']
    resumed.add(np.array([[.95]]), [app.Cue(20, 30, '', (0, 0, 1, 1))])
    assert resumed.statistics(['core'])['core']['peak_time'] == 20


@pytest.mark.parametrize('cache', ['peak', 'old_frames', 'missing', 'different_run', 'different_input'])
def test_one_representative_point_uses_only_matching_recorded_times(tmp_path, cache):
    scenes, markers = paired_records({'wd': [0, .95, 0, .8, 0]}, threshold=.35, marker_threshold=.65)
    if cache != 'peak':
        scenes['wd'][0]['items'][0]['statistics'].pop('peak_time')
    if cache == 'missing':
        for row in markers['wd'][0]['items']: row['tags'] = []
    if cache == 'different_run': markers['wd'][0]['meta']['inference_signature'] = 'other'
    if cache == 'different_input': markers['wd'][0]['meta']['frame_input_signature'] = 'other'
    metrics = Counter()
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, spans = merge.evidence(app, state, '1', policies(), policies(), metrics)
    assert len(hits) == 1
    if cache in ('peak', 'old_frames'):
        assert len(spans) == 1 and spans[0]['start'] == 1 and spans[0]['end'] is None
        assert spans[0]['derived'] == 'occurrence'
        assert len(spans[0]['occurrence_evidence']) == 1
        assert spans[0]['occurrence_evidence'][0]['score'] == .95
    else:
        assert spans == [] and metrics['occurrence_missing_timestamps'] == 1


def test_scene_peak_below_marker_retention_floor_still_supplies_a_point(tmp_path):
    scenes, markers = paired_records({'wd': [.31, .32, .34, .31, 0]}, threshold=.35, marker_threshold=.65)
    assert all(not row['tags'] for row in markers['wd'][0]['items'])
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
    assert hits[0]['qualification'] == 'persistent_support'
    assert [(s['start'], s['end']) for s in spans] == [(2, None)]


@pytest.mark.parametrize('block', ['core', 'scene'])
def test_occurrences_respect_all_exclusions(tmp_path, block):
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    cores, tags = policies(), policies()
    if block == 'core': cores['wd'] = frozenset({'core'})
    if block == 'scene': tags['wd'] = frozenset({'core'})
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        _, spans = merge.evidence(app, state, '1', cores, tags)
    assert spans == []


@pytest.mark.parametrize('case', ['unqualified', 'existing_range', 'disabled', 'scene_only'])
def test_occurrences_only_fill_missing_locations(tmp_path, monkeypatch, case):
    values = [.95, .95, 0] if case == 'existing_range' else [0, .95, 0]
    if case == 'unqualified': values = [0, .4, 0]
    scenes, markers = paired_records({'wd': values}, threshold=.35, marker_threshold=.65)
    if case == 'disabled': monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, occurrence_markers=False))
    if case == 'scene_only': monkeypatch.setattr(app, 'CREATE_MARKERS', False)
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        _, spans = merge.evidence(app, state, '1', policies(), policies())
    assert not any(s.get('derived') == 'occurrence' for s in spans)


@pytest.mark.parametrize('different_input', [False, True])
def test_agreement_is_one_point_with_separate_scores_and_compatible_inputs(tmp_path, different_input):
    scenes, markers = paired_records({'wd': [.95, 0, .85, 0], 'dtq': [0, 0, .90, 0]},
                                     threshold={'wd': .35, 'dtq': .20}, marker_threshold=.65)
    if different_input:
        for record in (scenes['dtq'][0], markers['dtq'][0]): record['meta']['frame_input_signature'] = 'other'
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        _, spans = merge.evidence(app, state, '1', policies(), policies())
    assert len(spans) == 1
    point = spans[0]
    assert point['score'] is None
    if different_input:
        assert len(point['occurrence_evidence']) == 1 and point['start'] == 0
    else:
        assert point['start'] == 2
        assert {w['source']: w['score'] for w in point['occurrence_evidence']} == {'wd': .85, 'dtq': .90}


def test_duplicate_records_do_not_count_as_two_model_votes(tmp_path):
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    scenes['wd'] *= 2; markers['wd'] *= 2
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        _, spans = merge.evidence(app, state, '1', policies(), policies())
    assert len(spans) == len(spans[0]['occurrence_evidence']) == 1


@pytest.mark.parametrize('use_end', [False, True])
def test_point_publication_is_idempotent_and_promotes_to_a_range(tmp_path, monkeypatch, use_end):
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    stash = Stash()
    monkeypatch.setattr(app, 'MARKER_USE_END_TIME', use_end)
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
        resolver = app.TagResolver(stash, source=merge.AI)
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), hits, spans)
        point = deepcopy(stash.scene['scene_markers'][0]); stash.writes.clear()
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), hits, spans)
        assert not stash.writes and len(stash.scene['scene_markers']) == 1
        assert point['seconds'] == 1 and point['end_seconds'] is None
        assert {t['name'] for t in point['tags']} == {merge.AI.owner_tag, OCCURRENCE_TAG, 'm:core:occurance'}
        assert point['title'].endswith(' · wykrycie')
        scenes, markers = paired_records({'wd': [.95, .95, .95]}, threshold=.35, marker_threshold=.65)
        for kind, value in (('scene', scenes), ('markers', markers)): state.put('1', 'wd', {kind: value['wd'][0]})
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), hits, spans)
        assert len(stash.scene['scene_markers']) == 1
        marker = stash.scene['scene_markers'][0]
        assert marker['id'] == point['id'] and not marker['title'].endswith('wykrycie')
        assert OCCURRENCE_TAG not in {t['name'] for t in marker['tags']}
        assert {t['name'] for t in marker['tags'] if t['name'].startswith('m:')} == {'m:core'}
        assert marker['end_seconds'] == (3 if use_end else None)


def test_manual_location_suppresses_automatic_point_without_modifying_it(tmp_path):
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    stash = Stash(); manual = deepcopy(stash.marker(merge.AI, 'core', 'manual', manual=True))
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
        merge_sync.sync_merged(app, stash, app.TagResolver(stash, source=merge.AI), state, stash.find_scene('1'), hits, spans)
        assert state.snapshot('1', 'markers')[0]['contributions'] == []
    assert stash.scene['scene_markers'] == [manual]


@pytest.mark.parametrize('kind', ['core', 'scene'])
def test_cleanup_revokes_only_selected_witness_and_never_moves_point(tmp_path, kind):
    scenes, markers = paired_records({'wd': [0, .95, 0], 'dtq': [0, .90, 0]}, threshold=.35, marker_threshold=.65)
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        _, spans = merge.evidence(app, state, '1', policies(), policies())
        original = deepcopy(spans)
        cores, tags = policies(), policies()
        (cores if kind == 'core' else tags)['wd'] = frozenset({'core'})
        cleaned = merge_cleanup.cleanup_support(app, state, '1', spans, {'wd'}, cores, tags)
        assert spans == original
        assert len(cleaned) == 1 and cleaned[0]['source'] == 'dtq' and cleaned[0]['start'] == 1
        assert len(cleaned[0]['occurrence_evidence']) == 1
        (cores if kind == 'core' else tags)['dtq'] = frozenset({'core'})
        assert merge_cleanup.cleanup_support(app, state, '1', cleaned, {'dtq'}, cores, tags) == []


def test_cleanup_rechecks_shared_scene_qualification_before_retaining_point(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, shared_marker_frames=False))
    scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [0, .86]})
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
        assert hits[0]['derived'] == 'shared_scene_frames' and spans[0]['derived'] == 'occurrence'
        blocked = policies(); blocked['wd'] = frozenset({'core'})
        assert merge_cleanup.cleanup_support(app, state, '1', spans, {'wd'}, policies(), blocked) == []


def test_marker_cap_has_priority_over_additional_points(tmp_path, monkeypatch):
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    monkeypatch.setattr(app, 'MAX_MARKERS_PER_SCENE', 1)
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, _ = merge.evidence(app, state, '1', policies(), policies())
        metrics = Counter()
        existing = [dict(tag='other', source='wd', start=0, end=3)]
        assert occurrence_contributions(app, hits, markers, existing, policies(), policies(), metrics) == []
        assert metrics['occurrence_limited'] == 1


def test_old_policy_snapshots_and_inference_identity_remain_compatible(pipeline, monkeypatch):
    settings = app.load_model_settings(app.MODELS_CONFIG_FILE, ['wd'], None)[0]
    identity = progress.model_identity(app, settings, pipeline.vocab)
    saved = asdict(app.TEMPORAL_POLICY); saved.pop('occurrence_markers')
    assert not TemporalPolicy.from_snapshot(saved).occurrence_markers
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, occurrence_markers=False))
    assert progress.model_identity(app, settings, pipeline.vocab) == identity


def test_pipeline_reuses_results_for_markers_only_dry_run_and_cleanup(pipeline, monkeypatch, tmp_path):
    p = pipeline
    p.scores['wd'][:] = 0; p.scores['wd'][3, 0] = .95
    assert app.main(p.args) == 0
    point = deepcopy(p.stash.scene['scene_markers'])
    assert len(point) == 1 and point[0]['seconds'] == 3 and point[0]['end_seconds'] is None
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        records = state.connection.execute('SELECT data FROM results ORDER BY kind').fetchall()
        assert state.records('1', 'scene')['wd'][0]['items'][0]['statistics']['peak_time'] == 3
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('merge or cleanup loaded a model'))
    monkeypatch.setattr(inference, 'fetch_cues', lambda *a: pytest.fail('merge or cleanup fetched media'))
    p.stash.writes.clear()
    assert app.main([*p.args, '--merge-only', '--markers-only']) == 0
    assert not p.stash.writes and p.stash.scene['scene_markers'] == point
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    p.stash.dry_run = True
    assert app.main([*p.args, '--merge-only', '--dry-run']) == 0
    assert before == (p.path.read_bytes(), p.stash.scene)
    p.stash.dry_run = False
    policy = tmp_path/'excluded.yml'; policy.write_text('version: 1\nexcluded_scene_tags: [common]\n')
    assert app.main([*p.args, '--cleanup-only', '--scene-tag-exclusions', str(policy)]) == 0
    assert not p.stash.scene['scene_markers'] and not p.stash.scene['tags']
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.connection.execute('SELECT data FROM results ORDER BY kind').fetchall() == records


def test_merge_only_recovers_old_cached_frames_without_rewriting_them(pipeline, monkeypatch):
    p = pipeline
    p.scores['wd'][:] = 0; p.scores['wd'][5, 0] = .95
    assert app.main(p.args) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
        record = state.records('1', 'scene')['wd'][0]
        for item in record['items']: item['statistics'].pop('peak_time', None)
        record['meta']['analysis_format'] = 2
        state.put('1', 'wd', {'scene': record})
    p.stash.scene['scene_markers'] = []
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('old cache caused inference'))
    assert app.main([*p.args, '--merge-only']) == 0
    assert len(p.stash.scene['scene_markers']) == 1 and p.stash.scene['scene_markers'][0]['seconds'] == 5


def test_enabling_points_does_not_invalidate_completed_inference(pipeline, monkeypatch, tmp_path):
    p = pipeline
    p.scores['wd'][:] = 0; p.scores['wd'][7, 0] = .95
    disabled = tmp_path/'old.yml'; disabled.write_text('version: 1\noccurrence_markers: false\n')
    assert app.main([*p.args, '--temporal-policy', str(disabled)]) == 0
    assert not p.stash.scene['scene_markers']
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('policy change repeated inference'))
    assert app.main(p.args) == 0
    assert len(p.stash.scene['scene_markers']) == 1 and p.stash.scene['scene_markers'][0]['seconds'] == 7


def test_point_publication_recovers_after_api_failure_without_inference(pipeline, monkeypatch):
    p = pipeline
    p.scores['wd'][:] = 0; p.scores['wd'][3, 0] = .95
    p.stash.fail_create = True
    assert app.main(p.args) == 1
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.snapshot('1', 'markers')[1]['contributions'][0]['derived'] == 'occurrence'
    p.stash.fail_create = False
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('API retry loaded a model'))
    assert app.main([*p.args, '--merge-only']) == 0
    assert len(p.stash.scene['scene_markers']) == 1
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.snapshot('1', 'markers')[1] is None


def test_global_cleanup_masks_revoke_points_without_touching_raw_evidence(tmp_path):
    from stash_ai.cleanup import begin, finish
    scenes, markers = paired_records({'wd': [0, .95, 0]}, threshold=.35, marker_threshold=.65)
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        _, spans = merge.evidence(app, state, '1', policies(), policies())
        state.publish('1', {'markers': {'contributions': spans}})
        original = state.records('1', 'scene')
        begin(state, 'cleanup', set(), {'core'}, set(), kinds=('markers',)); finish(state)
        assert state.snapshot('1', 'markers')[0]['contributions'] == []
        assert state.records('1', 'scene') == original


def test_cleanup_never_restores_a_manually_deleted_point(tmp_path):
    scenes, markers = paired_records({'wd': [0, .95, 0], 'dtq': [0, .90, 0]}, threshold=.35, marker_threshold=.65)
    stash = Stash()
    with merge.StateStore(tmp_path/'state', 'fixture') as state:
        put(state, scenes, markers)
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
        resolver = app.TagResolver(stash, source=merge.AI)
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), hits, spans)
        stash.scene['scene_markers'] = []
        cores = policies(); cores['wd'] = frozenset({'core'})
        cleaned = merge_cleanup.cleanup_support(app, state, '1', spans, {'wd'}, cores, policies())
        assert cleaned
        merge_sync.sync_merged(app, stash, resolver, state, stash.find_scene('1'), hits, cleaned, migrate=False, cleanup=True)
        assert not stash.scene['scene_markers']
