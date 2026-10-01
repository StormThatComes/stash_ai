"""Shared scene proof, short extras, bounded RAM/prefetch, and 9000-scene output."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
import threading

import numpy as np
import pytest

from stash_ai import cli as app
from stash_ai import fusion
from stash_ai import merge
from stash_ai import progress
from stash_ai.runtime import RamCache, RunReport, retained_size
from stash_ai.scene import shared_scene_contributions, cleanup_scene_support
from stash_ai.taggers import SOURCES
from stash_ai.temporal import SceneAccumulator, TemporalPolicy
from tests.helpers.evidence import segment
from tests.helpers.evidence import frames


from tests.helpers.evidence import policies
from tests.helpers.evidence import paired_records
pytestmark = pytest.mark.usefixtures("reset_temporal")


def shared(scenes, markers, excluded=None, metrics=None):
    return shared_scene_contributions(app, fusion.FrameIndex(app, markers, policies()), scenes,
                                      excluded or policies(), metrics=metrics)


def test_joint_scene_high_detections_keep_each_model_score():
    scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [0, .86]})
    hits = shared(scenes, markers)
    assert len(hits) == 1 and hits[0]['qualification'] == 'shared_repeated_high'
    assert hits[0]['support_sources'] == ['wd', 'dtq']
    assert hits[0]['source_evidence']['wd']['score'] == .425
    assert hits[0]['source_evidence']['dtq']['score'] == .43
    assert hits[0]['joint_statistics']['hits'] == 2


def test_joint_scene_repeated_weak_support_can_alternate_between_models():
    scenes, markers = paired_records({'wd': [.7, 0, .7, 0], 'dtq': [0, .71, 0, .71]})
    hits = shared(scenes, markers)
    assert hits[0]['qualification'] == 'shared_persistent_support'
    assert hits[0]['joint_statistics']['longest_hits'] == 4


@pytest.mark.parametrize('change', ['same_time', 'different_input', 'different_timeline', 'missing_scene',
                                   'different_run', 'different_threshold', 'duplicate_model', 'gaps'])
def test_joint_scene_needs_compatible_distinct_observations(change):
    scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [0, .86]})
    if change == 'same_time':
        scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [.86, 0]})
    elif change == 'different_input':
        markers['dtq'][0]['meta']['frame_input_signature'] = 'other'
        scenes['dtq'][0]['meta']['frame_input_signature'] = 'other'
    elif change == 'different_timeline':
        markers['dtq'][0]['items'][0]['end'] = .5
    elif change == 'missing_scene':
        scenes.pop('dtq')
    elif change == 'different_run':
        scenes['dtq'][0]['meta']['inference_signature'] = 'new'
    elif change == 'different_threshold':
        scenes['dtq'][0]['meta']['scene_threshold'] = .5
    elif change == 'duplicate_model':
        markers = {'wd': markers['wd'] * 2}; scenes = {'wd': scenes['wd'] * 2}
    elif change == 'gaps':
        scenes, markers = paired_records({'wd': [.7, 0, 0, .7, 0, 0], 'dtq': [0, .7, 0, 0, .7, 0]})
    assert shared(scenes, markers) == []


def test_scene_thresholds_are_per_model_and_marker_exclusions_do_not_apply():
    scenes, markers = paired_records({'wd': [.75, 0], 'dtq': [0, .9]}, {'wd': .7, 'dtq': .85})
    index = fusion.FrameIndex(app, markers, policies({'core'}))
    assert shared_scene_contributions(app, index, scenes, policies())[0]['tag'] == 'core'
    excluded = policies(); excluded['wd'] = frozenset({'core'})
    assert shared(scenes, markers, excluded) == []


def test_partial_cache_can_prove_positive_counts_but_does_not_invent_missing_scores():
    scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [0, .86]}, marker_threshold=.95)
    metrics = Counter()
    assert shared(scenes, markers, metrics=metrics)[0]['joint_statistics']['cache_partial']
    assert metrics['scene_cache_partial'] == 1
    scenes, markers = paired_records({'wd': [.7, 0, .7, 0], 'dtq': [0, .7, 0, .7]}, marker_threshold=.95)
    assert shared(scenes, markers) == []


def test_scene_rule_is_additive_and_can_be_disabled(tmp_path, monkeypatch):
    scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [0, .86]})
    with merge.StateStore(tmp_path/'state.sqlite3', 'fixture', cache_bytes=2**20) as state:
        for source in scenes:
            state.put('1', source, {'scene': scenes[source][0], 'markers': markers[source][0]})
        assert merge.evidence(app, state, '1', policies(), policies())[0][0]['derived'] == 'shared_scene_frames'
        monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, shared_scene_tags=False))
        assert merge.evidence(app, state, '1', policies(), policies())[0] == []
        # A strong single model remains sufficient under the existing rules.
        scenes, markers = paired_records({'wd': [.99, 0]})
        state.put('2', 'wd', {'scene': scenes['wd'][0], 'markers': markers['wd'][0]})
        hits, _ = merge.evidence(app, state, '2', policies(), policies())
        assert hits[0]['qualification'] == 'strong_peak' and 'derived' not in hits[0]


@pytest.mark.parametrize('change', ['exclude', 'changed_frames', 'changed_scene', 'new_source'])
def test_shared_scene_cleanup_uses_only_frozen_published_inputs(tmp_path, change):
    scenes, markers = paired_records({'wd': [.85, 0, 0], 'dtq': [0, .86, 0], 'oppai': [0, 0, .87]})
    with merge.StateStore(tmp_path/'state.sqlite3', 'fixture', cache_bytes=2**20) as state:
        for source in scenes:
            state.put('1', source, {'scene': scenes[source][0], 'markers': markers[source][0]})
        hits, _ = merge.evidence(app, state, '1', policies(), policies())
        excluded = policies(); excluded['wd'] = frozenset({'core'})
        if change in ('changed_frames', 'changed_scene'):
            kind = 'scene' if change == 'changed_scene' else 'markers'
            changed = deepcopy(scenes['dtq'][0] if kind == 'scene' else markers['dtq'][0])
            changed['meta']['captured_at'] = 'newer'
            state.put('1', 'dtq', {kind: changed})
            with pytest.raises(RuntimeError, match='sources changed'):
                cleanup_scene_support(app, state, '1', hits, {'wd'}, excluded)
        else:
            if change == 'new_source':
                new_scene, new_markers = paired_records({'joytag': [.99, .99, .99]})
                state.put('1', 'joytag', {'scene': new_scene['joytag'][0], 'markers': new_markers['joytag'][0]})
            cleaned = cleanup_scene_support(app, state, '1', hits, {'wd'}, excluded)
            assert cleaned[0]['support_sources'] == ['dtq', 'oppai']
            excluded['dtq'] = frozenset({'core'})
            assert cleanup_scene_support(app, state, '1', cleaned, {'dtq'}, excluded) == []


def test_short_extra_survives_a_long_marker_with_three_consecutive_high_samples(monkeypatch):
    rows = [[('core', .9)] + ([('detail', .9)] if 5 <= i < 8 else []) for i in range(100)]
    records = {'wd': [segment('wd', rows)]}
    markers = merge.union_intervals(merge.marker_contributions(app, records, policies({'detail'})))
    assert markers[0]['extra_tags'] == ['detail']
    proof = markers[0]['extra_evidence']['detail']
    assert proof['qualification'] == 'short_episode' and proof['longest_run_hits'] == 3
    assert proof['coverage_fraction'] == .03
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, short_marker_extras=False))
    assert merge.union_intervals(merge.marker_contributions(app, records, policies({'detail'})))[0]['extra_tags'] == []


@pytest.mark.parametrize('times', [[5], [5, 6], [5, 7, 9]])
def test_short_extra_does_not_accept_spikes_or_disconnected_hits(times):
    rows = [[('core', .9)] + ([('detail', .9)] if i in times else []) for i in range(100)]
    result = merge.union_intervals(merge.marker_contributions(app, {'wd': [segment('wd', rows)]}, policies({'detail'})))
    assert result[0]['extra_tags'] == []


def test_short_extra_runs_join_across_union_and_sources_without_double_counting():
    records = {
        'wd': [segment('wd', [[('core', .9)] if i < 50 else [] for i in range(100)])],
        'dtq': [segment('dtq', [[('core', .9)] if i >= 50 else [] for i in range(100)])],
        'oppai': [segment('oppai', [[('detail', .9)] if 49 <= i <= 51 else [] for i in range(100)])],
    }
    marker = merge.union_intervals(merge.marker_contributions(app, records, policies({'detail'})))[0]
    assert marker['end'] == 100 and marker['extra_evidence']['detail']['longest_run_hits'] == 3
    assert marker['extra_tags'] == ['detail'] and marker['sources'] == ['wd', 'dtq']
    assert set(marker['extra_evidence']['detail']['source_evidence']) == {'oppai'}


def test_short_extra_respects_min_hits_and_vtt_time_gaps(monkeypatch):
    rows = [[('core', .9), ('detail', .9)] if i < 3 else [('core', .9)] for i in range(100)]
    records = {'wd': [segment('wd', rows)]}
    for row in records['wd'][0]['items'][1:]:
        row['start'] += 10; row['end'] += 10
    monkeypatch.setattr(app, 'MARKER_MERGE_GAP_SECONDS', 20)
    marker = merge.union_intervals(merge.marker_contributions(app, records, policies({'detail'})))[0]
    assert marker['extra_tags'] == []
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, extra_min_hits=4))
    records = {'wd': [segment('wd', rows)]}
    assert merge.union_intervals(merge.marker_contributions(app, records, policies({'detail'})))[0]['extra_tags'] == []


def test_extra_queries_are_reused_for_all_primary_tags_without_changing_outputs(monkeypatch):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, short_marker_extras=False))
    rows = [[(f'tag{i}', .9) for i in range(30)]] * 20
    records = {source: [segment(source, rows)] for source in ('wd', 'dtq')}
    cache = RamCache(8 * 2**20)
    index = fusion.FrameIndex(app, records, policies(), cache=cache)
    spans = merge.marker_contributions(app, records, policies(), index=index)
    assert len(index.indexes) == 1 and next(iter(index.indexes.values())).scans == 1
    uncached = merge.marker_contributions(app, records, policies(), cache=RamCache(0))
    assert json.dumps(spans, sort_keys=True) == json.dumps(uncached, sort_keys=True)
    assert len(merge.union_intervals(spans)) == 30


def test_ram_cache_accounts_decoded_values_and_evicts_least_recent():
    value = {'rows': list(range(100))}
    cost = max(retained_size(('a', value)), retained_size(('b', deepcopy(value)))) + 128
    cache = RamCache(cost * 2)
    cache.put('a', value); cache.put('b', deepcopy(value)); cache.get('a'); cache.put('c', deepcopy(value))
    assert cache.get('b') is None and cache.get('a') == value
    assert cache.size <= cache.limit and cache.stats['evictions'] == 1
    cache.put('huge', bytes(cache.limit * 2))
    assert cache.get('huge') is None and cache.size <= cache.limit


def test_store_cache_is_detached_invalidated_and_rollback_safe(tmp_path):
    first = dict(mode='scene', meta={'origin': 'inference', 'inference_signature': 'a'}, items=[{'tag': 'core', 'score': .8}])
    second = {**first, 'meta': {**first['meta'], 'inference_signature': 'b'}, 'items': []}
    with merge.StateStore(tmp_path/'state.sqlite3', 'fixture', cache_bytes=2**20) as state:
        state.put('1', 'wd', {'scene': first})
        statements = []; state.connection.set_trace_callback(statements.append)
        state.records('1', 'scene')['wd'][0]['items'].clear()
        assert state.records('1', 'scene')['wd'][0]['items'] == first['items']
        assert sum(sql.startswith('SELECT r.model,r.data') for sql in statements) == 1
        assert progress.completed(state, '1', 'wd', ['scene'], 'a')
        try:
            with state.connection:
                state._put('1', 'wd', {'scene': second})
                assert state.records('1', 'scene')['wd'][0]['items'] == []
                raise RuntimeError('rollback')
        except RuntimeError:
            pass
        assert state.records('1', 'scene')['wd'][0]['items'] == first['items']
        state.put('1', 'wd', {'scene': second})
        assert progress.completed(state, '1', 'wd', ['scene'], 'b')
        progress.request_refresh(state, ['1'], ['wd'], ['scene'])
        assert not progress.completed(state, '1', 'wd', ['scene'], 'b')
        progress.finish(state, '1', 'wd', {'scene': second}, {})
        assert progress.completed(state, '1', 'wd', ['scene'], 'b')


def test_preloaded_collection_metadata_removes_pair_queries(tmp_path):
    record = dict(mode='scene', meta={'origin': 'inference', 'inference_signature': 'a'}, items=[])
    with merge.StateStore(tmp_path/'state.sqlite3', 'fixture', cache_bytes=2**20) as state:
        for scene in range(50):
            for source in ('wd', 'dtq'): state.put(str(scene), source, {'scene': record})
        state.preload_sources(); progress.preload_status(state)
        statements = []; state.connection.set_trace_callback(statements.append)
        for scene in range(50):
            assert state.sources(str(scene), 'scene') == {'wd', 'dtq'}
            for source in ('wd', 'dtq'): assert progress.completed(state, str(scene), source, ['scene'], 'a')
        assert not statements


@pytest.mark.parametrize('prefetch', [0, 2, 4, 8])
def test_multi_batch_prefetch_resume_matches_uninterrupted_results(pipeline, prefetch):
    p = pipeline; p.fail['wd'] = 4
    args = [*p.args, '--prefetch-batches', str(prefetch), '--ram-cache-gb', '0.01']
    with pytest.raises(KeyboardInterrupt): app.main(args)
    p.fail.clear()
    assert app.main(args) == 0
    assert p.loaded[-1].seen == list(range(4, 10))
    assert not any(thread.name.startswith('prepare-frames') for thread in threading.enumerate())
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        output = state.records('1', 'markers')['wd'][0]['items']
    assert len(output) == 10 and [row['start'] for row in output] == list(range(10))


def test_prefetch_queue_has_a_strict_bound(pipeline, monkeypatch):
    p = pipeline
    p.cues = [app.Cue(i, i+1, 'sprite', (0,0,1,1)) for i in range(30)]
    config = app.load_model_settings(app.MODELS_CONFIG_FILE, ['wd'], None)[0]
    model = app.load_tagger(config, p.vocab, app.prepare_image)
    ready = threading.Event()
    original = model.prepare
    def prepare(value):
        result = original(value)
        if value == 15: ready.set()
        return result
    monkeypatch.setattr(model, 'prepare', prepare)
    iterator = progress.prepared_batches(app, model, p.stash, {}, p.cues, 3)
    try:
        next(iterator)
        assert ready.wait(2)
        assert model.prepared == list(range(16))  # Current 4 frames + three queued batches.
    finally:
        iterator.close()
    assert not any(thread.name.startswith('prepare-frames') for thread in threading.enumerate())


def test_9000_repeated_errors_use_bounded_memory_and_output():
    lines = []; report = RunReport(write=lines.append)
    for scene in range(9000):
        report.detail(f'scene {scene} details')
        report.issue('merge', 'wd', scene, ValueError(), f'invalid source for {scene}')
    report.finish('Merged', Counter(errors=9000))
    assert len(lines) <= 5 and len(report.groups) == 1
    assert len(next(iter(report.groups.values()))['examples']) == 2
    assert '9000' in '\n'.join(lines)


def test_9000_scene_dry_run_prints_only_a_collection_summary(pipeline, monkeypatch, capsys):
    p = pipeline; p.stash.dry_run = True
    def scene(identifier):
        return dict(id=str(identifier), paths={}, tags=[], scene_markers=[])
    monkeypatch.setattr(p.stash, 'iter_scenes', lambda ids: (scene(i) for i in range(9000)))
    monkeypatch.setattr(p.stash, 'find_scene', scene)
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('merge-only loaded a model'))
    args = ['--models', 'wd', '--no-hierarchy', '--state-db', str(p.path), '--merge-only', '--dry-run', '--ram-cache-gb', '.001']
    assert app.main(args) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) <= 15 and 'scenes_synced=9000' in output
    assert not p.path.exists() and not p.stash.writes


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf'])
def test_invalid_ram_budget_is_rejected_before_work(pipeline, value):
    with pytest.raises(SystemExit) as exc:
        app.main([*pipeline.args, '--ram-cache-gb', value])
    assert exc.value.code == 2 and not pipeline.loaded and not pipeline.stash.writes


def test_new_policy_switches_do_not_invalidate_inference_or_old_snapshots(pipeline, monkeypatch):
    config = app.load_model_settings(app.MODELS_CONFIG_FILE, ['wd'], None)[0]
    before = progress.model_identity(app, config, pipeline.vocab)
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, shared_scene_tags=False,
                       short_marker_extras=False, extra_short_min_hits=7))
    assert progress.model_identity(app, config, pipeline.vocab) == before
    old = asdict(app.TEMPORAL_POLICY)
    for key in ('shared_scene_tags', 'short_marker_extras', 'extra_short_min_hits'): old.pop(key)
    restored = TemporalPolicy.from_snapshot(old)
    assert not restored.shared_scene_tags and not restored.short_marker_extras


def test_shared_scene_merge_only_reuses_cache_and_preserves_other_outputs(pipeline, monkeypatch, tmp_path):
    p = pipeline
    for values in p.scores.values(): values[:] = 0
    p.scores['wd'][0,0] = .6; p.scores['joytag'][1,0] = .62
    args = ['--models', 'wd', 'joytag', *p.args[2:]]
    old = tmp_path/'old.yml'; old.write_text('version: 1\nshared_scene_tags: false\n')
    assert app.main([*args, '--temporal-policy', str(old)]) == 0
    assert not p.stash.scene['tags']
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        before = state.connection.execute('SELECT model,kind,data FROM results ORDER BY model,kind').fetchall()
    markers = deepcopy(p.stash.scene['scene_markers'])
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('merge-only loaded a model'))
    monkeypatch.setattr(app, 'fetch_cues', lambda *args: pytest.fail('merge-only fetched input images'))
    assert app.main([*args, '--merge-only', '--scene-tags-only']) == 0
    assert [tag['name'] for tag in p.stash.scene['tags']] == ['ai:common']
    assert p.stash.scene['scene_markers'] == markers
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert before == state.connection.execute('SELECT model,kind,data FROM results ORDER BY model,kind').fetchall()
    p.stash.writes.clear()
    assert app.main([*args, '--merge-only', '--scene-tags-only']) == 0 and not p.stash.writes
    before_dry = p.path.read_bytes(), deepcopy(p.stash.scene)
    p.stash.dry_run = True
    assert app.main([*args, '--merge-only', '--dry-run', '--scene-tags-only']) == 0
    assert before_dry == (p.path.read_bytes(), p.stash.scene)


@pytest.mark.parametrize('manual_delete', [False, True])
def test_shared_scene_cleanup_command_preserves_sources_and_manual_removals(pipeline, monkeypatch, tmp_path, manual_delete):
    p = pipeline
    for values in p.scores.values(): values[:] = 0
    # Keep Oppai clearly above its scene threshold, including float32 rounding.
    p.scores['wd'][0,0] = .6; p.scores['joytag'][1,0] = .62; p.scores['oppai'][2,0] = .85
    args = ['--models', 'wd', 'joytag', 'oppai', *p.args[2:]]
    assert app.main(args) == 0
    assert [tag['name'] for tag in p.stash.scene['tags']] == ['ai:common']
    if manual_delete:
        p.stash.scene['tags'] = []
    excluded = tmp_path/'scene.yml'
    excluded.write_text('version: 1\nexcluded_scene_tags: []\nby_model:\n  joytag: [common]\n')
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('cleanup loaded a model'))
    cleanup = [*args, '--cleanup-scene-tags-only', '--scene-tag-exclusions', str(excluded)]
    assert app.main(cleanup) == 0
    assert [tag['name'] for tag in p.stash.scene['tags']] == ([] if manual_delete else ['ai:common'])
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        hits = state.snapshot('1', 'scene')[0]['contributions']
        assert not hits if manual_delete else hits[0]['support_sources'] == ['wd', 'oppai']
    p.stash.writes.clear()
    assert app.main(cleanup) == 0 and not p.stash.writes
