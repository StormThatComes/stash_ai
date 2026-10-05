"""Scene threshold replay, provenance and restart/resume equivalence."""

from tests.helpers.temporal import policy_args
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from stash_ai.rethreshold import execution as rethreshold_execution
from stash_ai.inference import sprites
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.storage.store as merge_store
import stash_ai.evidence.identity as cache_identity
from stash_ai.runtime.report import RunReport
from stash_ai.models import ModelSettings, SOURCES
from tests.helpers.evidence import policies

pytestmark = pytest.mark.usefixtures('reset_temporal')


def configure(monkeypatch, scene=.4, marker=.7):
    # Exercise automatic retention independently of the bundled cache floors.
    original = app.services.load_model_settings
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *a: [
        replace(c, scene_threshold=scene, marker_threshold=marker,
                scene_cache_floor=None, marker_cache_floor=None) for c in original(*a)])


def saved(path):
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), True) as state:
        return state.records('1', 'scene')


def outputs(path):
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), True) as state:
        hits, spans = merge.evidence(app, state, '1', policies(), policies())
        return ([(h['source'], h['tag'], h['score'], h.get('statistics'), h.get('qualification')) for h in hits],
                [(s['source'], s['tag'], s['start'], s['end'], s.get('derived'), s['score']) for s in spans])


@pytest.mark.parametrize('temporal', [True, False])
@pytest.mark.parametrize('scene_only', [True, False])
@pytest.mark.parametrize('aggregation', ['max', 'topk_mean'])
def test_cached_scene_statistics_and_occurrences_match_fresh_inference(
        pipeline, monkeypatch, tmp_path, temporal, scene_only, aggregation):
    p = pipeline
    configure(monkeypatch)
    monkeypatch.setattr(app, 'AGGREGATION', aggregation)
    p.scores['wd'][:, 0] = .45
    p.scores['wd'][:, 1] = .1
    p.scores['wd'][3, 1] = .95
    args = [*p.args, *(['--scene-tags-only'] if scene_only else []), *policy_args(p.path.parent, temporal)]
    assert app.main(args) == 0
    original = saved(p.path)
    configure(monkeypatch, scene=.8)
    assert app.main([*args, '--merge-only', '--rethreshold-scenes']) == 0
    cached = outputs(p.path)
    value = saved(p.path)['wd'][0]
    assert value.pop(cache_identity.SCENE_OVERRIDE) == .8
    assert value == original['wd'][0]
    p.stash.scene['scene_markers'] = []
    p.stash.scene['tags'] = []
    fresh = tmp_path/'fresh.sqlite3'
    assert app.main([*args, '--state-db', str(fresh), '--restart']) == 0
    assert outputs(fresh) == cached


def test_explain_reports_effective_thresholds_after_rethreshold(pipeline, monkeypatch, capsys):
    p = pipeline
    configure(monkeypatch, scene=.4, marker=.7)
    p.scores['wd'][:, 0] = .95
    assert app.main(p.args) == 0
    capsys.readouterr()

    configure(monkeypatch, scene=.8, marker=.9)
    assert app.main([
        *p.args, '--merge-only', '--rethreshold-scenes', '--rethreshold-markers'
    ]) == 0
    capsys.readouterr()

    assert app.main([*p.args, '--explain-tag', 'ai:common']) == 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index('[\n'):])[0]

    contributor = next(
        item for item in report['scene_contributors']
        if item['source'] == 'wd'
    )
    thresholds = contributor['thresholds']
    assert thresholds['stored_scene_threshold'] == pytest.approx(.4)
    assert thresholds['scene_threshold_override'] == pytest.approx(.8)
    assert thresholds['effective_scene_threshold'] == pytest.approx(.8)
    assert thresholds['scene_support_ratio'] == pytest.approx(.85)
    assert thresholds['effective_support_floor'] == pytest.approx(.68)
    assert thresholds['strong_peak_threshold'] == pytest.approx(.9)

    scene_record = report['recorded_sources']['scene']['wd']['records'][0]['thresholds']
    assert scene_record == thresholds

    marker = report['recorded_sources']['markers']['wd']['records'][0]['thresholds']
    assert marker['stored_marker_threshold'] == pytest.approx(.7)
    assert marker['marker_threshold_override'] == pytest.approx(.9)
    assert marker['effective_marker_threshold'] == pytest.approx(.9)
    assert marker['marker_continue_ratio'] == pytest.approx(.85)
    assert marker['effective_support_floor'] == pytest.approx(.765)


@pytest.mark.parametrize('scene_only', [True, False])
def test_merge_only_support_ratio_matches_fresh_inference(pipeline, monkeypatch, tmp_path, scene_only):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = .36
    p.scores['wd'][2:4, 1] = .6
    args = [*p.args, *(['--scene-tags-only'] if scene_only else [])]
    assert app.main(args) == 0
    original, baseline = saved(p.path), outputs(p.path)
    assert {t['name'] for t in p.stash.scene['tags']} == {'ai:common', 'ai:extra'}
    policy = tmp_path/'strict.yml'
    policy.write_text('version: 1\nscene_support_ratio: 1.0\n')

    with monkeypatch.context() as offline:
        for name in ('load_tagger', 'load_vocabulary'):
            offline.setattr(app.services, name, lambda *a: pytest.fail('accessed inference inputs'))
        offline.setattr(sprites, 'fetch_cues', lambda *a: pytest.fail('accessed inference inputs'))
        offline.setattr(p.stash, 'get_media', lambda *a: pytest.fail('fetched media'))
        assert app.main([*args, '--merge-only', '--temporal-policy', str(policy)]) == 0
        cached = outputs(p.path)
        assert {t['name'] for t in p.stash.scene['tags']} == {'ai:extra'}
        assert cached[0][0][3]['hits'] == 2
        assert saved(p.path) == original
        # The projection must not discard weaker observations needed on rollback.
        assert app.main([*args, '--merge-only']) == 0
        assert outputs(p.path) == baseline

    p.stash.scene['scene_markers'] = []
    p.stash.scene['tags'] = []
    fresh = tmp_path/'fresh.sqlite3'
    assert app.main([*args, '--state-db', str(fresh), '--temporal-policy', str(policy), '--restart']) == 0
    assert outputs(fresh) == cached


@pytest.mark.parametrize('missing', ['below_floor', 'frames'])
def test_support_ratio_change_rejects_incomplete_cache_before_publication(
        pipeline, monkeypatch, tmp_path, capsys, missing):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    if missing == 'frames':
        with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
            segment = state.records('1', 'scene')['wd'][0]
            segment.pop('frames')
            segment['meta'].pop('scene_frames_format')
            segment['meta'].pop('scene_score_floor')
            state.put('1', 'wd', {'scene': segment})
    before = saved(p.path), deepcopy(p.stash.scene)
    p.stash.writes.clear()
    policy = tmp_path/'changed.yml'
    policy.write_text(f'version: 1\nscene_support_ratio: {0.7 if missing == "below_floor" else 1.0}\n')
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main([*p.args, '--merge-only', '--temporal-policy', str(policy)]) == 1
    assert (saved(p.path), p.stash.scene) == before and not p.stash.writes
    assert 'run inference' in capsys.readouterr().out


def test_rethreshold_abort_report_is_explicit_and_groups_blockers(pipeline, monkeypatch, capsys):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    capsys.readouterr()

    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        segment = state.records('1', 'scene')['wd'][0]
        segment.pop('frames')
        segment['meta'].pop('scene_frames_format')
        segment['meta'].pop('scene_score_floor')
        state.put('1', 'wd', {'scene': segment})

    configure(monkeypatch, scene=.8)
    assert app.main([*p.args, '--merge-only', '--rethreshold-scenes']) == 1
    output = capsys.readouterr().out
    assert 'Threshold update aborted: 0 changes applied' in output
    assert 'Threshold preflight: checked=1, failed=1, applied=0' in output
    assert 'Threshold blockers: missing scene frame evidence=1' in output


def test_support_ratio_dry_run_preserves_source_and_stash(pipeline, monkeypatch, tmp_path, capsys):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = .36
    assert app.main(p.args) == 0
    capsys.readouterr()
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    p.stash.writes.clear()
    p.stash.dry_run = True
    policy = tmp_path/'strict.yml'
    policy.write_text('version: 1\nscene_support_ratio: 1.0\n')
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main([*p.args, '--merge-only', '--dry-run', '--temporal-policy', str(policy)]) == 0
    assert 'merged_scene_tags=0' in capsys.readouterr().out
    assert (p.path.read_bytes(), p.stash.scene) == before and not p.stash.writes


def test_both_thresholds_reuse_models_and_round_trip(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    before = saved(p.path)
    configure(monkeypatch, scene=.8, marker=.9)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main(p.args) == 0
    assert saved(p.path)['wd'][0][cache_identity.SCENE_OVERRIDE] == .8
    expected = outputs(p.path)
    assert app.main([*p.args, '--merge-only']) == 0
    assert outputs(p.path) == expected
    configure(monkeypatch)
    assert app.main(p.args) == 0
    assert saved(p.path) == before


def test_explicit_both_thresholds_need_no_assets_and_preserve_dry_run(pipeline, monkeypatch, capsys):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    capsys.readouterr()
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    configure(monkeypatch, scene=.8, marker=.9)
    for name in ('load_tagger', 'load_vocabulary'):
        monkeypatch.setattr(app.services, name, lambda *a: pytest.fail('accessed inference inputs'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *a: pytest.fail('accessed inference inputs'))
    monkeypatch.setattr(p.stash, 'get_media', lambda *a: pytest.fail('fetched media'))
    args = [*p.args, '--merge-only', '--rethreshold-scenes', '--rethreshold-markers']
    p.stash.dry_run = True
    assert app.main([*args, '--dry-run']) == 0
    stderr = capsys.readouterr().err
    assert 'Check scene + marker thresholds' in stderr
    assert 'Apply scene + marker thresholds' in stderr
    assert (p.path.read_bytes(), p.stash.scene) == before
    monkeypatch.setattr(app, 'DRY_RUN', False)
    p.stash.dry_run = False
    assert app.main(args) == 0


@pytest.mark.parametrize('change', ['old_cache', 'below_floor', 'pending', 'missing_result'])
def test_invalid_scene_cache_preflight_does_not_publish(pipeline, monkeypatch, change):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    if change == 'old_cache':
        with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
            segment = state.records('1', 'scene')['wd'][0]
            segment.pop('frames')
            segment['meta'].pop('scene_frames_format')
            segment['meta'].pop('scene_score_floor')
            state.put('1', 'wd', {'scene': segment})
    elif change == 'pending':
        with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
            state.prepare('1', {'scene': state.snapshot('1', 'scene')[0]})
    elif change == 'missing_result':
        with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
            with state.connection:
                state.connection.execute(
                    "DELETE FROM results WHERE scope=? AND scene='1' AND model='wd' AND kind='scene'",
                    (state.scope,))
    before = saved(p.path), deepcopy(p.stash.scene)
    configure(monkeypatch, scene=.3 if change == 'below_floor' else .8)
    args = p.args
    p.stash.writes.clear()
    assert app.main([*args, '--merge-only', '--rethreshold-scenes']) == 1
    assert (saved(p.path), p.stash.scene) == before and not p.stash.writes


def test_rethreshold_scenes_skips_model_with_marker_only_history(pipeline, monkeypatch, capsys):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        marker = state.records('1', 'markers')['wd'][0]
        state.put('1', 'dtq', {'markers': marker})
    capsys.readouterr()

    configure(monkeypatch, scene=.8)
    args = ['--models', 'wd', 'dtq', *p.args[2:], '--merge-only', '--rethreshold-scenes']
    assert app.main(args) == 0
    output = capsys.readouterr().out
    records = saved(p.path)
    assert records['wd'][0][cache_identity.SCENE_OVERRIDE] == pytest.approx(.8)
    assert 'dtq' not in records
    assert ('Threshold preflight: checked=1, failed=0, applied=1, '
            'selected=2, skipped_no_evidence=1') in output


@pytest.mark.parametrize('scene_only', [True, False])
def test_checkpoint_resume_retains_all_scene_frames_once(pipeline, monkeypatch, scene_only):
    p = pipeline
    configure(monkeypatch)
    args = [*p.args, *(['--scene-tags-only'] if scene_only else [])]
    p.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    p.fail.clear()
    assert app.main(args) == 0
    frames = saved(p.path)['wd'][0]['frames']
    assert len(frames) == len(p.cues) and [f['start'] for f in frames] == list(range(10))
    configure(monkeypatch, scene=.8)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main(args) == 0


def test_shared_scene_cleanup_uses_frozen_scene_thresholds(pipeline, monkeypatch, tmp_path, capsys):
    p = pipeline
    configure(monkeypatch)
    for scores in p.scores.values():
        scores[:] = 0
    p.scores['wd'][0, 0] = .62
    p.scores['joytag'][1, 0] = .63
    p.scores['oppai'][2, 0] = .64
    args = ['--models', 'wd', 'joytag', 'oppai', *p.args[2:]]
    assert app.main(args) == 0
    capsys.readouterr()
    configure(monkeypatch, scene=.6)
    assert app.main([*args, '--merge-only', '--rethreshold-scenes']) == 0
    stderr = capsys.readouterr().err
    assert 'Check scene thresholds' in stderr
    assert 'Apply scene thresholds' in stderr
    assert [t['name'] for t in p.stash.scene['tags']] == ['ai:common']
    # YAML alone must not alter the thresholds used by cleanup.
    configure(monkeypatch, scene=.9)
    exclusion = tmp_path/'exclude.yml'
    exclusion.write_text('version: 1\nexcluded_scene_tags: []\nby_model:\n  joytag: [common]\n')
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main([*args, '--cleanup-only', '--scene-tag-exclusions', str(exclusion)]) == 0
    assert [t['name'] for t in p.stash.scene['tags']] == ['ai:common']
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        hit = state.snapshot('1', 'scene')[0]['contributions'][0]
        assert hit['support_sources'] == ['wd', 'oppai']
        assert hit['meta']['scene_threshold'] == .6


def test_old_completed_cache_is_not_eagerly_reinferred(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        segment = state.records('1', 'scene')['wd'][0]
        segment.pop('frames')
        segment['meta'].pop('scene_frames_format')
        segment['meta'].pop('scene_score_floor')
        state.put('1', 'wd', {'scene': segment})
    assert app.main(p.args) == 0 and len(p.loaded) == 1
    configure(monkeypatch, scene=.8)
    assert app.main(p.args) == 0 and len(p.loaded) == 2
    assert 'frames' in saved(p.path)['wd'][0]


@pytest.mark.parametrize('args', [[], ['--merge-only', '--markers-only']])
def test_scene_flag_requires_scene_merge(args):
    with pytest.raises(SystemExit) as exc:
        app.main([*args, '--rethreshold-scenes'])
    assert exc.value.code == 2


def test_scene_rethreshold_never_decodes_publication_snapshots(tmp_path, monkeypatch):
    segment = {
        'mode': 'scene',
        'meta': {
            'origin': 'inference',
            'inference_signature': 'sig',
            'scene_threshold': .4,
            'marker_threshold': .7,
            'scene_frames_format': 1,
            'scene_score_floor': .3,
        },
        'items': [],
        'frames': [
            {'start': 0, 'end': 1, 'tags': [('core', .5)]},
            {'start': 1, 'end': 2, 'tags': [('core', .6)]},
        ],
    }
    config = ModelSettings(SOURCES['wd'], Path('.'), scene_threshold=.8, marker_threshold=.7, batch_size=4)
    with merge_store.StateStore(tmp_path/'state.sqlite3', 'scope') as state:
        state.put('1', 'wd', {'scene': segment})
        state.publish('1', {'scene': {'owned_ids': [], 'contributions': [{'payload': 'x' * 10000}]}})
        monkeypatch.setattr(
            state, 'snapshot',
            lambda *args, **kwargs: pytest.fail('decoded a publication snapshot during rethreshold'))
        report = RunReport(write=lambda _: None)

        result = rethreshold_execution.run(app, state, {'1': {}}, [config], report)

        assert result == {'scene_thresholds_updated': 1}
        assert state.records('1', 'scene')['wd'][0][cache_identity.SCENE_OVERRIDE] == .8
