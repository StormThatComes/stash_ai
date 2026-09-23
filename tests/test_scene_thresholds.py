"""Scene threshold replay, provenance and restart/resume equivalence."""
from copy import deepcopy
from dataclasses import replace

import pytest

import stash_ai as app
import stash_merge as merge
import stash_scene_cache as cache
from tests.helpers.evidence import policies

pytestmark = pytest.mark.usefixtures('reset_temporal')


def configure(monkeypatch, scene=.4, marker=.7):
    original = app.load_model_settings
    monkeypatch.setattr(app, 'load_model_settings', lambda *a: [
        replace(c, scene_threshold=scene, marker_threshold=marker) for c in original(*a)])


def saved(path):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True) as state:
        return state.records('1', 'scene')


def outputs(path):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True) as state:
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
    args = [*p.args, *(['--scene-tags-only'] if scene_only else []), *([] if temporal else ['--no-temporal'])]
    assert app.main(args) == 0
    original = saved(p.path)
    configure(monkeypatch, scene=.8)
    assert app.main([*args, '--merge-only', '--rethreshold-scenes']) == 0
    cached = outputs(p.path)
    value = saved(p.path)['wd'][0]
    assert value.pop(cache.SCENE_OVERRIDE) == .8
    assert value == original['wd'][0]
    p.stash.scene['scene_markers'] = []
    p.stash.scene['tags'] = []
    fresh = tmp_path/'fresh.sqlite3'
    assert app.main([*args, '--state-db', str(fresh), '--restart']) == 0
    assert outputs(fresh) == cached


def test_both_thresholds_reuse_models_and_round_trip(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    before = saved(p.path)
    configure(monkeypatch, scene=.8, marker=.9)
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main(p.args) == 0
    assert saved(p.path)['wd'][0][cache.SCENE_OVERRIDE] == .8
    expected = outputs(p.path)
    assert app.main([*p.args, '--merge-only']) == 0
    assert outputs(p.path) == expected
    configure(monkeypatch)
    assert app.main(p.args) == 0
    assert saved(p.path) == before


def test_explicit_both_thresholds_need_no_assets_and_preserve_dry_run(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    configure(monkeypatch, scene=.8, marker=.9)
    for name in ('load_tagger', 'load_vocabulary', 'fetch_cues'):
        monkeypatch.setattr(app, name, lambda *a: pytest.fail('accessed inference inputs'))
    monkeypatch.setattr(p.stash, 'get_media', lambda *a: pytest.fail('fetched media'))
    args = [*p.args, '--merge-only', '--rethreshold-scenes', '--rethreshold-markers']
    p.stash.dry_run = True
    assert app.main([*args, '--dry-run']) == 0
    assert (p.path.read_bytes(), p.stash.scene) == before
    monkeypatch.setattr(app, 'DRY_RUN', False)
    p.stash.dry_run = False
    assert app.main(args) == 0


@pytest.mark.parametrize('change', ['old_cache', 'below_floor', 'pending', 'missing_model'])
def test_invalid_scene_cache_preflight_does_not_publish(pipeline, monkeypatch, change):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    if change == 'old_cache':
        with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
            segment = state.records('1', 'scene')['wd'][0]
            segment.pop('frames')
            segment['meta'].pop('scene_frames_format')
            segment['meta'].pop('scene_score_floor')
            state.put('1', 'wd', {'scene': segment})
    elif change == 'pending':
        with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
            state.prepare('1', {'scene': state.snapshot('1', 'scene')[0]})
    before = saved(p.path), deepcopy(p.stash.scene)
    configure(monkeypatch, scene=.3 if change == 'below_floor' else .8)
    args = ['--models', 'wd', 'dtq', *p.args[2:]] if change == 'missing_model' else p.args
    p.stash.writes.clear()
    assert app.main([*args, '--merge-only', '--rethreshold-scenes']) == 1
    assert (saved(p.path), p.stash.scene) == before and not p.stash.writes


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
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main(args) == 0


def test_shared_scene_cleanup_uses_frozen_scene_thresholds(pipeline, monkeypatch, tmp_path):
    p = pipeline
    configure(monkeypatch)
    for scores in p.scores.values():
        scores[:] = 0
    p.scores['wd'][0, 0] = .62
    p.scores['joytag'][1, 0] = .63
    p.scores['oppai'][2, 0] = .64
    args = ['--models', 'wd', 'joytag', 'oppai', *p.args[2:]]
    assert app.main(args) == 0
    configure(monkeypatch, scene=.6)
    assert app.main([*args, '--merge-only', '--rethreshold-scenes']) == 0
    assert [t['name'] for t in p.stash.scene['tags']] == ['ai:common']
    # YAML alone must not alter the thresholds used by cleanup.
    configure(monkeypatch, scene=.9)
    exclusion = tmp_path/'exclude.yml'
    exclusion.write_text('version: 1\nexcluded_scene_tags: []\nby_model:\n  joytag: [common]\n')
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('loaded model'))
    assert app.main([*args, '--cleanup-only', '--scene-tag-exclusions', str(exclusion)]) == 0
    assert [t['name'] for t in p.stash.scene['tags']] == ['ai:common']
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        hit = state.snapshot('1', 'scene')[0]['contributions'][0]
        assert hit['support_sources'] == ['wd', 'oppai']
        assert hit['meta']['scene_threshold'] == .6


def test_old_completed_cache_is_not_eagerly_reinferred(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
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
