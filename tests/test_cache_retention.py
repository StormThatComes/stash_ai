"""Wider cache must support offline retuning without changing publication."""

from copy import deepcopy
from dataclasses import replace

import pytest
import yaml

from stash_ai.inference.identity import model_identity
from tests.helpers.runtime import runtime as app
from stash_ai.models import ModelSettings, SOURCES
from tests.test_marker_retention import marker_ranges, offline
from tests.test_scene_thresholds import outputs
from tests.test_thresholds import raw, records

pytestmark = pytest.mark.usefixtures('reset_temporal')


def model_config(path, *, scene=.8, marker=.85, **retention):
    path.write_text(yaml.safe_dump(dict(version=1, device='cpu', **{
        source: dict(scene_threshold=scene, marker_threshold=marker,
                     batch_size=2, **retention) for source in SOURCES})))
    return ['--model-config', str(path)]


@pytest.mark.parametrize('name', ['scene_cache_floor', 'marker_cache_floor'])
@pytest.mark.parametrize('value', [True, '0.2', -.1, 1.1, float('nan'), float('inf')])
def test_invalid_retention_is_rejected_by_yaml_loader(tmp_path, name, value):
    path = tmp_path/'models.yml'
    model_config(path, **{name: value})
    with pytest.raises(ValueError, match=f'wd.{name} must be null or a number between 0 and 1'):
        app.services.load_model_settings(path, ['wd'], None)


@pytest.mark.parametrize('value', [None, 0, .2, 1])
def test_retention_settings_apply_to_all_models(tmp_path, value):
    path = tmp_path/'models.yml'
    model_config(path, scene_cache_floor=value, marker_cache_floor=value)
    settings = app.services.load_model_settings(path, ['all'], None)
    assert [s.source.key for s in settings] == list(SOURCES)
    assert all(s.scene_cache_floor == s.marker_cache_floor == value for s in settings)


def test_omitted_retention_preserves_legacy_model_identity(tmp_path, pipeline):
    config = ModelSettings(SOURCES['wd'], tmp_path)
    expected = model_identity(app, config, pipeline.vocab)
    assert expected['retention'] == dict(temporal=True, marker_ratio=.85, scene_ratio=.85)
    assert model_identity(app, replace(config, scene_cache_floor=None,
        marker_cache_floor=None), pipeline.vocab) == expected
    for name in ('scene_cache_floor', 'marker_cache_floor'):
        assert model_identity(app, replace(config, **{name: .2}), pipeline.vocab) != expected


@pytest.mark.parametrize('temporal', [True, False])
@pytest.mark.parametrize('aggregation', ['max', 'topk_mean'])
def test_wider_cache_changes_no_publication_and_offline_lowering_matches_fresh(
        pipeline, monkeypatch, tmp_path, temporal, aggregation):
    p = pipeline
    monkeypatch.setattr(app, 'AGGREGATION', aggregation)
    p.scores['wd'][:, 0] = .5
    p.scores['wd'][:, 1] = .95
    path = tmp_path/'models.yml'
    args = [*p.args, *model_config(path), *([] if temporal else ['--no-temporal'])]
    assert app.main(args) == 0
    baseline = outputs(p.path), marker_ranges(p.stash)
    assert {t['name'] for t in p.stash.scene['tags']} == {'ai:extra'}

    model_config(path, scene_cache_floor=.2, marker_cache_floor=.25)
    assert app.main(args) == 0
    assert len(p.loaded) == 2  # Adding retention populates the previously missing scores.
    assert (outputs(p.path), marker_ranges(p.stash)) == baseline
    saved = records(p.path)
    scene, marker = saved['scene']['wd'][0], saved['markers']['wd'][0]
    assert scene['meta']['scene_score_floor'] == .2
    assert marker['meta']['marker_score_floor'] == .25
    assert {item['tag'] for item in scene['items']} == {'common', 'extra'}
    assert all(dict(row['tags'])['common'] == .5 for row in scene['frames'])
    assert all(dict(row['tags'])['common'] == .5 for row in marker['items'])
    original = raw(p.path)

    model_config(path, scene=.4, marker=.4, scene_cache_floor=.2, marker_cache_floor=.25)
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        # Dry-run must keep both the durable evidence and Stash unchanged.
        before = p.path.read_bytes(), deepcopy(p.stash.scene)
        p.stash.dry_run = True
        retune = [*args, '--merge-only', '--rethreshold-scenes', '--rethreshold-markers']
        assert app.main([*retune, '--dry-run']) == 0
        assert (p.path.read_bytes(), p.stash.scene) == before
        p.stash.dry_run = False
        monkeypatch.setattr(app, 'DRY_RUN', False)
        assert app.main(retune) == 0
        cached = outputs(p.path), marker_ranges(p.stash)
        assert raw(p.path) == original
        assert {t['name'] for t in p.stash.scene['tags']} == {'ai:common', 'ai:extra'}

    p.stash.scene['tags'], p.stash.scene['scene_markers'] = [], []
    fresh = tmp_path/'fresh.sqlite3'
    assert app.main([*args, '--state-db', str(fresh), '--restart']) == 0
    assert (outputs(fresh), marker_ranges(p.stash)) == cached


@pytest.mark.parametrize('kind', ['scene', 'markers'])
def test_individual_retention_supports_automatic_threshold_reuse(pipeline, monkeypatch, tmp_path, kind):
    p = pipeline
    p.scores['wd'][:] = .5
    path = tmp_path/'models.yml'
    options = dict(scene_cache_floor=.2) if kind == 'scene' else dict(marker_cache_floor=.2)
    args = [*p.args, *model_config(path, **options),
            '--scene-tags-only' if kind == 'scene' else '--markers-only']
    assert app.main(args) == 0
    model_config(path, scene=.4 if kind == 'scene' else .8,
                 marker=.4 if kind == 'markers' else .85, **options)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('loaded a model'))
    if kind == 'scene':
        assert app.main(args) == 0
        assert {t['name'] for t in p.stash.scene['tags']} == {'ai:common', 'ai:extra'}
    else:
        # Marker-only history uses explicit rethreshold without scene evidence.
        assert app.main([*args, '--merge-only', '--rethreshold-markers']) == 0
        assert marker_ranges(p.stash) == [('ai:common', 0, 10), ('ai:extra', 0, 10)]


@pytest.mark.parametrize('floor', [None, .2, .95])
@pytest.mark.parametrize('temporal', [True, False])
def test_retention_never_discards_current_publication_inputs(pipeline, tmp_path, floor, temporal):
    p = pipeline
    args = [*p.args, *model_config(tmp_path/'models.yml', scene_cache_floor=floor,
        marker_cache_floor=floor), *([] if temporal else ['--no-temporal'])]
    assert app.main(args) == 0
    saved = records(p.path)
    required_scene, required_marker = (.68, .7225) if temporal else (.8, .85)
    assert saved['scene']['wd'][0]['meta']['scene_score_floor'] == pytest.approx(
        min(required_scene, floor) if floor is not None else required_scene)
    meta = saved['markers']['wd'][0]['meta']
    assert meta.get('marker_score_floor', meta['marker_threshold']) == pytest.approx(
        min(required_marker, floor) if floor is not None else required_marker)


@pytest.mark.parametrize('kind', ['scene', 'markers'])
def test_lowering_below_retained_floor_still_refuses_publication(pipeline, tmp_path, kind):
    p = pipeline
    path = tmp_path/'models.yml'
    args = [*p.args, *model_config(path, scene_cache_floor=.2, marker_cache_floor=.2)]
    assert app.main(args) == 0
    before = raw(p.path), deepcopy(p.stash.scene)
    p.stash.writes.clear()
    model_config(path, scene=.15 if kind == 'scene' else .8,
        marker=.15 if kind == 'markers' else .85, scene_cache_floor=.2, marker_cache_floor=.2)
    assert app.main([*args, '--merge-only', f'--rethreshold-{kind if kind == "markers" else "scenes"}']) == 1
    assert (raw(p.path), p.stash.scene) == before and not p.stash.writes


@pytest.mark.parametrize('changed', [False, True])
def test_checkpoint_retains_weak_scores_and_retention_edit_restarts_it(pipeline, tmp_path, changed):
    p = pipeline
    p.scores['wd'][:] = .5
    path = tmp_path/'models.yml'
    args = [*p.args, *model_config(path, scene_cache_floor=.2, marker_cache_floor=.25)]
    p.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    assert p.loaded[-1].seen == [0, 1, 2, 3]
    p.fail.clear()
    if changed:
        model_config(path, scene_cache_floor=.1, marker_cache_floor=.25)
    assert app.main(args) == 0
    assert p.loaded[-1].seen == list(range(0 if changed else 4, 10))
    saved = records(p.path)
    for rows in (saved['scene']['wd'][0]['frames'], saved['markers']['wd'][0]['items']):
        assert [row['start'] for row in rows] == list(range(10))
        assert all(dict(row['tags']) == dict(common=.5, extra=.5) for row in rows)
    fresh = tmp_path/'fresh.sqlite3'
    assert app.main([*args, '--state-db', str(fresh), '--restart']) == 0
    assert outputs(p.path) == outputs(fresh)


def test_wider_marker_retention_can_replay_lower_continuation_ratio(pipeline, monkeypatch, tmp_path):
    p = pipeline
    p.scores['wd'][:] = 0
    p.scores['wd'][:2, 0] = p.scores['wd'][6:8, 0] = .95
    p.scores['wd'][2:6, 0] = .64
    args = [*p.args, *model_config(tmp_path/'models.yml', marker_cache_floor=.2), '--markers-only']
    assert app.main(args) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 2), ('ai:common', 6, 8)]
    policy = tmp_path/'policy.yml'
    policy.write_text('version: 1\nmarker_continue_ratio: 0.7\n')
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        assert app.main([*args, '--merge-only', '--temporal-policy', str(policy)]) == 0
        assert marker_ranges(p.stash) == [('ai:common', 0, 8)]
    p.stash.scene['scene_markers'] = []
    assert app.main([*args, '--state-db', str(tmp_path/'fresh.sqlite3'), '--restart',
        '--temporal-policy', str(policy)]) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 8)]


@pytest.mark.parametrize('temporal', [True, False])
def test_separate_output_ignores_persistent_retention_settings(pipeline, monkeypatch, tmp_path, temporal):
    p = pipeline
    monkeypatch.setattr(app, 'DEFAULT_RESUME', False)
    path = tmp_path/'models.yml'
    args = [*p.args, *model_config(path), '--output-mode', 'separate',
            *([] if temporal else ['--no-temporal'])]
    assert app.main(args) == 0
    before = {t['name'] for t in p.stash.scene['tags']}, marker_ranges(p.stash)
    model_config(path, scene_cache_floor=.1, marker_cache_floor=.1)
    assert app.main(args) == 0
    assert ({t['name'] for t in p.stash.scene['tags']}, marker_ranges(p.stash)) == before


def test_retained_weak_scores_do_not_become_consensus_votes(pipeline, monkeypatch, tmp_path):
    p = pipeline
    for scores in p.scores.values():
        scores[:, 0] = .5
        scores[:, 1] = .95
    path = tmp_path/'models.yml'
    args = ['--models', 'all', *p.args[2:], *model_config(path)]
    assert app.main(args) == 0
    baseline = outputs(p.path), marker_ranges(p.stash)
    model_config(path, scene_cache_floor=.2, marker_cache_floor=.1)
    assert app.main(args) == 0
    assert (outputs(p.path), marker_ranges(p.stash)) == baseline
    assert {t['name'] for t in p.stash.scene['tags']} == {'ai:extra'}
    assert all(tag == 'ai:extra' for tag, *_ in marker_ranges(p.stash))
    model_config(path, scene=.4, marker=.4, scene_cache_floor=.2, marker_cache_floor=.1)
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        assert app.main([*args, '--merge-only', '--rethreshold-scenes', '--rethreshold-markers']) == 0
        cached = outputs(p.path), marker_ranges(p.stash)
    p.stash.scene['tags'], p.stash.scene['scene_markers'] = [], []
    fresh = tmp_path/'fresh.sqlite3'
    assert app.main([*args, '--state-db', str(fresh), '--restart']) == 0
    assert (outputs(fresh), marker_ranges(p.stash)) == cached


def test_exact_scene_aggregate_includes_scores_below_retention(pipeline, monkeypatch, tmp_path):
    p = pipeline
    # A single retained observation cannot stand in for the exact top-k mean.
    monkeypatch.setattr(app, 'TOP_K_FRAMES', 5)
    p.scores['wd'][:] = .1
    p.scores['wd'][0, 0] = .5
    args = [*p.args, *model_config(tmp_path/'models.yml', scene_cache_floor=.2), '--scene-tags-only']
    assert app.main(args) == 0
    scene = records(p.path)['scene']['wd'][0]
    assert scene['items'] == [dict(tag='common', score=pytest.approx(.18))]
    assert len([row for row in scene['frames'] if row['tags']]) == 1

