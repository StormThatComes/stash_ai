"""Retune marker publication without inference or loss of source evidence."""
from copy import deepcopy
from dataclasses import replace
import sqlite3

import pytest

import stash_ai as app
import stash_merge as merge
import stash_thresholds as thresholds
from stash_taggers import SOURCES
from tests.helpers.evidence import paired_records, policies

pytestmark = pytest.mark.usefixtures('reset_temporal')


def configure(monkeypatch, *, marker=.70, scene=.40):
    original = app.load_model_settings
    monkeypatch.setattr(app, 'load_model_settings', lambda *a: [
        replace(c, marker_threshold=marker, scene_threshold=scene) for c in original(*a)])


def raw(path):
    with sqlite3.connect(path) as db:
        return db.execute('SELECT * FROM results ORDER BY scene,model,kind').fetchall()


def records(path):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True) as store:
        return {kind: store.records('1', kind) for kind in ('scene', 'markers')}


def fail_inference(monkeypatch):
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('loaded a model'))


def test_automatic_raise_and_return_to_original_keep_raw_evidence(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    before = records(p.path)
    fail_inference(monkeypatch)
    configure(monkeypatch, marker=.95)
    assert app.main(p.args) == 0
    after = records(p.path)
    assert after['scene'] == before['scene']
    segment = after['markers']['wd'][0]
    assert segment.pop(thresholds.OVERRIDE) == .95
    assert segment == before['markers']['wd'][0]
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        spans = state.snapshot('1', 'markers')[0]['contributions']
        assert all(s.get('derived') == 'occurrence' for s in spans)
    # An ordinary merge retains the chosen threshold without the new flag.
    p.stash.writes.clear()
    assert app.main([*p.args, '--merge-only']) == 0
    assert not p.stash.writes
    configure(monkeypatch, marker=.70)
    assert app.main(p.args) == 0
    assert records(p.path) == before
    assert app.main(p.args) == 0


@pytest.mark.parametrize('temporal', [True, False])
def test_rethreshold_matches_fresh_marker_inference(pipeline, monkeypatch, tmp_path, temporal):
    p = pipeline
    configure(monkeypatch)
    args = [*p.args, '--markers-only', *([] if temporal else ['--no-temporal'])]
    assert app.main(args) == 0
    configure(monkeypatch, marker=.88)
    assert app.main([*args, '--merge-only', '--rethreshold-markers']) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        _, cached = merge.evidence(app, state, '1', policies(), policies())
    # A separate reference DB must start with a fresh Stash fixture as well.
    p.stash.scene['scene_markers'] = []
    assert app.main([*args, '--state-db', str(tmp_path/'fresh.sqlite3'), '--restart']) == 0
    with merge.StateStore(tmp_path/'fresh.sqlite3', merge.scope_for(app.STASH_URL), True) as state:
        _, fresh = merge.evidence(app, state, '1', policies(), policies())
    def outputs(spans):
        return [(s['tag'], s['start'], s['end'], s['score'], s['hits'], s['extra_tags']) for s in spans]
    assert outputs(cached) == outputs(fresh)


def test_explicit_rethreshold_needs_no_models_media_or_vocabulary(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    configure(monkeypatch, marker=.85)
    fail_inference(monkeypatch)
    monkeypatch.setattr(app, 'load_vocabulary', lambda *a: pytest.fail('loaded vocabulary'))
    monkeypatch.setattr(app, 'fetch_cues', lambda *a: pytest.fail('fetched VTT'))
    monkeypatch.setattr(p.stash, 'get_media', lambda *a: pytest.fail('fetched sprite'))
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers']) == 0
    assert records(p.path)['markers']['wd'][0][thresholds.OVERRIDE] == .85


@pytest.mark.parametrize('change', ['lower', 'scene', 'legacy', 'missing', 'pending'])
def test_unsafe_selection_is_rejected_before_any_policy_or_stash_write(pipeline, monkeypatch, change):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    if change in ('legacy', 'missing', 'pending'):
        with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
            if change == 'legacy':
                state.put('1', 'wd', {'markers': dict(mode='spans', meta={'origin': 'legacy'}, items=[])})
            elif change == 'missing':
                with state.connection:
                    state.connection.execute("DELETE FROM results WHERE model='wd' AND kind='markers'")
            else:
                state.prepare('1', {'markers': state.snapshot('1', 'markers')[0]})
    configure(monkeypatch, marker=.50 if change == 'lower' else .85,
              scene=.45 if change == 'scene' else .40)
    before = raw(p.path), deepcopy(p.stash.scene)
    p.stash.writes.clear()
    fail_inference(monkeypatch)
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers']) == 1
    assert (raw(p.path), p.stash.scene) == before
    assert not p.stash.writes


def test_dry_run_and_model_selection(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(['--models', 'wd', 'dtq', *p.args[2:]]) == 0
    before = p.path.read_bytes(), raw(p.path), deepcopy(p.stash.scene)
    configure(monkeypatch, marker=.85)
    p.stash.dry_run = True
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers', '--dry-run']) == 0
    assert (p.path.read_bytes(), raw(p.path), p.stash.scene) == before
    monkeypatch.setattr(app, 'DRY_RUN', False)
    p.stash.dry_run = False
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers']) == 0
    saved = records(p.path)['markers']
    assert saved['wd'][0][thresholds.OVERRIDE] == .85
    assert thresholds.OVERRIDE not in saved['dtq'][0]


def test_preflight_rejects_missing_second_model_before_updating_first(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    before = raw(p.path)
    configure(monkeypatch, marker=.85)
    p.stash.writes.clear()
    assert app.main(['--models', 'wd', 'dtq', *p.args[2:], '--merge-only', '--rethreshold-markers']) == 1
    assert raw(p.path) == before and not p.stash.writes


def test_failed_publication_resumes_with_saved_threshold_without_models(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    configure(monkeypatch, marker=.85)
    p.stash.scene['scene_markers'] = []
    p.stash.fail_create = True
    fail_inference(monkeypatch)
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers']) == 1
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.has_pending_publications()
    p.stash.fail_create = False
    assert app.main(p.args) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert not state.has_pending_publications()
        assert state.records('1', 'markers')['wd'][0][thresholds.OVERRIDE] == .85


@pytest.mark.parametrize('change', ['media', 'restart', 'below_floor'])
def test_automatic_reuse_still_requires_matching_inputs_and_retention(pipeline, monkeypatch, change):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    configure(monkeypatch, marker=.50 if change == 'below_floor' else .85,
              scene=.45 if change == 'scene' else .40)
    if change == 'media':
        monkeypatch.setattr(p.stash, 'get_media', lambda url: b'new media')
    assert app.main([*p.args, *(['--restart'] if change == 'restart' else [])]) == 0
    assert len(p.loaded) == 2


def test_cleanup_replays_published_threshold_even_after_later_override(tmp_path):
    scenes, markers = paired_records({'wd': [.82, .82, .82], 'dtq': [.82, .82, .82]},
                                    threshold=.4, marker_threshold=.7)
    with merge.StateStore(tmp_path/'state.sqlite3', 'test') as state:
        for model in markers:
            state.put('1', model, {'scene': scenes[model][0], 'markers': markers[model][0]})
        old_spans = merge.marker_contributions(app, state.records('1', 'markers'), policies())
        for model in markers:
            config = type('Config', (), dict(source=SOURCES[model], marker_threshold=.9, scene_threshold=.4))
            assert thresholds.apply(app, state, '1', config)
        spans = merge.marker_contributions(app, state.records('1', 'markers'), policies())
        # At .9 both weak models vote; removing one must remove the marker.
        assert spans
        excluded = policies(); excluded['wd'] = frozenset({'core'})
        assert merge.cleanup_support(app, state, '1', spans, {'wd'}, excluded) == []
        # The same removal at the old .7 threshold leaves DTQ's strong evidence.
        cleaned = merge.cleanup_support(app, state, '1', old_spans, {'wd'}, excluded)
        assert cleaned and all(s['source'] == 'dtq' for s in cleaned)


@pytest.mark.parametrize('args', [[], ['--scene-tags-only', '--merge-only']])
def test_flag_requires_merge_with_marker_output(args):
    with pytest.raises(SystemExit) as exc:
        app.main([*args, '--rethreshold-markers'])
    assert exc.value.code == 2


def test_indexed_signature_includes_both_threshold_overrides():
    """Marker policy must remain visible in the progress index when scene override is also set."""
    base = {
        'mode': 'frames',
        'items': [],
        'meta': {'inference_signature': 'sig-a', 'marker_threshold': 0.7, 'scene_threshold': 0.4},
    }
    plain = thresholds.indexed_signature(base)
    marker_only = thresholds.indexed_signature({**base, thresholds.OVERRIDE: 0.9})
    scene_only = thresholds.indexed_signature({**base, 'scene_threshold_override': 0.8})
    both = thresholds.indexed_signature({**base, thresholds.OVERRIDE: 0.9, 'scene_threshold_override': 0.8})
    both_marker_changed = thresholds.indexed_signature(
        {**base, thresholds.OVERRIDE: 0.95, 'scene_threshold_override': 0.8})
    assert plain != marker_only != scene_only
    assert both != scene_only, 'marker override must affect signature when scene override is present'
    assert both != marker_only
    assert both != both_marker_changed
