"""Retune marker publication without inference or loss of source evidence."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from stash_ai.rethreshold import markers as marker_updates, execution as rethreshold_execution
from stash_ai.signatures import digest
from stash_ai.inference import sprites
from stash_ai.cleanup import evidence as cleanup_evidence
from stash_ai import cli as app
from stash_ai import merge
from stash_ai import scene_cache
from stash_ai import thresholds
from stash_ai.runtime import RunReport
from stash_ai.models import ModelSettings, SOURCES
from tests.helpers.evidence import paired_records, policies

pytestmark = pytest.mark.usefixtures('reset_temporal')


def configure(monkeypatch, *, marker=.70, scene=.40):
    # Exercise automatic retention independently of the bundled cache floors.
    original = app.load_model_settings
    monkeypatch.setattr(app, 'load_model_settings', lambda *a: [
        replace(c, marker_threshold=marker, scene_threshold=scene,
                scene_cache_floor=None, marker_cache_floor=None) for c in original(*a)])


def raw(path):
    with sqlite3.connect(path) as db:
        return db.execute('SELECT * FROM results ORDER BY scene,model,kind').fetchall()


def records(path):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True) as store:
        return {kind: store.records('1', kind) for kind in ('scene', 'markers')}


def fail_inference(monkeypatch):
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('loaded a model'))


def test_indexed_signature_combines_threshold_overrides_without_breaking_legacy_hashes():
    segment = {
        'mode': 'frames',
        'meta': {'inference_signature': 'sig-a'},
        'items': [],
    }

    assert thresholds.indexed_signature(segment) == 'sig-a'

    marker_only = {**segment, thresholds.OVERRIDE: .9}
    assert thresholds.indexed_signature(marker_only) == digest(['sig-a', .9])

    scene_only = {**segment, scene_cache.SCENE_OVERRIDE: .8}
    assert thresholds.indexed_signature(scene_only) == digest([
        'sig-a', {'scene_threshold': .8}
    ])

    both = {
        **segment,
        thresholds.OVERRIDE: .9,
        scene_cache.SCENE_OVERRIDE: .8,
    }
    combined = thresholds.indexed_signature(both)
    assert combined == digest([
        'sig-a',
        {'scene_threshold': .8, 'marker_threshold': .9},
    ])
    assert combined != thresholds.indexed_signature({**both, thresholds.OVERRIDE: .95})
    assert combined != thresholds.indexed_signature({**both, scene_cache.SCENE_OVERRIDE: .85})


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


@pytest.mark.parametrize('mode,scene,marker', [
    ('--scene-tags-only', .8, .7),
    ('--markers-only', .4, .9),
    (None, .8, .9),
])
def test_return_to_original_thresholds_after_scene_edit(pipeline, monkeypatch, mode, scene, marker):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:, 0] = .65
    p.scores['wd'][:, 1] = .85
    args = [*p.args, *([mode] if mode else [])]
    assert app.main(args) == 0
    before = records(p.path)
    original_tags = {t['name'] for t in p.stash.scene['tags']}

    def marker_outputs():
        return sorted((m['primary_tag']['name'], m['seconds'], m.get('end_seconds'))
                      for m in p.stash.scene['scene_markers'])

    original_markers = marker_outputs()
    fail_inference(monkeypatch)
    configure(monkeypatch, scene=scene, marker=marker)
    assert app.main(args) == 0
    assert records(p.path) != before

    # A changed scene takes the replacement-history reuse path before the
    # ordinary threshold cache path. Its original signature still matches.
    p.stash.scene['tags'].append(p.stash.tag('manual'))
    configure(monkeypatch)
    assert app.main(args) == 0
    assert records(p.path) == before
    assert {t['name'] for t in p.stash.scene['tags']} == original_tags | {'manual'}
    assert marker_outputs() == original_markers


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


def test_explicit_rethreshold_needs_no_models_media_or_vocabulary(pipeline, monkeypatch, capsys):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    capsys.readouterr()
    configure(monkeypatch, marker=.85)
    fail_inference(monkeypatch)
    monkeypatch.setattr(app, 'load_vocabulary', lambda *a: pytest.fail('loaded vocabulary'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *a: pytest.fail('fetched VTT'))
    monkeypatch.setattr(p.stash, 'get_media', lambda *a: pytest.fail('fetched sprite'))
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers']) == 0
    stderr = capsys.readouterr().err
    assert 'Check marker thresholds' in stderr
    assert 'Apply marker thresholds' in stderr
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


def test_preflight_skips_second_model_with_scene_only_history(pipeline, monkeypatch, capsys):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
        scene = state.records('1', 'scene')['wd'][0]
        state.put('1', 'dtq', {'scene': scene})
    capsys.readouterr()
    configure(monkeypatch, marker=.85)
    p.stash.writes.clear()
    assert app.main(['--models', 'wd', 'dtq', *p.args[2:], '--merge-only', '--rethreshold-markers']) == 0
    output = capsys.readouterr().out
    assert records(p.path)['markers']['wd'][0][thresholds.OVERRIDE] == .85
    assert ('Threshold preflight: checked=1, failed=0, applied=1, '
            'selected=2, skipped_no_evidence=1') in output


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
            assert marker_updates.apply(app, state, '1', config)
        spans = merge.marker_contributions(app, state.records('1', 'markers'), policies())
        # At .9 both weak models vote; removing one must remove the marker.
        assert spans
        excluded = policies(); excluded['wd'] = frozenset({'core'})
        assert cleanup_evidence.cleanup_support(app, state, '1', spans, {'wd'}, excluded) == []
        # The same removal at the old .7 threshold leaves DTQ's strong evidence.
        cleaned = cleanup_evidence.cleanup_support(app, state, '1', old_spans, {'wd'}, excluded)
        assert cleaned and all(s['source'] == 'dtq' for s in cleaned)


@pytest.mark.parametrize('args', [[], ['--scene-tags-only', '--merge-only']])
def test_flag_requires_merge_with_marker_output(args):
    with pytest.raises(SystemExit) as exc:
        app.main([*args, '--rethreshold-markers'])
    assert exc.value.code == 2


def test_marker_rethreshold_never_decodes_publication_snapshots(tmp_path, monkeypatch):
    scenes, markers = paired_records({'wd': [.9, .9, .9]}, threshold=.4, marker_threshold=.7)
    config = ModelSettings(SOURCES['wd'], Path('.'), marker_threshold=.85, scene_threshold=.4)
    with merge.StateStore(tmp_path/'state.sqlite3', 'scope') as state:
        state.put('1', 'wd', {'scene': scenes['wd'][0], 'markers': markers['wd'][0]})
        state.publish('1', {
            'scene': {'owned_ids': [], 'contributions': []},
            'markers': {'contributions': [{'payload': 'x' * 10000}]},
        })
        monkeypatch.setattr(
            state, 'snapshot',
            lambda *args, **kwargs: pytest.fail('decoded a publication snapshot during rethreshold'))
        report = RunReport(write=lambda _: None)

        result = rethreshold_execution.run(app, state, {'1': {}}, [config], report, scenes=False, markers=True)

        assert result == {'marker_thresholds_updated': 1}
        assert state.records('1', 'markers')['wd'][0][thresholds.OVERRIDE] == .85
