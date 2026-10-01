"""Cached marker coverage must survive changes to the inference-time blocklist."""
from copy import deepcopy
from dataclasses import replace

import pytest

from stash_ai import cli as app, merge, thresholds
from tests.helpers.evidence import segment

pytestmark = pytest.mark.usefixtures('reset_temporal')


def configure(monkeypatch):
    original = app.load_model_settings
    monkeypatch.setattr(app, 'load_model_settings', lambda *args: [
        replace(config, scene_threshold=.4, marker_threshold=.7)
        for config in original(*args)])


@pytest.mark.parametrize('temporal', [True, False])
@pytest.mark.parametrize('markers_only', [True, False])
@pytest.mark.parametrize('dry_run', [True, False])
def test_unblocking_rejects_merge_before_writes_and_restart_restores_range(
        pipeline, monkeypatch, capsys, temporal, markers_only, dry_run):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = 0
    p.scores['wd'][:, 0] = .9
    args = [*p.args, *(['--markers-only'] if markers_only else []),
            *([] if temporal else ['--no-temporal'])]
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {'common', 'unrelated'})
    assert app.main(args) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        saved = state.records('1', 'markers')['wd'][0]
    assert saved['meta']['marker_tag_blocklist'] == ['common', 'unrelated']
    assert all(not row['tags'] for row in saved['items'])
    before = deepcopy(p.stash.scene)
    p.stash.writes.clear()
    capsys.readouterr()
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {'unrelated'})
    with monkeypatch.context() as guard:
        guard.setattr(app, 'load_tagger', lambda *args: pytest.fail('loaded a model during merge'))
        assert app.main([*args, '--merge-only', *(['--dry-run'] if dry_run else [])]) == 1
    output = capsys.readouterr().out
    assert 'unblocked marker tags' in output and 'common' in output and '--restart' in output
    assert p.stash.scene == before and not p.stash.writes
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.records('1', 'markers')['wd'][0] == saved
    assert app.main([*args, '--restart']) == 0
    assert [(m['seconds'], m.get('end_seconds')) for m in p.stash.scene['scene_markers']
            if m['primary_tag']['name'] == 'ai:common'] == [(0, 10)]


@pytest.mark.parametrize('combined', [True, False])
def test_unblocking_aborts_rethreshold_before_any_policy_write(pipeline, monkeypatch, combined):
    p = pipeline
    configure(monkeypatch)
    args = ['--models', 'wd', 'dtq', *p.args[2:]]
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', set())
    assert app.main(args) == 0
    # WD is a safe earlier target; DTQ contains frames captured while common was blocked.
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'markers')['dtq']
        records[0]['meta']['marker_tag_blocklist'] = ['common']
        for row in records[0]['items']:
            row['tags'] = [pair for pair in row['tags'] if pair[0] != 'common']
        with state.connection:
            state.replace_records('1', 'dtq', 'markers', records)
        before = state.connection.execute('SELECT * FROM results ORDER BY model,kind').fetchall()
        policies = state.connection.execute('SELECT * FROM result_thresholds').fetchall()
    original = app.load_model_settings
    monkeypatch.setattr(app, 'load_model_settings', lambda *values: [
        replace(config, marker_threshold=.9, scene_threshold=.8 if combined else .4)
        for config in original(*values)])
    p.stash.writes.clear()
    with monkeypatch.context() as guard:
        guard.setattr(app, 'load_tagger', lambda *values: pytest.fail('loaded a model'))
        assert app.main([*args, '--merge-only', '--rethreshold-markers',
                         *(['--rethreshold-scenes'] if combined else [])]) == 1
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.connection.execute(
            'SELECT * FROM ' + state.table('results') + ' ORDER BY model,kind').fetchall() == before
        assert state.connection.execute(
            'SELECT * FROM ' + state.table('result_thresholds')).fetchall() == policies
    assert not p.stash.writes


def test_unchanged_and_stricter_blocklists_reuse_cache_without_models(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:, 0] = .9
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {'unrelated'})
    assert app.main(p.args) == 0
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('loaded a model'))
    assert app.main([*p.args, '--merge-only', '--force-merge']) == 0
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {'common', 'unrelated'})
    assert app.main([*p.args, '--merge-only']) == 0
    assert all(m['primary_tag']['name'] != 'ai:common' for m in p.stash.scene['scene_markers'])
    # Compare against the capture policy, not the stricter last publication.
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {'unrelated'})
    assert app.main([*p.args, '--merge-only']) == 0
    assert any(m['primary_tag']['name'] == 'ai:common' for m in p.stash.scene['scene_markers'])


@pytest.mark.parametrize('retained', [None, 'common', [None]])
def test_unknown_blocklist_rejects_marker_replay_but_not_scene_only(
        pipeline, monkeypatch, retained):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'markers')['wd']
        if retained is None:
            records[0]['meta'].pop('marker_tag_blocklist')
        else:
            records[0]['meta']['marker_tag_blocklist'] = retained
        with state.connection:
            state.replace_records('1', 'wd', 'markers', records)
    p.stash.writes.clear()
    assert app.main([*p.args, '--merge-only', '--force-merge']) == 1
    assert not p.stash.writes
    assert app.main([*p.args, '--scene-tags-only', '--merge-only']) == 0


def test_each_history_segment_is_checked_and_imported_ranges_keep_their_semantics():
    old = segment('wd', [[('common', .9)]] * 2)
    old['meta']['marker_tag_blocklist'] = ['missing']
    current = segment('wd', [[('common', .9)]] * 2, identity='new')
    with pytest.raises(ValueError, match='unblocked marker tags'):
        thresholds.validate_marker_cache(app, {'wd': [old, current]})
    imported = dict(mode='spans', meta={'origin': 'legacy'}, items=[])
    thresholds.validate_marker_cache(app, {'wd': [imported, current]})


def test_blocklist_preflight_reasons_are_stable():
    assert thresholds.threshold_failure_reason(ValueError(
        '123/wd: unblocked marker tags [\'common\'] are absent from cached frames; run inference.')) == (
            'unblocked marker tags require inference')
    assert thresholds.threshold_failure_reason(ValueError(
        '123/wd: marker blocklist metadata is missing or invalid; run inference.')) == 'unknown marker blocklist'
