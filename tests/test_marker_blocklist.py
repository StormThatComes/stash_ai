"""The retired blocklist must not silently restore labels lost from legacy caches."""

from tests.helpers.temporal import policy_args

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from tests.helpers.evidence import inject_records

import pytest

from stash_ai.rethreshold import targets as rethreshold_targets
from stash_ai.inference.identity import model_identity
from stash_ai.signatures import digest
from stash_ai.inference import sprites
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import thresholds as thresholds
import stash_ai.storage.store as merge_store
from stash_ai.storage import progress
from stash_ai.models import ModelSettings, Vocabulary, WD
from stash_ai.evidence.temporal.policy import TemporalPolicy
from tests.helpers.evidence import segment

pytestmark = pytest.mark.usefixtures('reset_temporal')


def configure(monkeypatch):
    original = app.services.load_model_settings
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *args: [
        replace(config, scene_threshold=.4, marker_threshold=.7)
        for config in original(*args)])


def test_current_inference_identity_drops_retired_hash_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', TemporalPolicy())
    config = ModelSettings(WD, tmp_path / 'missing', batch_size=2, scene_threshold=.4, marker_threshold=.7)
    identity = model_identity(app, config, Vocabulary(['common', 'extra'], {}))
    assert identity['format'] == 3
    assert 'frame_blocklist' not in identity and 'temporal' not in identity['retention']
    old = {**identity, 'format': 2, 'frame_blocklist': []}
    assert digest(identity) != digest(old)


@pytest.mark.parametrize('temporal', [True, False])
@pytest.mark.parametrize('markers_only', [True, False])
@pytest.mark.parametrize('dry_run', [True, False])
def test_legacy_blocklist_rejects_merge_before_writes_and_restart_restores_range(
        pipeline, monkeypatch, capsys, temporal, markers_only, dry_run):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = 0
    p.scores['wd'][:, 0] = .9
    args = [*p.args, *(['--markers-only'] if markers_only else []),
            *policy_args(p.path.parent, temporal)]
    assert app.main(args) == 0
    # Emulate frames captured by the old option, which discarded labels before storage.
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'markers')['wd']
        saved = records
        saved['meta']['analysis_format'] = 3
        saved['meta']['marker_tag_blocklist'] = ['common', 'unrelated']
        for row in saved['items']:
            row['tags'] = []
        with state.connection:
            inject_records(state, '1', 'wd', {'markers': records})
    assert saved['meta']['marker_tag_blocklist'] == ['common', 'unrelated']
    assert all(not row['tags'] for row in saved['items'])
    before = deepcopy(p.stash.scene)
    p.stash.writes.clear()
    capsys.readouterr()
    with monkeypatch.context() as guard:
        guard.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('loaded a model during merge'))
        assert app.main([*args, '--merge-only', *(['--dry-run'] if dry_run else [])]) == 1
    output = capsys.readouterr().out
    assert 'Obsolete active inference' in output and '--db-update' in output
    assert p.stash.scene == before and not p.stash.writes
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        assert state.unpack(state.connection.execute("SELECT data FROM " + state.table('results') + " WHERE kind='markers' AND model='wd'").fetchone()[0]) == saved
    import sqlite3
    from stash_ai.db import update_database
    with sqlite3.connect(p.path) as connection:
        connection.execute('PRAGMA user_version=14')
    update_database(p.path)
    assert app.main([*args, '--restart']) == 0
    assert [(m['seconds'], m.get('end_seconds')) for m in p.stash.scene['scene_markers']
            if m['primary_tag']['name'] == 'ai:common'] == [(0, 10)]


@pytest.mark.parametrize('combined', [True, False])
def test_legacy_blocklist_aborts_rethreshold_before_any_policy_write(pipeline, monkeypatch, combined):
    p = pipeline
    configure(monkeypatch)
    args = ['--models', 'wd', 'dtq', *p.args[2:]]
    assert app.main(args) == 0
    # WD is a safe earlier target; DTQ contains frames captured while common was blocked.
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'markers')['dtq']
        records['meta']['analysis_format'] = 3
        records['meta']['marker_tag_blocklist'] = ['common']
        for row in records['items']:
            row['tags'] = [pair for pair in row['tags'] if pair[0] != 'common']
        with state.connection:
            inject_records(state, '1', 'dtq', {'markers': records})
        before = state.connection.execute('SELECT * FROM results ORDER BY model,kind').fetchall()
        policies = state.connection.execute('SELECT * FROM result_thresholds').fetchall()
    original = app.services.load_model_settings
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *values: [
        replace(config, marker_threshold=.9, scene_threshold=.8 if combined else .4)
        for config in original(*values)])
    p.stash.writes.clear()
    with monkeypatch.context() as guard:
        guard.setattr(app.services, 'load_tagger', lambda *values: pytest.fail('loaded a model'))
        assert app.main([*args, '--merge-only', '--rethreshold-markers',
                         *(['--rethreshold-scenes'] if combined else [])]) == 1
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        assert state.connection.execute(
            'SELECT * FROM ' + state.table('results') + ' ORDER BY model,kind').fetchall() == before
        assert state.connection.execute(
            'SELECT * FROM ' + state.table('result_thresholds')).fetchall() == policies
    assert not p.stash.writes


def test_unfiltered_cache_retains_labels_and_reuses_predictions_without_models(pipeline, monkeypatch, tmp_path):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:, 0] = .9
    assert app.main(p.args) == 0
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        saved = state.records('1', 'markers')['wd']
        assert saved['meta']['analysis_format'] == 5
        assert 'marker_tag_blocklist' not in saved['meta']
        assert all(['common', pytest.approx(.9)] in row['tags'] for row in saved['items'])
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('loaded a model'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *args: pytest.fail('fetched VTT'))
    assert app.main([*p.args, '--merge-only', '--force-merge']) == 0
    excluded = tmp_path / 'tags.yml'
    excluded.write_text('version: 1\nexcluded_scene_tags: [common]\n')
    assert app.main([*p.args, '--merge-only', '--scene-tag-exclusions', str(excluded)]) == 0
    assert all(m['primary_tag']['name'] != 'ai:common' for m in p.stash.scene['scene_markers'])
    # YAML policy changes preserve the raw evidence and can be reversed without inference.
    assert app.main([*p.args, '--merge-only']) == 0
    assert any(m['primary_tag']['name'] == 'ai:common' for m in p.stash.scene['scene_markers'])
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        assert state.unpack(state.connection.execute("SELECT data FROM " + state.table('results') + " WHERE kind='markers' AND model='wd'").fetchone()[0]) == saved


@pytest.mark.parametrize('retained', [None, 'common', [None]])
def test_unknown_blocklist_rejects_marker_replay_but_not_scene_only(
        pipeline, monkeypatch, retained):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'markers')['wd']
        records['meta']['analysis_format'] = 3
        if retained is None:
            records['meta'].pop('marker_tag_blocklist', None)
        else:
            records['meta']['marker_tag_blocklist'] = retained
        with state.connection:
            inject_records(state, '1', 'wd', {'markers': records})
    p.stash.writes.clear()
    assert app.main([*p.args, '--merge-only', '--force-merge']) == 1
    assert not p.stash.writes
    assert app.main([*p.args, '--scene-tags-only', '--merge-only']) == 1
    assert not p.stash.writes


def test_each_history_segment_is_checked_and_imported_ranges_keep_their_semantics():
    old = segment('wd', [[('common', .9)]] * 2)
    old['meta']['analysis_format'] = 3
    old['meta']['marker_tag_blocklist'] = ['missing']
    current = segment('wd', [[('common', .9)]] * 2, identity='new')
    with pytest.raises(ValueError, match='Invalid active inference record'):
        thresholds.validate_marker_cache(app, {'wd': [old, current]})
    imported = dict(mode='spans', meta={'origin': 'legacy'}, items=[])
    with pytest.raises(ValueError, match='Active results require current inference'):
        thresholds.validate_marker_cache(app, {'wd': imported})


def test_retired_cache_preflight_reason_is_stable():
    assert rethreshold_targets.threshold_failure_reason(ValueError(
        'Obsolete active inference evidence; run --db-update and inference again.')) == 'obsolete inference evidence'
