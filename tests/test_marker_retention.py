"""Marker-policy replay must never treat unretained detections as absence."""
from copy import deepcopy
from dataclasses import replace
import sqlite3

import pytest

from stash_ai.inference import sprites
from tests.helpers.runtime import runtime as app
import stash_ai.evidence.identity as thresholds_identity
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.source_markers as merge_source_markers
import stash_ai.storage.store as merge_store
from tests.test_thresholds import configure, raw
from tests.helpers.temporal import install_threshold_policy

pytestmark = pytest.mark.usefixtures('reset_temporal')


def marker_ranges(stash):
    return sorted((m['primary_tag']['name'], m['seconds'], m.get('end_seconds'))
                  for m in stash.scene['scene_markers'])


def stored_outputs(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute(f'SELECT * FROM {table} ORDER BY 1,2,3').fetchall()
                for table in ('results', 'snapshots')}


def offline(monkeypatch, stash):
    def unexpected(*args, **kwargs):
        pytest.fail('accessed inference inputs')
    for name in ('load_tagger', 'load_vocabulary'):
        monkeypatch.setattr(app.services, name, unexpected)
    monkeypatch.setattr(sprites, 'fetch_cues', unexpected)
    monkeypatch.setattr(stash, 'get_media', unexpected)


@pytest.mark.parametrize('old_cache', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
def test_wider_support_policy_checks_historical_retention_before_publication(
        pipeline, monkeypatch, capsys, old_cache, dry_run):
    p = pipeline
    p.scores['wd'][:] = 0
    p.scores['wd'][:8, 0] = .6
    p.scores['wd'][:2, 0] = p.scores['wd'][6:8, 0] = .8
    args = [*p.args, '--markers-only']
    with monkeypatch.context() as previous:
        install_threshold_policy(previous, app, p.path.parent)
        if old_cache:
            settings = app.services.load_model_settings
            previous.setattr(app.services, 'load_model_settings', lambda *a: [
                replace(config, scene_cache_floor=None, marker_cache_floor=None)
                for config in settings(*a)])
        assert app.main([*args]) == 0
    if old_cache:
        from stash_ai.storage.codec import encode
        with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL)) as state:
            records = state.records('1', 'markers')['wd']
            records['meta']['analysis_format'] = 3
            records['meta']['marker_tag_blocklist'] = []
            records['meta'].pop('marker_score_floor', None)
            with state.connection:
                state.connection.execute("UPDATE results SET data=? WHERE kind='markers'", (encode(records),))
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        meta = state.unpack(state.connection.execute("SELECT data FROM " + state.table('results') + " WHERE kind='markers' AND model='wd'").fetchone()[0])['meta']
        assert ('marker_score_floor' not in meta) == old_cache
    before = stored_outputs(p.path), deepcopy(p.stash.scene)
    database_bytes = p.path.read_bytes()
    p.stash.writes.clear()
    p.stash.dry_run = dry_run
    capsys.readouterr()
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        status = app.main([*args, '--merge-only', *(['--dry-run'] if dry_run else [])])
    assert status == int(old_cache)
    if old_cache or dry_run:
        assert (stored_outputs(p.path), p.stash.scene) == before
        assert not p.stash.writes
    if dry_run:
        assert p.path.read_bytes() == database_bytes
    if old_cache:
        output = capsys.readouterr().out
        assert 'Obsolete active inference' in output and '--db-update' in output
    elif not dry_run:
        assert marker_ranges(p.stash) == [('ai:common', 0, 8)]
    monkeypatch.setattr(app, 'DRY_RUN', False)
    p.stash.dry_run = False
    if old_cache:
        from stash_ai.db import update_database
        with sqlite3.connect(p.path) as connection:
            connection.execute('PRAGMA user_version=14')
        update_database(p.path)
    assert app.main([*args, '--restart']) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 8)]


def test_old_high_only_cache_remains_usable_with_matching_support_policy(pipeline, monkeypatch):
    p = pipeline
    configure(monkeypatch)
    install_threshold_policy(monkeypatch, app, p.path.parent)
    args = [*p.args, '--markers-only']
    assert app.main(args) == 0
    before = stored_outputs(p.path), deepcopy(p.stash.scene)
    p.stash.writes.clear()
    offline(monkeypatch, p.stash)
    assert app.main([*args, '--merge-only', '--force-merge']) == 0
    assert (stored_outputs(p.path), p.stash.scene) == before
    assert not p.stash.writes


@pytest.mark.parametrize('override', [.8, .9])
def test_unknown_historical_floor_requires_inference_even_at_high_threshold(override):
    from tests.helpers.evidence import segment, policies
    frames = segment('wd', [[('core', .95)]] * 2)
    frames['meta']['analysis_format'] = 3
    frames['meta']['marker_tag_blocklist'] = []
    frames['meta'].pop('marker_score_floor', None)
    frames['meta']['marker_threshold'] = .7
    frames[thresholds_identity.PUBLICATION_THRESHOLD] = override
    with pytest.raises(ValueError, match='Obsolete active inference'):
        merge_source_markers.marker_contributions(app, {'wd': frames}, policies())


def test_historical_incomplete_segment_cannot_hide_behind_complete_history():
    from tests.helpers.evidence import segment, policies
    old = segment('wd', [[], []])
    old['meta']['analysis_format'] = 3
    old['meta']['marker_tag_blocklist'] = []
    old['meta'].pop('marker_score_floor', None)
    current = segment('wd', [[('core', .9)]] * 2)
    with pytest.raises(ValueError, match='Invalid active inference record'):
        merge_source_markers.marker_contributions(app, {'wd': [old, current]}, policies())


@pytest.mark.parametrize('markers_only', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
def test_lower_ratio_rejects_incomplete_cache_before_publication(
        pipeline, monkeypatch, tmp_path, capsys, markers_only, dry_run):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = 0
    p.scores['wd'][:2, 0] = .8
    p.scores['wd'][2:6, 0] = .55
    p.scores['wd'][6:8, 0] = .8
    args = [*p.args, *(['--markers-only'] if markers_only else [])]
    assert app.main(args) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 2), ('ai:common', 6, 8)]
    policy = tmp_path/'relaxed.yml'
    policy.write_text('version: 1\nmarker_continue_ratio: 0.7\n')
    before = stored_outputs(p.path), deepcopy(p.stash.scene)
    database_bytes = p.path.read_bytes()
    p.stash.writes.clear()
    p.stash.dry_run = dry_run
    capsys.readouterr()
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        assert app.main([*args, '--merge-only', '--temporal-policy', str(policy),
                         *(['--dry-run'] if dry_run else [])]) == 1
    assert (stored_outputs(p.path), p.stash.scene) == before
    if dry_run:
        assert p.path.read_bytes() == database_bytes
    assert not p.stash.writes
    message = capsys.readouterr().out
    assert 'wd' in message and 'cache floor' in message and 'run inference' in message
    # A normal run refreshes the incomplete evidence and connects the range.
    monkeypatch.setattr(app, 'DRY_RUN', False)
    p.stash.dry_run = False
    assert app.main([*args, '--temporal-policy', str(policy)]) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 8)]


@pytest.mark.parametrize('marker,ratio,expected', [
    (.7, 1.0, [('ai:common', 0, 2), ('ai:common', 6, 8)]),
    (.9, .7, [('ai:common', 0, 8)]),
])
def test_sufficient_cache_matches_fresh_inference_with_effective_threshold(
        pipeline, monkeypatch, tmp_path, marker, ratio, expected):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = 0
    p.scores['wd'][:2, 0] = .95
    p.scores['wd'][2:6, 0] = .64
    p.scores['wd'][6:8, 0] = .95
    args = [*p.args, '--markers-only']
    assert app.main(args) == 0
    configure(monkeypatch, marker=marker)
    assert app.main([*args, '--merge-only', '--rethreshold-markers']) == 0
    previous = marker_ranges(p.stash)
    observations = raw(p.path)
    policy = tmp_path/'changed.yml'
    policy.write_text(f'version: 1\nmarker_continue_ratio: {ratio}\n')
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        assert app.main([*args, '--merge-only', '--temporal-policy', str(policy)]) == 0
        assert marker_ranges(p.stash) == expected
        assert raw(p.path) == observations
        assert app.main([*args, '--merge-only']) == 0
        assert marker_ranges(p.stash) == previous
    p.stash.scene['scene_markers'] = []
    assert app.main([*args, '--state-db', str(tmp_path/'fresh.sqlite3'), '--restart',
                     '--temporal-policy', str(policy)]) == 0
    assert marker_ranges(p.stash) == expected


def test_scene_tags_only_does_not_require_marker_cache(pipeline, monkeypatch, tmp_path):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    policy = tmp_path/'relaxed.yml'
    policy.write_text('version: 1\nmarker_continue_ratio: 0.7\n')
    before = deepcopy(p.stash.scene)
    offline(monkeypatch, p.stash)
    assert app.main([*p.args, '--merge-only', '--scene-tags-only', '--temporal-policy', str(policy)]) == 0
    assert p.stash.scene == before


def test_noninference_cannot_claim_current_retention_contract():
    from tests.helpers.evidence import segment, policies
    imported = segment('wd', [[('core', .9)]] * 2)
    imported['meta'].update(origin='legacy')
    imported['meta'].pop('marker_score_floor', None)
    with pytest.raises(ValueError, match='Active results require current inference'):
        merge_source_markers.marker_contributions(app, {'wd': imported}, policies())


@pytest.mark.parametrize('source', ['wd', 'dtq'])
def test_incomplete_history_cannot_contribute_consensus_votes(source):
    from tests.helpers.evidence import segment, policies
    records = {key: segment(key, [[('core', .7)]] * 3) for key in ('wd', 'dtq')}
    for segments in records.values():
        segments['meta']['marker_score_floor'] = .6
    incomplete = deepcopy(records[source])
    incomplete['meta']['marker_score_floor'] = .75
    records[source] = incomplete
    with pytest.raises(ValueError, match=source + ': marker cache floor.*run inference'):
        merge_source_markers.marker_contributions(app, records, policies())
