"""Cleanup can revoke published proofs from caches predating blocklist metadata."""

from copy import deepcopy
from dataclasses import replace

import pytest

from stash_ai import inference
from stash_ai import merge_cleanup
from stash_ai import cli as app, merge, thresholds
from stash_ai.models import SOURCES
from tests.helpers.evidence import segment

pytestmark = pytest.mark.usefixtures('reset_temporal')


@pytest.fixture(params=[False, True], ids=['known-retention', 'unknown-retention'])
def old_consensus(pipeline, monkeypatch, tmp_path, request):
    p = pipeline
    settings = app.load_model_settings
    monkeypatch.setattr(app, 'load_model_settings', lambda *args: [
        replace(config, scene_threshold=.4, marker_threshold=.8)
        for config in settings(*args)])
    for model in ('wd', 'dtq'):
        p.scores[model][:] = 0
        p.scores[model][:, 0] = .7
    p.args = ['--models', 'wd', 'dtq', *p.args[2:]]

    # Emulate publication before #82, optionally also before retention metadata.
    # Both cached frames and published proof signatures lack these fields.
    # Restore strict checks after capturing the historical publication.
    capture = merge.capture_outputs
    with monkeypatch.context() as previous:
        def capture_old(*args, **kwargs):
            outputs = capture(*args, **kwargs)
            outputs['markers']['meta'].pop('marker_tag_blocklist')
            if request.param:
                outputs['markers']['meta'].pop('marker_score_floor')
            return outputs
        previous.setattr(merge, 'capture_outputs', capture_old)
        previous.setattr(thresholds, 'validate_marker_cache', lambda *args, **kwargs: None)
        assert app.main(p.args) == 0
    assert len(p.stash.scene['scene_markers']) == 1
    assert p.stash.scene['scene_markers'][0]['primary_tag']['name'] == 'ai:common'
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        published, _ = state.snapshot('1', 'markers')
        assert any({span['source'], *span.get('support_sources', [])} == {'wd', 'dtq'}
                   for span in published['contributions'])
        assert all('marker_tag_blocklist' not in records[0]['meta']
                   for records in state.records('1', 'markers').values())
        assert all(('marker_score_floor' not in records[0]['meta']) == request.param
                   for records in state.records('1', 'markers').values())
    core = tmp_path/'exclude.yml'
    core.write_text('version: 1\nexcluded_core_tags: []\nby_model:\n  wd: [common]\n')
    p.cleanup_args = [*p.args, '--cleanup-markers-only', '--cleanup-strategy', 'scenes',
                      '--marker-core-exclusions', str(core)]
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('loaded a model'))
    monkeypatch.setattr(inference, 'fetch_cues', lambda *args: pytest.fail('fetched VTT'))
    monkeypatch.setattr(p.stash, 'get_media', lambda *args: pytest.fail('fetched sprites'))
    p.stash.writes.clear()
    return p


@pytest.mark.parametrize('dry_run', [False, True])
def test_cleanup_revokes_an_old_consensus_vote_without_inference(old_consensus, dry_run):
    p = old_consensus
    before = deepcopy(p.stash.scene)
    p.stash.dry_run = dry_run
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        evidence = state.records('1', 'markers')
    assert app.main([*p.cleanup_args, *(['--dry-run'] if dry_run else [])]) == 0
    if dry_run:
        assert p.stash.scene == before and not p.stash.writes
    else:
        assert not p.stash.scene['scene_markers']
        assert any(write[0] == 'delete' for write in p.stash.writes)
        assert all(write[0] != 'create' for write in p.stash.writes)
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        assert state.records('1', 'markers') == evidence


def test_old_cache_still_blocks_normal_merge_and_rethreshold(old_consensus):
    p = old_consensus
    assert app.main(p.cleanup_args) == 0
    p.stash.writes.clear()
    before = deepcopy(p.stash.scene)
    assert app.main([*p.args, '--merge-only', '--force-merge']) == 1
    assert app.main([*p.args, '--merge-only', '--rethreshold-markers']) == 1
    assert p.stash.scene == before and not p.stash.writes


def test_cleanup_does_not_adopt_a_new_unpublished_vote(old_consensus):
    p = old_consensus
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
        fresh = segment('oppai', [[('common', .7)]] * 10)
        fresh['meta'].pop('marker_tag_blocklist')
        state.put('1', 'oppai', {'markers': fresh})
    assert app.main(p.cleanup_args) == 0
    assert not p.stash.scene['scene_markers']


def test_cleanup_rejects_changes_to_published_frame_proofs(old_consensus):
    p = old_consensus
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'markers')['dtq']
        records[0]['items'][0]['tags'][0][1] = .95
        with state.connection:
            state.replace_records('1', 'dtq', 'markers', records)
    before = deepcopy(p.stash.scene)
    assert app.main(p.cleanup_args) == 1
    assert p.stash.scene == before and not p.stash.writes


@pytest.mark.parametrize('retained', [None, 'common', [None], ['common']])
def test_cleanup_exception_does_not_accept_invalid_or_known_incomplete_metadata(retained):
    frames = segment('wd', [[('common', .9)]] * 2)
    frames['meta']['marker_tag_blocklist'] = retained
    with pytest.raises(ValueError, match='marker blocklist metadata|unblocked marker tags'):
        thresholds.validate_marker_cache(
            app, {'wd': [frames]}, allow_unknown_blocklist=True)


@pytest.mark.parametrize('floor', [None, float('nan'), .75])
def test_cleanup_retention_exception_still_checks_explicit_floors(floor):
    frames = segment('wd', [[('common', .9)]] * 2)
    frames['meta']['marker_score_floor'] = floor
    with pytest.raises(ValueError, match='marker cache floor'):
        thresholds.validate_marker_cache(app, {'wd': [frames]}, allow_unknown_retention=True)


def test_blocklist_exception_does_not_relax_historical_retention():
    frames = segment('wd', [[('common', .9)]] * 2)
    frames['meta'].pop('marker_tag_blocklist')
    frames['meta'].pop('marker_score_floor')
    with pytest.raises(ValueError, match='marker cache floor'):
        thresholds.validate_marker_cache(app, {'wd': [frames]}, allow_unknown_blocklist=True)


def test_repeated_cleanup_preserves_nonempty_historical_bans(tmp_path):
    policies = {source: frozenset() for source in SOURCES}
    records = {
        'wd': [segment('wd', [[('core', .9)]] * 4)],
        'dtq': [segment('dtq', [[('detail', .9), ('revoked', .9), ('remaining', .9)]] * 4)],
    }
    # WD discarded detail at capture time; DTQ still has it in an older cache.
    # The published proof applied the old blocklist to both sources.
    records['wd'][0]['meta']['marker_tag_blocklist'] = ['detail']
    frozen_tags = {source: frozenset({'detail'}) for source in SOURCES}
    spans = merge.marker_contributions(app, records, policies, tag_policies=frozen_tags,
                                       legacy_blocklist=frozenset({'detail'}))
    for span in spans:
        span['build_options']['blocklist'] = ['detail']
        span['build_options']['tag_exclusions'] = {source: [] for source in SOURCES}
    assert merge.union_intervals(spans)[0]['extra_tags'] == ['remaining', 'revoked']

    with merge.StateStore(tmp_path / 'state', 'fixture') as state:
        for source, segments in records.items():
            state.put('1', source, {'markers': segments[0]})
        original = state.records('1', 'markers')
        tags = {**policies, 'dtq': frozenset({'revoked'})}
        first = merge_cleanup.cleanup_support(app, state, '1', spans, {'dtq'}, policies, tags)
        assert merge.union_intervals(first)[0]['extra_tags'] == ['remaining']
        assert all(span['build_options']['blocklist'] == ['detail'] for span in first)
        tags['dtq'] = frozenset({'remaining', 'revoked'})
        second = merge_cleanup.cleanup_support(app, state, '1', first, {'dtq'}, policies, tags)
        assert merge.union_intervals(second)[0]['extra_tags'] == []
        assert state.records('1', 'markers') == original
        with pytest.raises(ValueError, match='unblocked marker tags'):
            merge.marker_contributions(app, original, policies)
