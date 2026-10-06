"""Cleanup can revoke published proofs from caches predating blocklist metadata."""

from copy import deepcopy
from dataclasses import replace

import pytest

from stash_ai.inference import sprites
from stash_ai.cleanup import evidence as cleanup_evidence
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import thresholds as thresholds
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.source_markers as merge_source_markers
import stash_ai.inference.results as merge_results
import stash_ai.storage.store as merge_store
from stash_ai.models import SOURCES
from tests.helpers.evidence import segment, inference_record, inject_records
from stash_ai.storage.archive import records as archived_records

pytestmark = pytest.mark.usefixtures('reset_temporal')


@pytest.fixture(params=[False, True], ids=['known-retention', 'unknown-retention'])
def old_consensus(pipeline, monkeypatch, tmp_path, request):
    p = pipeline
    settings = app.services.load_model_settings
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *args: [
        replace(config, scene_threshold=.4, marker_threshold=.8)
        for config in settings(*args)])
    for model in ('wd', 'dtq'):
        p.scores[model][:] = 0
        p.scores[model][:, 0] = .7
    p.args = ['--models', 'wd', 'dtq', *p.args[2:]]

    # Emulate publication before #82, optionally also before retention metadata.
    # Both cached frames and published proof signatures lack these fields.
    # Restore strict checks after capturing the historical publication.
    capture = merge_results.capture_outputs
    with monkeypatch.context() as previous:
        def capture_old(*args, **kwargs):
            outputs = capture(*args, **kwargs)
            outputs['markers']['meta']['analysis_format'] = 3
            outputs['markers']['meta'].pop('marker_tag_blocklist', None)
            if request.param:
                outputs['markers']['meta'].pop('marker_score_floor')
            return outputs
        previous.setattr(merge_results, 'capture_outputs', capture_old)
        from stash_ai.evidence import formats
        previous.setattr(formats, 'validate_current', lambda *a, **k: None)
        previous.setattr(thresholds, 'validate_marker_cache', lambda *args, **kwargs: None)
        assert app.main(p.args) == 0
    assert len(p.stash.scene['scene_markers']) == 1
    assert p.stash.scene['scene_markers'][0]['primary_tag']['name'] == 'ai:common'
    from stash_ai.db.update import update_database
    import sqlite3
    with sqlite3.connect(p.path) as connection:
        connection.execute('PRAGMA user_version=18')
    update_database(p.path)
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        published, _ = state.snapshot('1', 'markers')
        assert any({span['source'], *span.get('support_sources', [])} == {'wd', 'dtq'}
                   for span in published['contributions'])
        assert all('marker_tag_blocklist' not in records[0]['meta']
                   for records in archived_records(state, '1', 'markers').values())
        assert all(('marker_score_floor' not in records[0]['meta']) == request.param
                   for records in archived_records(state, '1', 'markers').values())
    core = tmp_path/'exclude.yml'
    core.write_text('version: 1\nexcluded_core_tags: []\nby_model:\n  wd: [common]\n')
    p.cleanup_args = [*p.args, '--cleanup-markers-only', '--cleanup-strategy', 'scenes',
                      '--marker-core-exclusions', str(core)]
    p.rebase_services = (app.services.load_tagger, sprites.fetch_cues, p.stash.get_media)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('loaded a model'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *args: pytest.fail('fetched VTT'))
    monkeypatch.setattr(p.stash, 'get_media', lambda *args: pytest.fail('fetched sprites'))
    p.stash.writes.clear()
    return p


@pytest.mark.parametrize('dry_run', [False, True])
def test_retired_consensus_requires_rebase_without_any_stash_mutations(old_consensus, dry_run):
    p = old_consensus
    before = deepcopy(p.stash.scene)
    p.stash.dry_run = dry_run
    assert app.main([*p.cleanup_args, *(['--dry-run'] if dry_run else [])]) == 1
    assert p.stash.scene == before and not p.stash.writes
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        assert all(item['requires_rebase'] for item in state.snapshot('1', 'markers')[0]['contributions'])


@pytest.mark.parametrize('floor', [None, float('nan'), .75])
def test_cleanup_has_the_same_current_retention_contract(floor):
    frames = segment('wd', [[('common', .9)]] * 2)
    frames['meta']['marker_score_floor'] = floor
    with pytest.raises(ValueError, match='marker cache floor|retention floors'):
        thresholds.validate_marker_cache(app, {'wd': frames})


def test_repeated_cleanup_preserves_frozen_temporal_bans(tmp_path):
    policies = {source: frozenset() for source in SOURCES}
    records = {
        'wd': segment('wd', [[('core', .9)]] * 4),
        'dtq': segment('dtq', [[('detail', .9), ('revoked', .9), ('remaining', .9)]] * 4),
    }
    frozen_tags = {source: frozenset({'detail'}) for source in SOURCES}
    spans = merge_source_markers.marker_contributions(app, records, policies, tag_policies=frozen_tags)
    assert merge_intervals.union_intervals(spans)[0]['extra_tags'] == ['remaining', 'revoked']

    with merge_store.StateStore(tmp_path / 'state', 'fixture') as state:
        for source, segments in records.items():
            from stash_ai.evidence.identity import marker_segment_signature
            with state.connection:
                state.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
                    (state.scope, '1', source, 'markers', marker_segment_signature(segments), state.pack(segments)))
        state.publish('1', {'markers': {'contributions': spans}})
        original = archived_records(state, '1', 'markers')
        tags = {**policies, 'dtq': frozenset({'revoked'})}
        first = cleanup_evidence.cleanup_support(app, state, '1', spans, {'dtq'}, policies, tags)
        assert merge_intervals.union_intervals(first)[0]['extra_tags'] == ['remaining']
        assert all('blocklist' not in span['build_options'] and all(
            'detail' in bans for bans in span['build_options']['tag_exclusions'].values()) for span in first)
        tags['dtq'] = frozenset({'remaining', 'revoked'})
        second = cleanup_evidence.cleanup_support(app, state, '1', first, {'dtq'}, policies, tags)
        assert merge_intervals.union_intervals(second)[0]['extra_tags'] == []
        assert archived_records(state, '1', 'markers') == original
        assert merge_source_markers.marker_contributions(app, {source: versions[0] for source, versions in original.items()}, policies)




def test_fresh_inference_rebases_retired_publication_and_preserves_marker_id(old_consensus, monkeypatch):
    p = old_consensus
    before_id = p.stash.scene['scene_markers'][0]['id']
    human = p.stash.tag('human')
    p.stash.scene['scene_markers'][0]['tags'].append(human)
    monkeypatch.setattr(app.services, 'load_tagger', p.rebase_services[0])
    monkeypatch.setattr(sprites, 'fetch_cues', p.rebase_services[1])
    monkeypatch.setattr(p.stash, 'get_media', p.rebase_services[2])
    assert app.main(p.args) == 0
    marker = p.stash.scene['scene_markers'][0]
    assert marker['id'] == before_id and human in marker['tags']
    with merge_store.StateStore(p.path, merge_store.scope_for(app.STASH_URL), True) as state:
        saved = state.snapshot('1', 'markers')[0]
        assert all(not item.get('requires_rebase') for item in saved['contributions'])
        assert all('algorithm' not in item['build_options'] for item in saved['contributions'])
    assert app.main(p.cleanup_args) == 0
    assert not p.stash.scene['scene_markers']
