"""Actual reuse explanations stay observational and never demand fresh cache."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from stash_ai.history.store import HistoryStore
from stash_ai.storage.store import scope_for
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


def decisions(path):
    with HistoryStore(Path(str(path) + '.runs.sqlite3'), readonly=True) as store:
        return store.decisions(scope_for(app.STASH_URL))


def settings(monkeypatch, tmp_path, **changes):
    from stash_ai.backends.vocabulary import metadata_files, weight_files
    original = app.services.load_model_settings
    base = original(app.MODELS_CONFIG_FILE, ['wd'], None, False)[0]
    directory = tmp_path / 'model'
    directory.mkdir()
    config = replace(base, model_dir=directory, **changes)
    for name in (*metadata_files(config.source), *weight_files(config)):
        (directory / name).write_bytes(b'first asset')
    selected = [config]
    monkeypatch.setattr(app.services, 'load_model_settings', lambda *a: selected)
    return selected


def test_new_then_cached_result_and_offline_report(pipeline, monkeypatch, tmp_path):
    assert app.main(pipeline.args) == 0
    first = decisions(pipeline.path)[0]
    assert first['action'] == 'infer' and first['status'] == 'finished'
    assert {'missing_scene_evidence', 'missing_markers_evidence'} <= set(first['reasons'])
    loaded = len(pipeline.loaded)
    assert app.main(pipeline.args) == 0
    latest = decisions(pipeline.path)[0]
    assert latest['action'] == 'cache_reused'
    assert latest['reasons'] == ['input_fetch_skipped', 'matching_completed_identity']
    assert latest['details']['input_verification'] == 'skipped'
    assert len(pipeline.loaded) == loaded
    before = Path(str(pipeline.path) + '.runs.sqlite3').read_bytes()
    for service in ('create_stash_client', 'load_tagger', 'load_vocabulary', 'load_model_settings'):
        monkeypatch.setattr(app.services, service, lambda *a, **k: pytest.fail('reuse report reached Stash/models'))
    destination = tmp_path / 'reuse.json'
    assert app.main(['--reuse-report', str(destination), '--state-db', str(pipeline.path),
                     '--models', 'wd', '--scene-id', '1', '--run-limit', '2']) == 0
    report = json.loads(destination.read_text())
    assert report['models'][0]['actions'] == {'infer': 1, 'cache_reused': 1}
    assert len(report['decisions']) == 2
    assert Path(str(pipeline.path) + '.runs.sqlite3').read_bytes() == before


def test_detailed_proofs_are_optional_and_v1_history_migrates_without_inference(pipeline):
    assert app.main([*pipeline.args, '--no-run-history']) == 0
    loaded = len(pipeline.loaded)
    path = Path(str(pipeline.path) + '.runs.sqlite3')
    with HistoryStore(path):
        pass
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE reuse_proofs')
        db.execute('DROP TABLE reuse_decisions')
        db.execute('PRAGMA user_version=1')
    before = path.read_bytes()
    with HistoryStore(path, readonly=True) as store:
        assert store.decisions(scope_for(app.STASH_URL)) == []
    assert path.read_bytes() == before
    assert app.main(pipeline.args) == 0
    assert len(pipeline.loaded) == loaded
    assert decisions(pipeline.path)[0]['action'] == 'cache_reused'
    with HistoryStore(path, readonly=True) as store:
        assert store.connection.execute('SELECT count(*) FROM reuse_proofs').fetchone()[0] == 0
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    assert len(pipeline.loaded) == loaded
    with HistoryStore(path, readonly=True) as store:
        assert store.connection.execute('SELECT count(*) FROM reuse_proofs').fetchone()[0] == 2


def test_changed_vtt_cues_identify_the_exact_difference(pipeline):
    from stash_ai.domain import Cue
    assert app.main(pipeline.args) == 0
    pipeline.cues = [Cue(cue.start, cue.end - .25, cue.image, cue.rect) for cue in pipeline.cues]
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    row = decisions(pipeline.path)[0]
    assert row['action'] == 'infer' and 'vtt_cues_changed' in row['reasons']
    assert 'historical_identity_unavailable' not in row['reasons']
    assert row['details']['changes']['scene']['cues']['before'] != row['details']['changes']['scene']['cues']['after']


def test_changed_sprite_bytes_and_no_plaintext_media_urls(pipeline, monkeypatch):
    secret = 'SECRET_MEDIA_API_KEY'
    original = pipeline.stash.get_media
    monkeypatch.setattr(app, 'STASH_API_KEY', secret)
    pipeline.stash.scene['paths']['vtt'] += '?apikey=' + secret
    assert app.main(pipeline.args) == 0
    monkeypatch.setattr(pipeline.stash, 'get_media', lambda url: original(url) + b'changed sprite')
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    row = decisions(pipeline.path)[0]
    assert 'sprite_media_changed' in row['reasons']
    with HistoryStore(Path(str(pipeline.path) + '.runs.sqlite3'), readonly=True) as store:
        payload = ''.join(str(row) for row in store.connection.execute('SELECT proof FROM reuse_proofs'))
    assert secret not in payload + json.dumps(row)
    assert '/sprite' not in payload and '/vtt' not in payload


def test_changed_vtt_bytes_even_with_identical_cues_are_reported(pipeline, monkeypatch):
    from stash_ai.inference import sprites
    from stash_ai.stash_api.transport import absolute_stash_url
    original_fetch = sprites.fetch_cues
    original_media = pipeline.stash.get_media
    def fetch(runtime, media, paths):
        media.get_media(absolute_stash_url(paths['vtt'], runtime.STASH_URL))
        return original_fetch(runtime, media, paths)
    monkeypatch.setattr(sprites, 'fetch_cues', fetch)
    assert app.main(pipeline.args) == 0
    monkeypatch.setattr(pipeline.stash, 'get_media', lambda url: original_media(url) + (b'new vtt' if '/vtt' in url else b''))
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    row = decisions(pipeline.path)[0]
    assert 'vtt_media_changed' in row['reasons']
    assert 'vtt_cues_changed' not in row['reasons'] and 'sprite_media_changed' not in row['reasons']


def test_changed_assets_are_detected_by_content_not_timestamps(pipeline, monkeypatch, tmp_path):
    from stash_ai.backends.vocabulary import weight_files
    selected = settings(monkeypatch, tmp_path)
    assert app.main(pipeline.args) == 0
    asset = selected[0].model_dir / weight_files(selected[0])[0]
    asset.write_bytes(b'other asset')
    assert app.main(pipeline.args) == 0
    row = decisions(pipeline.path)[0]
    assert 'model_assets_changed' in row['reasons']
    assert 'sprite_media_changed' not in row['reasons']
    assert row['details']['changes']['scene']['assets']


def test_more_demanding_retention_explains_required_inference(pipeline, monkeypatch, tmp_path):
    selected = settings(monkeypatch, tmp_path)
    assert app.main(pipeline.args) == 0
    selected[0] = replace(selected[0], scene_cache_floor=.01)
    assert app.main(pipeline.args) == 0
    row = decisions(pipeline.path)[0]
    assert 'retention_changed' in row['reasons']
    assert 'insufficient_retention' in row['reasons']


def test_threshold_only_change_uses_rethreshold_without_loading_model(pipeline, monkeypatch, tmp_path):
    selected = settings(monkeypatch, tmp_path, scene_cache_floor=.01, marker_cache_floor=.01)
    assert app.main(pipeline.args) == 0
    loaded = len(pipeline.loaded)
    selected[0] = replace(selected[0], scene_threshold=.45, marker_threshold=.75)
    assert app.main(pipeline.args) == 0
    row = decisions(pipeline.path)[0]
    assert row['action'] == 'rethreshold'
    assert row['reasons'] == ['input_fetch_skipped', 'retained_evidence_covers_thresholds']
    assert len(pipeline.loaded) == loaded


def test_insufficient_retention_explains_why_threshold_replay_failed(pipeline, monkeypatch, tmp_path):
    selected = settings(monkeypatch, tmp_path)
    assert app.main(pipeline.args) == 0
    selected[0] = replace(selected[0], scene_threshold=.05, marker_threshold=.1)
    assert app.main(pipeline.args) == 0
    row = decisions(pipeline.path)[0]
    assert row['action'] == 'infer'
    assert 'insufficient_retention' in row['reasons']
    assert 'rethreshold_policy_incompatible' in row['reasons']
    assert row['details']['outputs']['scene']['retained_floor'] > row['details']['outputs']['scene']['required_retention_floor']


def test_legacy_input_change_is_not_given_a_fabricated_specific_cause(pipeline):
    from stash_ai.domain import Cue
    assert app.main([*pipeline.args, '--no-run-history']) == 0
    pipeline.cues = [Cue(cue.start, cue.end - .25, cue.image, cue.rect) for cue in pipeline.cues]
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    row = decisions(pipeline.path)[0]
    assert 'historical_identity_unavailable' in row['reasons'] and 'input_changed' in row['reasons']
    assert 'sprite_media_changed' not in row['reasons'] and 'vtt_cues_changed' not in row['reasons']


def test_explicit_restart_is_distinguished_from_stale_cache(pipeline):
    assert app.main(pipeline.args) == 0
    assert app.main([*pipeline.args, '--restart']) == 0
    row = decisions(pipeline.path)[0]
    assert 'explicit_restart' in row['reasons']
    assert 'vtt_cues_changed' not in row['reasons'] and 'model_assets_changed' not in row['reasons']


def test_checkpoint_resume_and_changed_checkpoint_input(pipeline):
    from stash_ai.domain import Cue
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    assert decisions(pipeline.path)[0]['status'] == 'interrupted'
    pipeline.fail.clear()
    assert app.main(pipeline.args) == 0
    resumed = decisions(pipeline.path)[0]
    assert resumed['action'] == 'checkpoint_resumed'
    assert resumed['details']['checkpoint']['resumed_frames'] == 4
    assert 'checkpoint_compatible' in resumed['reasons']
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main([*pipeline.args, '--restart'])
    pipeline.cues = [Cue(cue.start, cue.end - .25, cue.image, cue.rect) for cue in pipeline.cues]
    pipeline.fail.clear()
    assert app.main(pipeline.args) == 0
    fresh = decisions(pipeline.path)[0]
    assert fresh['action'] == 'infer' and 'checkpoint_signature_changed' in fresh['reasons']
    assert fresh['details']['checkpoint']['resumed_frames'] == 0


def test_missing_inputs_are_blocked_without_deleting_cache(pipeline):
    assert app.main(pipeline.args) == 0
    pipeline.stash.scene['paths']['vtt'] = None
    assert app.main(pipeline.args) == 1
    row = decisions(pipeline.path)[0]
    assert row['action'] == row['status'] == 'blocked'
    assert row['reasons'] == ['missing_media']


def test_diagnostic_database_failure_does_not_change_reuse(pipeline, monkeypatch):
    assert app.main(pipeline.args) == 0
    loaded = len(pipeline.loaded)
    monkeypatch.setattr(HistoryStore, 'queue', lambda *a: (_ for _ in ()).throw(sqlite3.OperationalError('disk full')))
    assert app.main(pipeline.args) == 0
    assert len(pipeline.loaded) == loaded


def test_opt_out_skips_diagnostic_proofs_without_skipping_input_validation(pipeline, monkeypatch):
    from stash_ai.inference import reuse_diagnostics
    from stash_ai.domain import Cue
    monkeypatch.setattr(reuse_diagnostics, 'proof', lambda *a: pytest.fail('disabled diagnostics built a proof'))
    arguments = [*pipeline.args, '--no-run-history', '--force-fetch-vtt']
    assert app.main(arguments) == 0
    loaded = len(pipeline.loaded)
    assert app.main(arguments) == 0
    assert len(pipeline.loaded) == loaded
    pipeline.cues = [Cue(cue.start, cue.end - .25, cue.image, cue.rect) for cue in pipeline.cues]
    assert app.main(arguments) == 0
    assert len(pipeline.loaded) == loaded + 1


def test_verbose_opt_out_keeps_diagnostic_explanations(pipeline, monkeypatch):
    from stash_ai.inference import reuse_diagnostics
    original, calls = reuse_diagnostics.proof, []
    def observe(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(reuse_diagnostics, 'proof', observe)
    assert app.main([*pipeline.args, '--no-run-history', '--verbose']) == 0
    assert calls


@pytest.mark.parametrize('empty', [False, True])
def test_publication_exclusions_and_empty_results_do_not_require_reinference(pipeline, monkeypatch, tmp_path, empty):
    if empty:
        pipeline.scores['wd'].fill(0)
    assert app.main(pipeline.args) == 0
    policy = tmp_path / 'exclude.yml'
    policy.write_text('version: 1\nexcluded_scene_tags: [common]\n')
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('diagnostics required unnecessary inference'))
    assert app.main([*pipeline.args, '--scene-tag-exclusions', str(policy)]) == 0
    row = decisions(pipeline.path)[0]
    assert row['action'] == 'cache_reused'
    assert row['reasons'] == ['input_fetch_skipped', 'matching_completed_identity']


def test_changed_vocabulary_is_distinguished_from_changed_weights(pipeline):
    assert app.main(pipeline.args) == 0
    pipeline.vocab.labels.reverse()
    assert app.main(pipeline.args) == 0
    row = decisions(pipeline.path)[0]
    assert 'vocabulary_changed' in row['reasons'] and 'model_assets_changed' not in row['reasons']


def test_filters_select_latest_matching_run_before_applying_limit(tmp_path):
    scope = scope_for(app.STASH_URL)
    path = tmp_path / 'history.sqlite3'
    with HistoryStore(path) as store:
        for identifier, command, scene, model in [('1', 'inference', 'one', 'wd'),
                ('2', 'inference', 'two', 'dtq'), ('3', 'db_compact', None, None)]:
            store.start(identifier, scope, command, '2026-01-0' + identifier, {})
            if scene:
                store.record_decision(identifier, scope, scene, model, 'infer', 'finished', [], {})
            store.finish(identifier, '2026-01-04', 'finished', 1, 1,
                         models={}, counts={}, outcomes={}, metrics=[], issues={})
    with HistoryStore(path, readonly=True) as store:
        assert [row['run_id'] for row in store.decisions(scope)] == ['2']
        filtered = store.decisions(scope, scenes=['one'], models=['wd'], command='inference', status='finished')
        assert [row['run_id'] for row in filtered] == ['1']
        assert store.decisions(scope, status='interrupted') == []


def test_existing_reuse_report_is_never_overwritten(tmp_path):
    destination = tmp_path / 'report.json'
    destination.write_text('preserve me')
    with pytest.raises(SystemExit):
        app.main(['--reuse-report', str(destination)])
    assert destination.read_text() == 'preserve me'


def test_decision_batching_keeps_pending_memory_bounded_and_reads_own_proofs(tmp_path):
    with HistoryStore(tmp_path / 'history.sqlite3') as store:
        scope = scope_for(app.STASH_URL)
        store.start('run', scope, 'inference', '2026-01-01', {})
        for index in range(1000):
            store.record_decision('run', scope, str(index), 'wd', 'cache_reused', 'finished', ['matching_completed_identity'], {})
            assert len(store.pending) < 64
        store.put_proof(scope, 'one', 'wd', 'scene', 'first', {'value': 1})
        assert store.get_proof(scope, 'one', 'wd', 'scene', 'first') == {'value': 1}
        store.put_proof(scope, 'one', 'wd', 'scene', 'second', {'value': 2})
        assert store.get_proof(scope, 'one', 'wd', 'scene', 'first') is None
        assert store.get_proof(scope, 'one', 'wd', 'scene', 'second') == {'value': 2}
    with HistoryStore(tmp_path / 'history.sqlite3', readonly=True) as store:
        assert len(store.decisions(scope)) == 1000


@pytest.mark.parametrize('extra', [['--record-run'], ['--restart'], ['--merge-only'], ['--dry-run'], ['--models', 'unknown']])
def test_reuse_report_rejects_execution_modifiers(tmp_path, extra):
    with pytest.raises(SystemExit):
        app.main(['--reuse-report', str(tmp_path / 'new.json'), *extra])
