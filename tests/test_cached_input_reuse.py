"""Completed scene reruns avoid media GETs without bypassing partial-run gates."""
from dataclasses import replace
import sqlite3

import pytest

from stash_ai.commands.arguments import parse_args
from stash_ai.history.lifecycle import configuration
from stash_ai.inference.inputs import RunInputs
from stash_ai.inference.sprites import fetch_cues as real_fetch_cues
from stash_ai.storage.codec import decode, encode
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.vtt import format_time
from tests.helpers.runtime import runtime as app
from tests.test_reuse_diagnostics import settings

pytestmark = pytest.mark.usefixtures('reset_progress')


def media_requests(pipeline, monkeypatch):
    """Use the real VTT downloader/parser; only model pixels are synthetic."""
    requests = []
    monkeypatch.setattr('stash_ai.inference.sprites.fetch_cues', real_fetch_cues)

    def get_media(url):
        requests.append(url)
        if '/vtt' in url:
            cues = '\n\n'.join(f'{format_time(c.start)} --> {format_time(c.end)}\n/sprite#xywh=0,0,1,1'
                               for c in pipeline.cues)
            return ('WEBVTT\n\n' + cues + '\n').encode()
        return b'fixture sprite bytes'

    monkeypatch.setattr(pipeline.stash, 'get_media', get_media)
    return requests


def durable_rows(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute('SELECT * FROM ' + table + ' ORDER BY 1,2,3').fetchall()
                for table in ('results', 'snapshots', 'inference_archive', 'inference_identities',
                              'result_index', 'checkpoints', 'checkpoint_frames', 'checkpoint_analysis',
                              'refresh_requests')}


def model_args(pipeline, *models):
    return ['--models', *models, *pipeline.args[2:]]


@pytest.mark.parametrize('initialize', [False, True])
def test_interrupted_oppai_resume_skips_completed_joytag_media_pass(pipeline, monkeypatch, initialize):
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'joytag', 'oppai')
    pipeline.fail['oppai'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main([*args, '--db-init'] if initialize else args)
    with sqlite3.connect(pipeline.path) as db:
        assert db.execute("SELECT processed FROM checkpoints WHERE model='oppai'").fetchone()[0] == 4
        assert db.execute("SELECT count(*) FROM results WHERE model='joytag'").fetchone()[0] == 2
    before = durable_rows(pipeline.path)
    loaded = len(pipeline.loaded)
    pipeline.fail.clear()
    requests.clear()
    inspected = []
    original = RunInputs.inspect

    def inspect(self, runtime, paths, identity):
        inspected.append(identity['source'])
        return original(self, runtime, paths, identity)

    monkeypatch.setattr(RunInputs, 'inspect', inspect)
    assert app.main(args) == 0
    assert inspected == ['oppai']
    assert len(requests) == 2
    assert [m.settings.source.key for m in pipeline.loaded[loaded:]] == ['oppai']
    assert pipeline.loaded[-1].seen == list(range(4, 10))
    after = durable_rows(pipeline.path)
    assert [r for r in after['results'] if r[2] == 'joytag'] == [r for r in before['results'] if r[2] == 'joytag']
    assert not after['checkpoints'] and not after['refresh_requests']


@pytest.mark.parametrize('models', [('wd', 'dtq'), ('dtq', 'wd'), ('all',)])
@pytest.mark.parametrize('missing', ['result', 'refresh'])
def test_unchanged_incomplete_model_does_not_fetch_for_completed_peers(pipeline, monkeypatch, models, missing):
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, *models)
    assert app.main(args) == 0
    if missing == 'result':
        with sqlite3.connect(pipeline.path) as db:
            for table in ('results', 'result_index'):
                db.execute('DELETE FROM ' + table + " WHERE model='dtq'")
    else:
        from stash_ai.storage.progress import request_refresh
        with StateStore(pipeline.path, scope_for(app.STASH_URL)) as store:
            request_refresh(store, ['1'], ['dtq'], ['scene', 'markers'])
    before = durable_rows(pipeline.path)['results']
    requests.clear()
    inspected = []
    original = RunInputs.inspect

    def inspect(self, runtime, paths, identity):
        inspected.append(identity['source'])
        return original(self, runtime, paths, identity)

    monkeypatch.setattr(RunInputs, 'inspect', inspect)
    assert app.main(args) == 0
    assert inspected == ['dtq'] and len(requests) == 2
    after = durable_rows(pipeline.path)['results']
    assert [r for r in before if r[2] != 'dtq'] == [r for r in after if r[2] != 'dtq']


@pytest.mark.parametrize('change', ['vtt', 'sprite'])
def test_changed_inputs_block_skipped_peer_before_interrupted_inference(pipeline, monkeypatch, change):
    from stash_ai.storage.progress import pending_refreshes
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'joytag', 'oppai')
    assert app.main(args) == 0
    with sqlite3.connect(pipeline.path) as db:
        for table in ('results', 'result_index'):
            db.execute('DELETE FROM ' + table + " WHERE model='oppai'")
    if change == 'vtt':
        pipeline.cues = [replace(c, end=c.end-.1) for c in pipeline.cues]
    else:
        original = pipeline.stash.get_media
        monkeypatch.setattr(pipeline.stash, 'get_media',
                            lambda url: original(url) + (b'changed' if '/sprite' in url else b''))
    pipeline.stash.writes.clear()
    pipeline.fail['oppai'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    assert not pipeline.stash.writes
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert ('joytag', 'scene') in pending_refreshes(store, ['scene', 'markers'])['1']
        assert ('joytag', 'markers') in pending_refreshes(store, ['scene', 'markers'])['1']
    pipeline.fail.clear()
    assert app.main(args) == 0
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert not pending_refreshes(store, ['scene', 'markers'])
        assert len({store.result_metadata('1', 'scene', model)['meta']['frame_input_signature']
                    for model in ('joytag', 'oppai')}) == 1
    requests.clear()
    assert app.main(args) == 0
    assert not requests


@pytest.mark.parametrize('models', [('wd',), ('all',)])
def test_ready_rerun_skips_all_media_and_model_work_but_force_verifies(pipeline, monkeypatch, models):
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, *models)
    assert app.main(args) == 0
    assert len(requests) == 2
    assert requests[0].endswith('/vtt') and requests[1].endswith('/sprite')
    before = durable_rows(pipeline.path)
    requests.clear()
    pipeline.stash.writes.clear()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('completed cache loaded a model'))
    original = RunInputs.inspect
    monkeypatch.setattr(RunInputs, 'inspect', lambda *a: pytest.fail('ready rerun verified inputs'))
    assert app.main(args) == 0
    assert not requests and not pipeline.stash.writes
    assert durable_rows(pipeline.path) == before
    monkeypatch.setattr(RunInputs, 'inspect', original)
    assert app.main([*args, '--force-fetch-vtt']) == 0
    assert len(requests) == 2
    assert not pipeline.stash.writes and durable_rows(pipeline.path) == before


def test_no_history_is_needed_to_reuse_inputs(pipeline, monkeypatch):
    requests = media_requests(pipeline, monkeypatch)
    args = [*pipeline.args, '--no-run-history']
    assert app.main(args) == 0
    before = durable_rows(pipeline.path)
    requests.clear()
    assert app.main(args) == 0
    assert not requests and durable_rows(pipeline.path) == before
    assert not pipeline.path.with_name(pipeline.path.name + '.runs.sqlite3').exists()


@pytest.mark.parametrize('change', ['vtt', 'sprite'])
def test_same_url_content_changes_are_opt_in_for_ready_scenes(pipeline, monkeypatch, change):
    requests = media_requests(pipeline, monkeypatch)
    assert app.main(pipeline.args) == 0
    before = durable_rows(pipeline.path)
    loaded = len(pipeline.loaded)
    if change == 'vtt':
        pipeline.cues = [replace(c, end=c.end-.1) for c in pipeline.cues]
    else:
        original = pipeline.stash.get_media
        monkeypatch.setattr(pipeline.stash, 'get_media',
                            lambda url: original(url) + (b'changed' if '/sprite' in url else b''))
    requests.clear()
    assert app.main(pipeline.args) == 0
    assert not requests and len(pipeline.loaded) == loaded
    assert durable_rows(pipeline.path) == before
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    assert len(requests) == 2 and len(pipeline.loaded) == loaded + 1
    assert durable_rows(pipeline.path)['results'] != before['results']


@pytest.mark.parametrize('missing', ['result', 'refresh'])
def test_changed_inputs_in_incomplete_model_refresh_completed_peers(pipeline, monkeypatch, missing):
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'wd', 'dtq')
    assert app.main(args) == 0
    if missing == 'result':
        with sqlite3.connect(pipeline.path) as db:
            for table in ('results', 'result_index'):
                db.execute('DELETE FROM ' + table + " WHERE model='dtq'")
    else:
        from stash_ai.storage.progress import request_refresh
        with StateStore(pipeline.path, scope_for(app.STASH_URL)) as store:
            request_refresh(store, ['1'], ['dtq'], ['scene', 'markers'])
    pipeline.cues = [replace(c, end=c.end-.1) for c in pipeline.cues]
    requests.clear()
    loaded = len(pipeline.loaded)
    inspected = []
    original = RunInputs.inspect

    def inspect(self, runtime, paths, identity):
        inspected.append(identity['source'])
        return original(self, runtime, paths, identity)

    monkeypatch.setattr(RunInputs, 'inspect', inspect)
    assert app.main(args) == 0
    assert set(inspected) == {'wd', 'dtq'} and len(requests) == 2
    assert {m.settings.source.key for m in pipeline.loaded[loaded:]} == {'wd', 'dtq'}
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert len({store.result_metadata('1', 'scene', model)['meta']['frame_input_signature']
                    for model in ('wd', 'dtq')}) == 1


def test_ready_model_read_failure_blocks_publication_when_other_model_is_incomplete(pipeline, monkeypatch):
    from stash_ai.storage.input_checks import pending
    media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'wd', 'dtq')
    assert app.main(args) == 0
    with sqlite3.connect(pipeline.path) as db:
        for table in ('results', 'result_index'):
            db.execute('DELETE FROM ' + table + " WHERE model='dtq'")
    pipeline.cues = [replace(c, end=c.end-.1) for c in pipeline.cues]
    original = RunInputs.inspect

    def inspect(self, runtime, paths, identity):
        if identity['source'] == 'wd':
            raise OSError('VTT unavailable')
        return original(self, runtime, paths, identity)

    monkeypatch.setattr(RunInputs, 'inspect', inspect)
    pipeline.stash.writes.clear()
    assert app.main(args) == 1
    assert not pipeline.stash.writes
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert pending(store, '1') == {'wd': 'read_error'}


def test_force_read_failure_is_retried_by_the_next_normal_rerun(pipeline, monkeypatch):
    from stash_ai.storage.input_checks import pending
    requests = media_requests(pipeline, monkeypatch)
    assert app.main(pipeline.args) == 0
    loaded = len(pipeline.loaded)
    original = pipeline.stash.get_media
    monkeypatch.setattr(pipeline.stash, 'get_media', lambda url: (_ for _ in ()).throw(OSError('offline')))
    assert app.main(pipeline.args) == 0  # Ready cached inputs remain usable.
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 1
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert pending(store, '1') == {'wd': 'read_error'}
    monkeypatch.setattr(pipeline.stash, 'get_media', original)
    requests.clear()
    assert app.main(pipeline.args) == 0
    assert len(requests) == 2 and len(pipeline.loaded) == loaded
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert pending(store, '1') == {}


def test_completed_models_with_different_saved_inputs_verify_and_align(pipeline, monkeypatch):
    requests = media_requests(pipeline, monkeypatch)
    args = model_args(pipeline, 'wd', 'dtq')
    assert app.main(args) == 0
    pipeline.cues = [replace(c, end=c.end-.1) for c in pipeline.cues]
    assert app.main([*pipeline.args, '--force-fetch-vtt']) == 0
    loaded = len(pipeline.loaded)
    requests.clear()
    assert app.main(args) == 0
    assert len(requests) == 2
    assert [m.settings.source.key for m in pipeline.loaded[loaded:]] == ['dtq']
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        assert len({store.result_metadata('1', 'scene', model)['meta']['frame_input_signature']
                    for model in ('wd', 'dtq')}) == 1


def test_incomplete_scene_does_not_disable_reuse_for_other_scenes(pipeline, monkeypatch):
    from copy import deepcopy
    from tests.helpers.cleanup import FastStash
    stash = FastStash()
    stash.scenes['2'] = dict(deepcopy(stash.scene), id='2')
    for scene in stash.scenes.values():
        scene['paths'] = dict(vtt='/vtt/' + scene['id'], sprite='/sprite/' + scene['id'])
    pipeline.stash = stash
    monkeypatch.setattr(app.services, 'create_stash_client', lambda _: stash)
    requests = media_requests(pipeline, monkeypatch)
    args = ['--models', 'wd', 'dtq', '--no-hierarchy', '--state-db', str(pipeline.path)]
    assert app.main(args) == 0
    assert len(requests) == 4
    with sqlite3.connect(pipeline.path) as db:
        for table in ('results', 'result_index'):
            db.execute('DELETE FROM ' + table + " WHERE model='dtq' AND scene='2'")
    loaded = len(pipeline.loaded)
    requests.clear()
    assert app.main(args) == 0
    assert len(requests) == 2 and all(url.endswith('/2') for url in requests)
    assert [m.settings.source.key for m in pipeline.loaded[loaded:]] == ['dtq']


def test_changed_later_input_reopens_earlier_tagger_only_for_stale_scene(pipeline, monkeypatch):
    from copy import deepcopy
    from tests.helpers.cleanup import FastStash
    stash = FastStash()
    stash.scenes['2'] = dict(deepcopy(stash.scene), id='2')
    for scene in stash.scenes.values():
        scene['paths'] = dict(vtt='/vtt/' + scene['id'], sprite='/sprite/' + scene['id'])
    pipeline.stash = stash
    monkeypatch.setattr(app.services, 'create_stash_client', lambda _: stash)
    requests = media_requests(pipeline, monkeypatch)
    args = ['--models', 'wd', 'dtq', '--no-hierarchy', '--state-db', str(pipeline.path)]
    assert app.main(args) == 0
    with sqlite3.connect(pipeline.path) as db:
        for table in ('results', 'result_index'):
            db.execute('DELETE FROM ' + table + " WHERE (model='wd' AND scene='2') OR (model='dtq' AND scene='1')")
    original = stash.get_media
    monkeypatch.setattr(stash, 'get_media', lambda url: original(url) +
                        (b'changed' if '/sprite/1' in url else b''))
    loaded = len(pipeline.loaded)
    requests.clear()
    assert app.main(args) == 0
    # WD first works on scene 2 and closes; DTQ discovers changed scene 1.
    # The second WD instance refreshes only scene 1, without reusing a closed model.
    assert [m.settings.source.key for m in pipeline.loaded[loaded:]] == ['wd', 'dtq', 'wd']
    assert all(model.closed for model in pipeline.loaded)
    assert len(requests) == 4
    with StateStore(pipeline.path, scope_for(app.STASH_URL), True) as store:
        for scene in ('1', '2'):
            assert len({store.result_metadata(scene, 'scene', model)['meta']['frame_input_signature']
                        for model in ('wd', 'dtq')}) == 1


def test_pending_threshold_change_falls_back_to_per_scene_error_handling(pipeline, monkeypatch, tmp_path):
    from stash_ai.inference.identity import model_identity
    from stash_ai.inference.reuse import cached_reuse
    selected = settings(monkeypatch, tmp_path, scene_cache_floor=.01, marker_cache_floor=.01)
    media_requests(pipeline, monkeypatch)
    assert app.main(pipeline.args) == 0
    selected[0] = replace(selected[0], scene_threshold=.45)
    with StateStore(pipeline.path, scope_for(app.STASH_URL)) as store:
        store.prepare('1', {'markers': {'contributions': []}})
        identity = model_identity(app, selected[0], pipeline.vocab)
        assert cached_reuse(app, store, '1', selected[0], identity,
                            pipeline.stash.scene['paths'], ['scene', 'markers']) is None
        assert store.has_pending_publication('1')


@pytest.mark.parametrize('change', ['batch', 'threshold', 'easier_retention'])
def test_compatible_setting_edits_do_not_fetch_or_reinfer(pipeline, monkeypatch, tmp_path, change):
    selected = settings(monkeypatch, tmp_path, scene_cache_floor=.01, marker_cache_floor=.01)
    requests = media_requests(pipeline, monkeypatch)
    assert app.main(pipeline.args) == 0
    before = durable_rows(pipeline.path)['results']
    changes = {'batch': dict(batch_size=8), 'threshold': dict(scene_threshold=.45, marker_threshold=.75),
               'easier_retention': dict(scene_cache_floor=.02, marker_cache_floor=.02)}
    selected[0] = replace(selected[0], **changes[change])
    requests.clear()
    loaded = len(pipeline.loaded)
    assert app.main(pipeline.args) == 0
    assert not requests and len(pipeline.loaded) == loaded
    assert durable_rows(pipeline.path)['results'] == before


def strip_registry_context(path):
    with sqlite3.connect(path) as db:
        for scope, scene, model, signature, data in db.execute('SELECT * FROM inference_identities').fetchall():
            value = decode(data)
            value.pop('_signature_proof', None)
            value.pop('_input_options', None)
            db.execute('UPDATE inference_identities SET data=? WHERE scope=? AND scene=? AND model=? AND signature=?',
                       (encode(value), scope, scene, model, signature))


def test_existing_plain_registry_reuses_without_bootstrap_fetch(pipeline, monkeypatch):
    requests = media_requests(pipeline, monkeypatch)
    assert app.main([*pipeline.args, '--no-run-history']) == 0
    strip_registry_context(pipeline.path)
    before = durable_rows(pipeline.path)
    requests.clear()
    assert app.main(pipeline.args) == 0
    assert not requests and durable_rows(pipeline.path) == before


def test_plain_registry_with_sampling_verifies_once_then_caches_compatible_identity(pipeline, monkeypatch, tmp_path):
    selected = settings(monkeypatch, tmp_path)
    requests = media_requests(pipeline, monkeypatch)
    monkeypatch.setattr(app, 'MAX_FRAMES_PER_SCENE', 4)
    assert app.main(pipeline.args) == 0
    strip_registry_context(pipeline.path)
    selected[0] = replace(selected[0], batch_size=8)
    loaded = len(pipeline.loaded)
    requests.clear()
    assert app.main(pipeline.args) == 0
    assert len(requests) == 2 and len(pipeline.loaded) == loaded
    requests.clear()
    assert app.main(pipeline.args) == 0
    assert not requests and len(pipeline.loaded) == loaded


@pytest.mark.parametrize('change', ['sampling', 'paths', 'restart'])
def test_changed_input_options_and_restart_use_normal_verification(pipeline, monkeypatch, change):
    requests = media_requests(pipeline, monkeypatch)
    assert app.main(pipeline.args) == 0
    args = pipeline.args
    if change == 'sampling':
        monkeypatch.setattr(app, 'MAX_FRAMES_PER_SCENE', 4)
    elif change == 'paths':
        pipeline.stash.scene['paths']['vtt'] += '?revision=2'
    else:
        args = [*args, '--restart']
    requests.clear()
    assert app.main(args) == 0
    assert len(requests) == 2


@pytest.mark.parametrize('mode', [
    ['--merge-only'], ['--rethreshold-scenes'], ['--db-check'], ['--db-update'],
    ['--model-report', 'out.json'], ['--cache-report', 'out.json'],
    ['--run-report', 'out.json'], ['--reuse-report', 'out.json'], ['--cleanup-only'],
])
def test_force_fetch_is_rejected_for_commands_without_inference(mode):
    with pytest.raises(SystemExit) as exc:
        parse_args(app, [*mode, '--force-fetch-vtt'])
    assert exc.value.code == 2


def test_force_fetch_is_explicit_and_profile_configuration_distinguishes_it():
    default = parse_args(app, [])
    forced = parse_args(app, ['--force-fetch-vtt', '--resume'])
    assert default.force_fetch_vtt is False and forced.force_fetch_vtt is True
    ordinary, verified = configuration(app, default), configuration(app, forced)
    assert ordinary['force_fetch_vtt'] is False and verified['force_fetch_vtt'] is True
    assert ordinary['fingerprint'] != verified['fingerprint']
