import json

import pytest

from stash_ai.recognition.backend import LazyBackend
from stash_ai.recognition.database import FaceDatabase
from tests.helpers.recognition import Backend
from tests.test_recognition_run import setup, run


def test_models_are_shared_across_scenes_and_remain_lazy_on_cached_run(tmp_path, monkeypatch):
    config, _, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    assert len(backends) == 1 and backends[0].calls == 6 and backends[0].closed
    factory = services.backend
    assert run(config, services, '1', '2') == 0
    assert len(backends) == 1 and services.backend is factory


def test_scan_and_video_portraits_share_the_invocation_backend(tmp_path, monkeypatch):
    from tests.test_recognition_metadata import setup as video_setup, run as video_run
    config, _, services = video_setup(tmp_path, monkeypatch)
    backends = []
    def factory(*args):
        model = Backend()
        backends.append(model)
        return model
    services.backend = factory
    assert video_run(config, services, '1', '2') == 0
    assert len(backends) == 1 and backends[0].calls > 6 and backends[0].closed


def test_borrowed_cleanup_keeps_models_alive_and_owner_closes_once():
    model, closes, loads = Backend(), [], []
    model.close = lambda: closes.append(1)
    owner = LazyBackend(None, {}, lambda *args: model, loaded=lambda value: loads.append(value))
    first = owner.borrow()
    first.close()
    second = owner.borrow(None, {})
    assert second.dimensions == 4 and not closes and loads == [model]
    owner.close()
    owner.close()
    second.close()
    assert closes == [1]
    with pytest.raises(RuntimeError, match='closed'):
        owner.borrow()


@pytest.mark.parametrize('method', ['detect', 'embed'])
def test_failed_inference_discards_models_without_masking_error(method):
    models = [Backend(), Backend()]
    loaded = iter(models)
    owner = LazyBackend(None, {}, lambda *args: next(loaded))
    lease = owner.borrow()
    def fail(*args):
        raise ValueError('inference failed')
    def bad_close():
        raise RuntimeError('close failed')
    setattr(models[0], method, fail)
    models[0].close = bad_close
    with pytest.raises(ValueError, match='inference failed') as caught:
        getattr(lease, method)(None, *([None] if method == 'embed' else []))
    assert any('close failed' in note for note in caught.value.__notes__)
    replacement = owner.borrow()
    assert replacement.model is models[1]
    lease.close()
    assert not models[1].closed
    owner.close()
    assert models[1].closed


def test_backend_registration_failure_closes_the_model():
    model = Backend()
    def fail(value):
        raise RuntimeError('history failed')
    owner = LazyBackend(None, {}, lambda *args: model, loaded=fail)
    with pytest.raises(RuntimeError, match='history failed'):
        owner.borrow()
    assert model.closed and owner.model is None
    owner.close()


def test_failed_scene_reinitializes_backend_and_keeps_completed_checkpoints(tmp_path, monkeypatch):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    models = [Backend(fail_at=2), Backend()]
    pending = iter(models)
    services.backend = lambda *args: next(pending)
    assert run(config, services, '1', '2') == 1
    assert all(model.closed for model in models) and models[1].calls == 3
    with FaceDatabase(config.database) as state:
        scans = state.connection.execute('SELECT scene,cursor,status FROM scans ORDER BY scene').fetchall()
        assert [tuple(row) for row in scans] == [('1', 1, 'partial'), ('2', 3, 'complete')]


def test_interrupt_closes_shared_models_and_records_only_completed_cues(tmp_path, monkeypatch):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    model = Backend()
    embed = model.embed
    def interrupt(image, face):
        if model.calls == 1:
            raise KeyboardInterrupt()
        return embed(image, face)
    model.embed = interrupt
    services.backend = lambda *args: model
    with pytest.raises(KeyboardInterrupt):
        run(config, services, '1', '2')
    assert model.closed
    with FaceDatabase(config.database) as state:
        assert tuple(state.connection.execute('SELECT cursor,status FROM scans').fetchone()) == (1, 'partial')
        history = state.connection.execute('SELECT * FROM runs').fetchone()
        assert history['status'] == 'interrupted'
        assert json.loads(history['selection']) == ['1', '2']


import json
import os
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('failure', ['embed', 'detect'])
def test_video_backend_lease_is_reacquired_after_one_candidate_failure(tmp_path, monkeypatch, failure):
    from stash_ai.recognition.database import FaceDatabase
    from tests.helpers.recognition import Backend
    from tests.test_recognition_metadata import setup, run, Decoder
    config, client, services = setup(tmp_path, monkeypatch)
    models = []
    def load(*a):
        first = not models
        model = Backend(fail_at=4 if first and failure == 'embed' else None)
        if first and failure == 'detect':
            detect = model.detect
            def failing_detect(image):
                if model.calls == 3:
                    raise RuntimeError('synthetic portrait detection failure')
                return detect(image)
            model.detect = failing_detect
        models.append(model)
        return model
    services.backend = load
    assert run(config, services, '1') == 0
    assert len(models) == 2 and models[0].calls == (4 if failure == 'embed' else 3) and models[1].calls == 1
    assert len(Decoder.instances[0].calls) == 2
    with FaceDatabase(config.database) as db:
        video = db.connection.execute("SELECT image,metadata FROM portraits WHERE json_extract(metadata,'$.source_type')='video'").fetchall()
        assert len(video) == 2 and sum(r['image'] is not None for r in video) == 1
        final = json.loads(db.connection.execute("SELECT metadata FROM portraits WHERE status='published'").fetchone()[0])
        assert final['source_type'] == 'video'

