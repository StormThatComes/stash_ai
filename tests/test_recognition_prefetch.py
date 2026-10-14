from dataclasses import replace
from threading import Event, get_ident

import pytest
from PIL import Image

from stash_ai.domain import Cue
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.prefetch import FramePrefetch, ScenePrefetch
from stash_ai.runtime.cancellation import current
from stash_ai.runtime.cleanup import closing
from tests.helpers.recognition import Backend
from tests.test_recognition_run import setup, run


class Frames:
    def __init__(self, count=10, *, fail_at=None):
        self.count, self.fail_at = count, fail_at
        self.originals, self.closed = [], Event()
        self.third = Event()

    def frames(self, start=0):
        try:
            for index in range(start, self.count):
                if index == self.fail_at:
                    raise ValueError('source failed')
                with closing(Image.new('RGB', (8, 8), (index, 0, 0))) as image:
                    self.originals.append(image)
                    if len(self.originals) == 3:
                        self.third.set()
                    yield index, Cue(index, index+1, 'sprite', (0, 0, 8, 8)), image
        finally:
            self.closed.set()


def assert_closed(images):
    for image in images:
        with pytest.raises(ValueError, match='closed'):
            image.getpixel((0, 0))


def track_copies(monkeypatch):
    copies, original = [], Image.Image.copy
    def copy(image):
        value = original(image)
        copies.append(value)
        return value
    monkeypatch.setattr(Image.Image, 'copy', copy)
    return copies


def test_frames_keep_order_pixels_and_resume_index_after_source_crops_expire():
    source = Frames()
    stream = FramePrefetch(source, 3, 2).launch()
    try:
        seen = []
        for index, cue, image in stream:
            with closing(image):
                seen.append((index, cue.start, image.getpixel((0, 0))[0]))
        assert seen == [(index, index, index) for index in range(3, 10)]
    finally:
        stream.close()
    assert source.closed.is_set() and not stream.channel.worker.is_alive()
    assert_closed(source.originals)


def test_full_queue_stops_production_and_close_frees_all_owned_images(monkeypatch):
    copies, source = track_copies(monkeypatch), Frames(100)
    stream = FramePrefetch(source, 0, 2).launch()
    try:
        assert source.third.wait(2)
        assert stream.channel.queue.qsize() == 2
        assert len(source.originals) == 3  # two queued crops and one producer
    finally:
        stream.close()
    assert len(copies) <= 3 and source.closed.is_set()
    assert_closed(copies + source.originals)


def test_source_failure_arrives_after_completed_frames_and_closes_generator():
    source, seen = Frames(fail_at=3), []
    stream = FramePrefetch(source, 0, 2).launch()
    try:
        with pytest.raises(ValueError, match='source failed'):
            for index, _, image in stream:
                with closing(image):
                    seen.append(index)
    finally:
        stream.close()
    assert seen == [0, 1, 2] and source.closed.is_set()


def test_close_cancels_an_active_frame_read():
    entered, release, stopped = Event(), Event(), Event()
    class Waiting:
        def frames(self, start):
            token = current()
            token.subscribe(release.set)
            try:
                entered.set()
                assert release.wait(2)
                token.check()
                yield  # cancellation must happen before this
            finally:
                token.unsubscribe(release.set)
                stopped.set()
    stream = FramePrefetch(Waiting(), 0, 1).launch()
    try:
        assert entered.wait(2)
    finally:
        stream.close()
    assert stopped.is_set() and not stream.channel.worker.is_alive()


def test_interrupted_thread_start_stops_started_producer_and_frees_images(monkeypatch):
    import stash_ai.recognition.prefetch as prefetch
    original, copies, source = prefetch.Thread.start, track_copies(monkeypatch), Frames(100)
    def start(thread):
        original(thread)
        raise KeyboardInterrupt()
    monkeypatch.setattr(prefetch.Thread, 'start', start)
    stream = FramePrefetch(source, 0, 2)
    with pytest.raises(KeyboardInterrupt):
        stream.launch()
    assert not stream.channel.worker.is_alive()
    assert_closed(copies + source.originals)


def readers(client):
    created, factory = [], client.new_media_reader
    def make():
        reader = factory()
        reader.closed = False
        reader.close = lambda: setattr(reader, 'closed', True)
        created.append(reader)
        return reader
    client.new_media_reader = make
    return created


def test_next_scene_loads_during_inference_with_isolated_readers_and_caller_owned_state(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    owner, ready, sources, calls = get_ident(), Event(), [], []
    created = readers(client)
    class Source(SpriteInput):
        def __init__(self, reader, scene, config):
            assert get_ident() != owner and reader is not client
            super().__init__(reader, scene, config)
            self.closed = False
            sources.append(self)
            if scene['id'] == '2':
                ready.set()
        def close(self):
            self.closed = True
            super().close()
    model = Backend()
    embed = model.embed
    def infer(image, face):
        assert get_ident() == owner
        if not model.calls:
            assert ready.wait(2), 'next scene was not prepared while current inference was active'
            assert all(not source.closed for source in sources)
        return embed(image, face)
    model.embed = infer
    services.source, services.backend = Source, lambda *args: model
    original = FaceDatabase.progress
    def progress(database, *args, **kwargs):
        calls.append(get_ident())
        return original(database, *args, **kwargs)
    monkeypatch.setattr(FaceDatabase, 'progress', progress)
    assert run(config, services, '1', '2') == 0
    assert model.calls == 6 and model.closed and set(calls) == {owner}
    assert len(created) == 2 and created[0].media is not created[1].media
    assert all(reader.closed for reader in created) and all(source.closed for source in sources)


def test_failed_next_scene_input_does_not_drop_current_or_later_scene(tmp_path, monkeypatch):
    config, client, services, backends = setup(tmp_path, monkeypatch)
    created = readers(client)
    def source(reader, scene, config):
        if scene['id'] == '2':
            raise ValueError('next input unavailable')
        return SpriteInput(reader, scene, config)
    services.source = source
    assert run(config, services, '1', '2', '3') == 1
    assert len(backends) == 1 and backends[0].calls == 6
    assert all(reader.closed for reader in created)
    with FaceDatabase(config.database) as state:
        assert [tuple(row) for row in state.connection.execute('SELECT scene,status FROM scans ORDER BY scene')] == [
            ('1', 'complete'), ('3', 'complete')]


def test_unconsumed_ready_scene_is_closed_on_shutdown(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    created = readers(client)
    inputs = ScenePrefetch(client, config, services.source)
    inputs.submit('1')
    prepared = inputs.pending.result(timeout=2)
    assert prepared.source.cache
    inputs.close()
    assert not prepared.source.cache and created[0].closed
    inputs.close()


def test_interrupt_during_handoff_retains_future_until_input_cleanup(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    created = readers(client)
    inputs = ScenePrefetch(client, config, services.source)
    inputs.submit('1')
    prepared = inputs.pending.result(timeout=2)
    original = inputs.pending.result
    def interrupt(*args, **kwargs):
        monkeypatch.setattr(inputs.pending, 'result', original)
        raise KeyboardInterrupt()
    monkeypatch.setattr(inputs.pending, 'result', interrupt)
    with pytest.raises(KeyboardInterrupt):
        inputs.take('1')
    inputs.close()
    assert not prepared.source.cache and created[0].closed


def test_shutdown_cancels_in_progress_next_input(tmp_path, monkeypatch):
    config, client, _, _ = setup(tmp_path, monkeypatch)
    created, entered, release = readers(client), Event(), Event()
    def source(*args):
        token = current()
        token.subscribe(release.set)
        try:
            entered.set()
            assert release.wait(2)
            token.check()
        finally:
            token.unsubscribe(release.set)
        pytest.fail('cancelled input was allowed to finish')
    inputs = ScenePrefetch(client, config, source)
    inputs.submit('1')
    assert entered.wait(2)
    inputs.close()
    assert created[0].closed


def test_disabling_prefetch_preserves_scan_identity_and_reuses_cached_models(tmp_path, monkeypatch):
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    config, client, services, backends = setup(tmp_path, monkeypatch)
    synchronous = replace(config, data=plain(config.data))
    synchronous.data['execution'].update(prefetch_frames=0, prefetch_scenes=False)
    assert scan_identity(config, namespace(config)) == scan_identity(synchronous, namespace(synchronous))
    client.new_media_reader = lambda: pytest.fail('disabled scene prefetch started an HTTP reader')
    assert run(synchronous, services, '1', '2') == 0
    assert backends[0].calls == 6
    del client.new_media_reader  # restore the fake reader factory for the cached run
    assert run(config, services, '1', '2') == 0 and len(backends) == 1


def test_isolated_recognition_reader_keeps_scene_validation_and_read_only_policy():
    from stash_ai.recognition.stash import RecognitionStash
    client = RecognitionStash.__new__(RecognitionStash)
    client.url, client.api_key, client.request_timeout, client.http_retries = 'http://stash/', 'synthetic-key', (5., 120.), 2
    client.report, client.api, client.media = None, object(), object()
    reader = client.new_media_reader()
    try:
        assert reader.dry_run and reader.api is not client.api and reader.media is not client.media
        assert reader.media.headers['ApiKey'] == client.api_key
        reader.gql = lambda *args: {'findScene': {'id': '2'}}
        assert reader.scene('2') == {'id': '2'}
        with pytest.raises(RuntimeError, match='ID'):
            reader.scene('1')
    finally:
        reader.close()
