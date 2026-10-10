"""Input lookahead cancellation, transport isolation and shared media snapshots."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event, current_thread, enumerate as threads
from types import SimpleNamespace

import pytest

from stash_ai.inference.input_prefetch import SceneInputPrefetch
from stash_ai.inference.inputs import RunInputs, RunMedia, InspectedInput, VerifiedInput
from stash_ai.runtime.cache import RamCache
from stash_ai.runtime.cancellation import current
from stash_ai.stash_api.transport import StashTransport


def test_close_during_batch_registration_does_not_start_a_late_producer(monkeypatch):
    from stash_ai.inference import batch_stream
    started, cancelled = Event(), Event()
    create = batch_stream.PreparedBatchStream

    def delayed_create(*args):
        token = current()
        token.subscribe(cancelled.set)
        try:
            started.set()
            assert cancelled.wait(5), 'Input cleanup did not cancel batch registration'
            return create(*args)
        finally:
            token.unsubscribe(cancelled.set)

    inputs = SimpleNamespace(open_background_reader=lambda: None,
                             read_with=lambda _: nullcontext(), inspect=lambda *args: InspectedInput(VerifiedInput.capture([], {}), None, 'sig'))
    prefetch = SceneInputPrefetch(None, inputs, None, 'wd')
    monkeypatch.setattr(batch_stream, 'PreparedBatchStream', delayed_create)
    prefetch.submit('next', {}, {}, tagger=object(), prefetch=1)
    assert started.wait(5)
    prefetch.close()
    assert prefetch.executor is None and not prefetch.streams
    assert not any(thread.name.startswith('scene-inputs') for thread in threads())


@pytest.mark.parametrize('limit', [1, 16, 128])
def test_prepared_input_queue_is_bounded_and_releases_inputs_on_close(monkeypatch, limit):
    from stash_ai.inference import batch_stream
    full, refilled = Event(), Event()
    produced = []

    def batches(*args, **kwargs):
        for index in range(limit + 10):
            produced.append(index)
            if index == limit:
                full.set()
            if index == limit + 1:
                refilled.set()
            yield [index], [index]

    inputs = SimpleNamespace(open_background_reader=lambda: SimpleNamespace(close=lambda: None),
                             read_with=lambda _: nullcontext(), inspect=lambda *args: InspectedInput(VerifiedInput.capture([], {}), None, 'sig'))
    monkeypatch.setattr(batch_stream, 'prepared_batches', batches)
    prefetch = SceneInputPrefetch(None, inputs, None, 'wd')
    try:
        prefetch.submit('next', {}, {}, tagger=object(), prefetch=limit)
        prefetch.take('next')
        stream = prefetch.take_batches('next', 0)
        assert full.wait(5), 'Configured CPU lookahead was capped before filling the queue'
        assert produced == list(range(limit + 1))  # Queued inputs plus one awaiting space.
        assert next(stream) == ([0], [0])
        assert refilled.wait(5), 'Reading a batch did not refill the input queue'
        assert produced == list(range(limit + 2))
        assert next(stream) == ([1], [1])
        stream.close()
        assert stream.queue.empty()
    finally:
        prefetch.close()
    assert not any(thread.name.startswith(('scene-inputs', 'scene-batches')) for thread in threads())


def test_ctrl_c_waiting_for_input_keeps_read_token_until_cleanup(monkeypatch):
    started, cancelled, finished = Event(), Event(), Event()
    owner, interrupted = current_thread(), False
    result = Future.result

    def inspect(*args):
        token = current()
        token.subscribe(cancelled.set)
        try:
            started.set()
            assert cancelled.wait(5), 'Interrupted result wait lost the input cancellation token'
            token.check()
        finally:
            token.unsubscribe(cancelled.set)
            finished.set()

    inputs = SimpleNamespace(open_background_reader=lambda: None,
                             read_with=lambda _: nullcontext(), inspect=inspect)
    prefetch = SceneInputPrefetch(None, inputs, None, 'wd')
    prefetch.submit('next', {}, {})
    assert started.wait(5)

    def interrupt(future, *args, **kwargs):
        nonlocal interrupted
        if current_thread() is owner and not interrupted:
            interrupted = True
            raise KeyboardInterrupt('input wait interrupted')
        return result(future, *args, **kwargs)

    monkeypatch.setattr(Future, 'result', interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match='input wait interrupted'):
            prefetch.take('next')
        assert prefetch.pending is not None and prefetch.token is not None
    finally:
        prefetch.close()
    assert cancelled.is_set() and finished.is_set()
    assert not any(thread.name.startswith('scene-inputs') for thread in threads())


def test_background_transport_has_independent_read_only_sessions():
    client = StashTransport(False, url='http://stash.test', api_key='test-key',
                           request_timeout=(1, 2), http_retries=3)
    client.report = object()
    reader = client.new_media_reader()
    try:
        assert reader.media is not client.media and reader.api is not client.api
        assert reader.media.adapters['http://'] is not client.media.adapters['http://']
        assert reader.media.headers['ApiKey'] == client.media.headers['ApiKey']
        assert reader.request_timeout == (1, 2) and reader.http_retries == 3
        assert reader.report is client.report and reader.dry_run
        reader.report = None
        with pytest.raises(RuntimeError, match='mutation attempted'):
            reader.gql('mutation StashAIUnexpectedWrite { tagDestroy }')
        with pytest.raises(ValueError, match='different Stash origin'):
            reader.get_media('http://elsewhere.test/sprite')
        reader.media.cookies.set('reader-only', 'value')
        assert not client.media.cookies.get('reader-only')
    finally:
        reader.close()
        client.close()


def test_concurrent_readers_reject_different_bytes_for_the_same_url():
    rendezvous = Barrier(2)
    original = SimpleNamespace(get_media=lambda _: pytest.fail('Explicit reader was not used'))
    media = RunMedia(original, RamCache(0))

    def read(data):
        def get_media(url):
            rendezvous.wait(timeout=5)
            return data
        with media.read_with(SimpleNamespace(get_media=get_media)):
            try:
                return media.get_media('/same-sprite')
            except RuntimeError as error:
                return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.submit(read, b'first'), executor.submit(read, b'second')
        output = [first.result(timeout=5), second.result(timeout=5)]
    assert sum(isinstance(value, bytes) for value in output) == 1
    failures = [value for value in output if isinstance(value, RuntimeError)]
    assert len(failures) == 1 and 'media changed' in str(failures[0])


def test_verified_media_leaves_background_reader_before_inference():
    settings = SimpleNamespace(STASH_URL='http://stash.test', MAX_FRAMES_PER_SCENE=0)
    vtt = b'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nsprite.jpg#xywh=0,0,1,1\n'
    calls = []

    def normal(url):
        calls.append(('normal', url))
        return b'sprite'

    def background(url):
        calls.append(('background', url))
        return vtt if url.endswith('/vtt') else b'sprite'

    cache = RamCache(0)
    inputs = RunInputs(SimpleNamespace(get_media=normal), cache)
    with inputs.read_with(SimpleNamespace(get_media=background)):
        result = inputs.inspect(settings, {'vtt': '/vtt', 'sprite': '/sprite'}, {})
        cues, media = result.snapshot.cues, result.media
    media.cache.clear()
    media.size = 0
    assert media.get_media(cues[0].image) == b'sprite'
    assert [kind for kind, _ in calls] == ['background', 'background', 'normal']
