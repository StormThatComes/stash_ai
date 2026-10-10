"""Every producer is stopped, and cleanup cannot hide a primary failure."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from stash_ai.inference.batch_stream import PreparedBatchStream
from stash_ai.inference.input_prefetch import SceneInputPrefetch
from stash_ai.runtime.cancellation import Cancellation
from stash_ai.runtime.cleanup import cleanup
from stash_ai.stash_api.transport import StashTransport


def fail():
    raise OSError('cleanup failed')


def test_cancel_runs_every_callback_even_if_one_fails():
    token, calls = Cancellation(), []
    def good(): calls.append('good')
    def bad():
        calls.append('bad')
        fail()
    token.subscribe(good)
    token.subscribe(bad)
    with pytest.raises(OSError, match='cleanup failed'):
        token.cancel()
    assert sorted(calls) == ['bad', 'good'] and token.event.is_set()


def test_prefetch_stops_all_queues_before_joining_a_failed_stream():
    prefetch = SceneInputPrefetch(None, None, None, 'wd')
    prefetch.executor = ThreadPoolExecutor(max_workers=1)
    prefetch.batch_executor = ThreadPoolExecutor(max_workers=2)
    first, second = [PreparedBatchStream(1, None, 'wd') for _ in range(2)]
    prefetch.streams = [first, second]
    first.cancellation.subscribe(fail)
    second.queue.put(('full', []))
    entered, done = Event(), Event()
    def producer():
        entered.set()
        second._send(('blocked', []))
    second.task = prefetch.batch_executor.submit(producer)
    assert entered.wait(2)
    errors = []
    def close():
        try: prefetch.close()
        except BaseException as exc: errors.append(exc)
        finally: done.set()
    closer = Thread(target=close, daemon=True)
    closer.start()
    try:
        assert done.wait(2), 'cleanup joined before stopping the second full queue'
        assert isinstance(errors[0], OSError)
        assert first.stopped and second.stopped
        assert first.queue.empty() and second.queue.empty()
        assert prefetch.executor is None
    finally:
        second.channel.stopped.set()
        closer.join(2)


def test_transport_closes_both_sessions_and_preserves_primary_exception():
    closed = []
    client = StashTransport(False, url='http://stash.test', api_key='', request_timeout=1, http_retries=0)
    client.api.close()
    client.media.close()
    client.api = SimpleNamespace(close=fail)
    client.media = SimpleNamespace(close=lambda: closed.append('media'))
    primary = ValueError('inference failed')
    with pytest.raises(ValueError) as caught:
        try: raise primary
        finally: client.close()
    assert caught.value is primary and closed == ['media']
    assert 'cleanup failed' in primary.__notes__[0]


def test_broken_cleanup_diagnostic_does_not_replace_primary_exception():
    primary = KeyboardInterrupt('interrupted')
    report = SimpleNamespace(detail=lambda _: fail())
    with pytest.raises(KeyboardInterrupt) as caught:
        try: raise primary
        finally: cleanup([fail], report=report)
    assert caught.value is primary and len(primary.__notes__) == 2
