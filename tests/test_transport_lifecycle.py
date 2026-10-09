"""Mutation safety and recovery from optional/background thread failures."""
from threading import Event
from types import SimpleNamespace

import pytest
import requests

from stash_ai.runtime.report import RunReport
from stash_ai.stash_api.transport import StashTransport


@pytest.mark.parametrize('prefix', ['# comment\n', '\ufeff,# comment\r\n ', '\n# mutation in comment\n'])
def test_commented_mutation_is_never_sent_in_dry_run(prefix):
    client = StashTransport(True, url='http://stash.test', api_key='', request_timeout=1, http_retries=3)
    client.api.post = lambda *a, **k: pytest.fail('dry-run sent a mutation')
    try:
        with pytest.raises(RuntimeError, match='mutation attempted'):
            client.gql(prefix + 'mutation StashAIWrite { tagDestroy }')
    finally:
        client.close()


def test_commented_mutation_is_not_retried():
    client = StashTransport(False, url='http://stash.test', api_key='', request_timeout=1, http_retries=3)
    calls = []
    def post(*a, **k):
        calls.append(k)
        raise requests.ConnectionError('ambiguous write')
    client.api.post = post
    try:
        with pytest.raises(requests.ConnectionError):
            client.gql('# comment\nmutation StashAIWrite { tagDestroy }')
        assert len(calls) == 1
    finally:
        client.close()


def test_graphql_document_read_classification():
    assert StashTransport._read_only('# mutation\nquery Read { field(arg: "mutation") }')
    assert StashTransport._read_only('{ field }')
    assert not StashTransport._read_only('query Read { field } mutation Write { update }')
    assert not StashTransport._read_only('subscription Watch { changes }')
    assert not StashTransport._read_only('unknown')


def test_closer_can_restart_after_failed_start(monkeypatch):
    from stash_ai.stash_api import deadlines
    from threading import Thread
    closed = Event()
    starts = []
    class Worker(Thread):
        def start(self):
            starts.append(1)
            if len(starts) == 1:
                raise RuntimeError('cannot start')
            super().start()
    monkeypatch.setattr(deadlines, '_closer', None)
    monkeypatch.setattr(deadlines, 'Thread', Worker)
    response = SimpleNamespace(close=closed.set)
    with pytest.raises(RuntimeError, match='cannot start'):
        deadlines.close_later(response)
    assert deadlines._closer is None
    deadlines.close_later(response)
    assert closed.wait(1)


def test_heartbeat_start_failure_does_not_abort_request_or_leak_active_entry(monkeypatch):
    from stash_ai.runtime import timing
    report = RunReport(write=lambda _: None)
    class Worker:
        def __init__(self, **kwargs): pass
        def start(self): raise RuntimeError('cannot start')
        def is_alive(self): return False
    monkeypatch.setattr(timing.threading, 'Thread', Worker)
    with report.request('read', True):
        assert len(report.active_requests) == 1
    assert not report.active_requests
    assert report.counts['heartbeat_start_failures'] == 1


def test_short_requests_share_one_ticker(monkeypatch):
    from stash_ai.runtime import timing
    original = timing.threading.Thread
    starts = []
    def thread(**kwargs):
        starts.append(kwargs['name'])
        return original(**kwargs)
    monkeypatch.setattr(timing.threading, 'Thread', thread)
    report = RunReport(write=lambda _: None)
    for _ in range(100):
        with report.request('read', True): pass
    assert starts == ['stash-request-heartbeat']
