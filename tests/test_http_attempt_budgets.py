"""Elapsed boundaries cover a single mutation and all recognition media reads."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread, enumerate as threads
import time

import pytest
import requests

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import bounded_media
from stash_ai.recognition.publication import perform
from stash_ai.recognition.video import VideoFrames
from stash_ai.runtime.mutations import write
from stash_ai.runtime.report import RunReport
from stash_ai.stash_api.attempts import MutationAttempt
from stash_ai.stash_api.deadlines import ReadDeadlineExceeded
from stash_ai.stash_api.transport import StashTransport


@contextmanager
def trickling_source(*, headers=False):
    stop, disconnected = Event(), Event()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args): pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            self.do_GET()

        def do_GET(self):
            calls.append((self.command, self.headers.get('ApiKey')))
            try:
                if headers:
                    for byte in b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n':
                        self.wfile.write(bytes([byte])); self.wfile.flush()
                        if stop.wait(.025): return
                else:
                    self.send_response(200)
                    self.send_header('Content-Length', '100')
                    self.end_headers()
                for _ in range(100):
                    self.wfile.write(b'x'); self.wfile.flush()
                    if stop.wait(.025): return
            except (BrokenPipeError, ConnectionResetError, OSError):
                disconnected.set()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    worker.start()
    client = StashTransport(False, url=f'http://127.0.0.1:{server.server_port}', api_key='test-key',
                            request_timeout=.12, http_retries=3)
    client.api.trust_env = client.media.trust_env = False
    try:
        yield client, calls, disconnected
    finally:
        stop.set(); client.close(); server.shutdown(); worker.join(); server.server_close()


@pytest.mark.parametrize('headers', [True, False])
def test_single_mutation_trickle_is_bounded_and_uncertain_intent_remains(tmp_path, headers):
    with trickling_source(headers=headers) as (client, calls, disconnected), FaceDatabase(tmp_path/'face.sqlite') as db:
        report = RunReport(write=lambda *args: None)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match='unknown') as raised:
            perform(db, client, report, scope='scope', scene='', person='person', kind='name', target='101',
                    payload={'name': 'Requested'}, precondition={'name': 'Before'},
                    callback=lambda: client.gql('mutation Test { update }'))
        assert time.monotonic() - started < .5
        assert raised.value.write_outcome == 'unknown'
        assert calls == [('POST', 'test-key')]
        assert report.counts['mutation_unknown'] == 1 and report.counts['mutation_confirmed'] == 0
        assert db.connection.execute('SELECT status FROM publications').fetchone()[0] == 'unknown'
        assert disconnected.wait(1)


@pytest.mark.parametrize('headers', [True, False])
def test_recognition_media_trickle_closes_operation_within_deadline(headers):
    with trickling_source(headers=headers) as (client, calls, disconnected):
        started = time.monotonic()
        with pytest.raises(ReadDeadlineExceeded):
            bounded_media(client, client.url + '/sprite', 1024, .12)
        assert time.monotonic() - started < .5
        assert calls == [('GET', 'test-key')]
        assert disconnected.wait(1)


def test_pre_dispatch_expiry_is_failed_and_gate_forbids_late_dispatch(monkeypatch):
    import stash_ai.stash_api.deadlines as deadlines
    client = StashTransport(False, url='http://stash/', api_key='', request_timeout=1., http_retries=5)
    calls, report = [], RunReport(write=lambda *args: None)
    client.api.post = lambda *args, **kwargs: calls.append(1)
    original = deadlines.monotonic
    monkeypatch.setattr(deadlines, 'monotonic', lambda: original() + 2)
    try:
        with pytest.raises(RuntimeError, match='before dispatch') as raised:
            write(report, 'test', lambda: client.gql('mutation Test { update }'))
        assert raised.value.write_outcome == 'failed'
        assert calls == [] and report.counts['mutation_failed'] == 1 and not report.counts['mutation_unknown']
    finally:
        client.close()
    gate = MutationAttempt()
    assert not gate.expire()
    with pytest.raises(ReadDeadlineExceeded):
        gate.dispatch(lambda timeout: calls.append(1), 1., original() + 10)
    assert calls == []


def test_late_success_does_not_confirm_or_retry_a_mutation():
    client = StashTransport(False, url='http://stash/', api_key='', request_timeout=.02, http_retries=3)
    release, closed, entered = Event(), Event(), Event()
    calls, report = [], RunReport(write=lambda *args: None)
    response = requests.Response()
    response.status_code, response._content = 200, b'{"data":{"ok":true}}'
    response.close = closed.set
    def post(*args, **kwargs):
        calls.append(1); entered.set(); release.wait(1)
        return response
    client.api.post = post
    try:
        with pytest.raises(RuntimeError, match='unknown'):
            write(report, 'test', lambda: client.gql('mutation Test { update }'))
        assert entered.is_set() and report.counts['mutation_unknown'] == 1
        release.set()
        assert closed.wait(1)
        assert calls == [1] and report.counts['mutation_confirmed'] == 0
    finally:
        release.set(); client.close()


def test_unexpected_body_failure_closes_response_and_preserves_primary():
    class BadResponse(requests.Response):
        @property
        def content(self): raise ValueError('adapter body failure')

    client = StashTransport(False, url='http://stash/', api_key='', request_timeout=1., http_retries=0)
    closed = Event()
    response = BadResponse()
    response.close = closed.set
    client.api.post = lambda *args, **kwargs: response
    try:
        with pytest.raises(ValueError, match='adapter body failure'):
            client.gql('mutation Test { update }')
        assert closed.is_set()
    finally:
        client.close()


def test_video_bridge_close_cancels_a_trickling_body(tmp_path):
    from dataclasses import replace
    config = load(root=tmp_path)
    config = replace(config, data={**config.data, 'portraits': {**config.data['portraits'], 'timeout_seconds': .2}})
    with trickling_source() as (client, calls, disconnected):
        decoder = VideoFrames(client, {'paths': {'stream': '/video'}}, config).__enter__()
        address = f'http://127.0.0.1:{decoder.server.server_port}{decoder.path}'
        response = requests.get(address, stream=True, timeout=1)
        started = time.monotonic()
        try:
            decoder.close()
            assert time.monotonic() - started < .6 and not decoder.thread.is_alive()
            assert disconnected.wait(1)
        finally:
            response.close(); decoder.close()
