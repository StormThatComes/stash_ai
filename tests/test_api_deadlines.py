"""Hard read budgets, live request visibility and unchanged mutation retry rules."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest
import requests

from stash_ai.runtime.report import RunReport
from stash_ai.stash_api.deadlines import ReadDeadlineExceeded
from stash_ai.stash_api.transport import StashTransport


def transport(timeout=(.1, .15), retries=4):
    return StashTransport(False, url='http://stash.test', api_key='secret-value',
                          request_timeout=timeout, http_retries=retries)


def test_total_budget_includes_retry_backoff(monkeypatch):
    client = transport()
    calls = []
    client.api.post = lambda *args, **kwargs: (calls.append(kwargs) or SimpleNamespace(
        status_code=503, close=lambda: None))
    started = time.monotonic()
    try:
        with pytest.raises(ReadDeadlineExceeded):
            client.gql('query StashAISlow { ping }')
        assert time.monotonic() - started < .4
        assert len(calls) == 1
        connect, read = calls[0]['timeout']
        assert connect <= .1 and read <= .15
    finally:
        client.close()


def test_response_arriving_at_deadline_is_closed_without_blocking_caller(monkeypatch):
    from stash_ai.stash_api import deadlines
    closed, release = Event(), Event()
    response = requests.Response()
    def close():
        closed.set()
        release.wait(1)
    response.close = close
    budgets = iter([1, 0])
    def remaining(_):
        value = next(budgets)
        if not value:
            raise ReadDeadlineExceeded('expired')
        return value
    monkeypatch.setattr(deadlines, 'remaining', remaining)
    try:
        with pytest.raises(ReadDeadlineExceeded):
            deadlines.call_with_deadline(lambda: response, 0)
        assert closed.wait(1)
        assert not release.is_set()
    finally:
        release.set()


def test_heartbeat_shows_inflight_write_and_stops_after_failure():
    client = transport()
    messages, entered, release = [], Event(), Event()
    report = RunReport(write=messages.append)
    report.heartbeat_seconds = .01
    client.report = report
    calls = []
    def write(*args, **kwargs):
        calls.append(1)
        entered.set()
        release.wait(1)
        raise requests.ConnectionError('ambiguous write')
    client.api.post = write
    failures = []
    def run():
        try:
            client.gql('mutation StashAITagDelete { tagDestroy }')
        except Exception as exc:
            failures.append(exc)
    thread = Thread(target=run)
    thread.start()
    assert entered.wait(1)
    time.sleep(.04)
    assert len(report.active_requests) == 1
    assert any('StashAITagDelete' in message and 'active=1' in message and 'write' in message for message in messages)
    assert all('secret-value' not in message for message in messages)
    release.set()
    thread.join(1)
    assert len(calls) == 1 and isinstance(failures[0], requests.ConnectionError)
    assert not report.active_requests
    count = len(messages)
    time.sleep(.025)
    assert len(messages) == count
    client.close()


def test_trickling_body_is_bounded_by_wall_clock_not_socket_inactivity():
    disconnected = Event()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '100')
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b'x')
                    self.wfile.flush()
                    time.sleep(.01)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = transport((.1, .15))
    client.url = f'http://127.0.0.1:{server.server_port}'
    started = time.monotonic()
    try:
        with pytest.raises(ReadDeadlineExceeded):
            client.get_media(client.url + '/sprite')
        assert time.monotonic() - started < .4
        assert disconnected.wait(1)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(1)
