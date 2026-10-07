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


@pytest.mark.parametrize('graphql', [False, True])
def test_repeated_header_timeouts_disconnect_and_do_not_accumulate_read_threads(graphql):
    from threading import enumerate as threads
    disconnected = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            event = Event()
            disconnected.append(event)
            try:
                self.wfile.write(b'HTTP/1.1 200 OK\r\nX-Slow: ')
                self.wfile.flush()
                for _ in range(1000):
                    self.wfile.write(b'x')
                    self.wfile.flush()
                    time.sleep(.005)
            except (BrokenPipeError, ConnectionResetError):
                event.set()
        do_POST = do_GET
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = transport((.1, .06), retries=0)
    client.url = f'http://127.0.0.1:{server.server_port}'
    before = sum(t.name == 'stash-deadline-read' for t in threads())
    try:
        for _ in range(8):
            with pytest.raises(ReadDeadlineExceeded):
                client.gql('query StashAISlow { ping }') if graphql else client.get_media(client.url + '/sprite')
            assert disconnected[-1].wait(1)
        end = time.monotonic() + 1
        while time.monotonic() < end and sum(t.name == 'stash-deadline-read' for t in threads()) > before:
            time.sleep(.01)
        assert sum(t.name == 'stash-deadline-read' for t in threads()) == before
        assert len(disconnected) == 8
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_uninterruptible_reads_are_bounded_and_capacity_wait_keeps_caller_deadline():
    from stash_ai.stash_api import deadlines
    release, started = Event(), []
    def blocked():
        started.append(True)
        release.wait(2)
    try:
        for _ in range(deadlines.MAX_ACTIVE_READS + 3):
            with pytest.raises(ReadDeadlineExceeded):
                deadlines.call_with_deadline(blocked, time.monotonic() + .02)
        assert len(started) == deadlines.MAX_ACTIVE_READS
    finally:
        release.set()


def test_expired_read_cannot_abort_a_connection_reused_by_another_read():
    from stash_ai.stash_api.read_connections import ReadControl
    closed = []
    class Socket:
        def shutdown(self, *args):
            closed.append(True)
    sock, owner = Socket(), SimpleNamespace()
    first, second = ReadControl(), ReadControl()
    first.track(sock, owner)
    second.track(sock, owner)
    first.cancel()
    assert not closed
    second.cancel()
    assert closed == [True]


@pytest.mark.parametrize('body', [False, True])
def test_https_read_deadline_disconnects_headers_and_body(tmp_path, body):
    import shutil
    import ssl
    import subprocess
    openssl = shutil.which('openssl')
    if openssl is None:
        pytest.skip('openssl is needed to create the local TLS certificate')
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    subprocess.run([openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1'],
                   check=True, capture_output=True)
    disconnected = Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                prefix = (b'HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n'
                          if body else b'HTTP/1.1 200 OK\r\nX-Slow: ')
                self.wfile.write(prefix)
                self.wfile.flush()
                for _ in range(1000):
                    self.wfile.write(b'x')
                    self.wfile.flush()
                    time.sleep(.005)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                disconnected.set()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = transport((.1, .15), retries=0)
    client.url = f'https://127.0.0.1:{server.server_port}'
    client.media.trust_env = False
    client.media.verify = str(cert)
    try:
        with pytest.raises(ReadDeadlineExceeded):
            client.get_media(client.url + '/sprite')
        assert disconnected.wait(1)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_prefetch_close_cancels_active_header_read_and_releases_producer(monkeypatch):
    from PIL import Image
    from stash_ai.inference import preparation, sprites
    from tests.helpers.runtime import runtime as app
    entered, disconnected = Event(), Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            entered.set()
            try:
                self.wfile.write(b'HTTP/1.1 200 OK\r\nX-Slow: ')
                self.wfile.flush()
                for _ in range(1000):
                    self.wfile.write(b'x')
                    self.wfile.flush()
                    time.sleep(.005)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = transport((.1, 30), retries=4)
    client.url = f'http://127.0.0.1:{server.server_port}'
    def frames(*args):
        with Image.new('RGB', (1, 1)) as image:
            yield object(), image
            client.get_media(client.url + '/sprite')
    monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    iterator = preparation.prepared_batches(app, SimpleNamespace(batch_size=1, prepare=lambda image: 1),
                                           client, {}, [], 1)
    try:
        assert next(iterator)[0] == [1]
        assert entered.wait(1)
        started = time.monotonic()
        iterator.close()
        assert time.monotonic() - started < .5
        assert disconnected.wait(1)
        from threading import enumerate as threads
        assert not any(t.name.startswith('prepare-frames') for t in threads())
    finally:
        iterator.close()
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(1)
