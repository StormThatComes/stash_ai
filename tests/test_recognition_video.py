from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest
import requests

from stash_ai.recognition.config import load
from stash_ai.recognition.video import VideoFrames


@pytest.fixture
def video(tmp_path):
    binary = shutil.which('ffmpeg')
    if not binary:
        pytest.skip('optional system FFmpeg is not installed')
    destination = tmp_path / 'tiny.mp4'
    subprocess.run([binary, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'testsrc=size=160x160:rate=10', '-t', '1', '-threads', '1',
                    '-c:v', 'mpeg4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)],
                    check=True, timeout=15, capture_output=True)
    return destination.read_bytes()


@contextmanager
def media_server(data):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append((self.path, self.headers.get('ApiKey'), self.headers.get('Range')))
            assert self.headers.get('ApiKey') == 'private-key'
            byte_range = self.headers.get('Range')
            start, end = 0, len(data) - 1
            if byte_range:
                values = byte_range.removeprefix('bytes=').split('-')
                start = int(values[0])
                if values[1]:
                    end = min(end, int(values[1]))
            if start > end:
                self.send_error(416)
                return
            body = data[start:end + 1]
            self.send_response(206 if byte_range else 200)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('ETag', '"test-video-v1"')
            if byte_range:
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    thread.start()
    session = requests.Session()
    session.trust_env = False
    session.headers['ApiKey'] = 'private-key'
    client = SimpleNamespace(url=f'http://127.0.0.1:{server.server_port}', media=session)
    try:
        yield client, calls
    finally:
        session.close()
        server.shutdown()
        thread.join()
        server.server_close()


def test_actual_frame_seek_uses_authenticated_range_bridge_and_cleans_workers(tmp_path, video, monkeypatch):
    config = load(root=tmp_path)
    commands = []
    original = subprocess.Popen
    def popen(command, **kwargs):
        commands.append(command)
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    with media_server(video) as (client, calls):
        scene = {'id': '1', 'paths': {'stream': '/stream?apikey=signed-secret'}}
        with VideoFrames(client, scene, config) as decoder:
            image, metadata = decoder.frame(.5)
            with image:
                assert image.size == (160, 160)
            assert metadata['requested_seconds'] == .5
            assert .5 <= metadata['actual_seconds'] <= .6
            assert metadata['source_validator'] == ['"test-video-v1"', None]
        assert not decoder.thread.is_alive() and decoder.process is None
        assert calls and all(item[1] == 'private-key' for item in calls)
        assert any(item[2] for item in calls)
    assert 'signed-secret' not in repr(commands) and 'private-key' not in repr(commands)
    assert '-vsync' not in commands[0] and '-fps_mode' in commands[0]


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='optional system FFmpeg is not installed')
def test_unsupported_video_and_byte_budget_leave_no_workers(tmp_path):
    config = load(root=tmp_path)
    with media_server(b'not a video') as (client, _):
        with VideoFrames(client, {'paths': {'stream': '/stream'}}, config) as decoder:
            with pytest.raises(RuntimeError, match='unavailable'):
                decoder.frame(0)
        assert not decoder.thread.is_alive()
    with media_server(b'x' * 100) as (client, _):
        with VideoFrames(client, {'paths': {'stream': '/stream'}}, config) as decoder:
            decoder.remaining = 10
            with pytest.raises(RuntimeError, match='source failed'):
                decoder.frame(0)
        assert not decoder.thread.is_alive()


def test_cross_origin_stream_refused_before_starting_resources(tmp_path):
    with pytest.raises(ValueError, match='same-origin'):
        VideoFrames(SimpleNamespace(url='http://stash'), {'paths': {'stream': 'http://other/video'}}, load(root=tmp_path))


def test_cancellation_kills_decoder_and_closes_owned_bridge(tmp_path, video, monkeypatch):
    config = load(root=tmp_path)
    original = subprocess.Popen
    def popen(command, **kwargs):
        process = original(command, **kwargs)
        wait = process.wait
        first = True
        def interrupted(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt()
            return wait(*args, **kwargs)
        process.wait = interrupted
        return process
    monkeypatch.setattr(subprocess, 'Popen', popen)
    with media_server(video) as (client, _):
        with pytest.raises(KeyboardInterrupt):
            with VideoFrames(client, {'paths': {'stream': '/stream'}}, config) as decoder:
                decoder.frame(.5)
        assert not decoder.thread.is_alive() and decoder.process is None
