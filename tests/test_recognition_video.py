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
from stash_ai.recognition.video import VideoFrames, scene_timing


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
    assert commands[0][commands[0].index('-vf') + 1] == 'showinfo'
    assert '-max_alloc' not in commands[0]


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


@pytest.mark.parametrize('container,offset', [('mp4', 0), ('mp4', 12), ('mpegts', 0), ('mpegts', 12)])
def test_container_timestamps_are_normalized_without_extra_media_probe(tmp_path, monkeypatch, container, offset):
    binary = shutil.which('ffmpeg')
    if not binary:
        pytest.skip('optional system FFmpeg is not installed')
    destination = tmp_path / ('tiny.mp4' if container == 'mp4' else 'tiny.ts')
    command = [binary, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
               '-i', 'testsrc=size=160x160:rate=10', '-t', '2', '-threads', '1',
               '-c:v', 'mpeg4' if container == 'mp4' else 'mpeg2video', '-pix_fmt', 'yuv420p',
               '-output_ts_offset', str(offset)]
    # Every TS frame is independently seekable so this control checks timeline
    # coordinates rather than a sparse-GOP decoder's keyframe approximation.
    command += ['-movflags', '+faststart'] if container == 'mp4' else ['-g', '1', '-bf', '0', '-f', 'mpegts']
    subprocess.run([*command, str(destination)], check=True, timeout=15, capture_output=True)
    processes = []
    original = subprocess.Popen
    def popen(command, **kwargs):
        processes.append((command, original(command, **kwargs)))
        return processes[-1][1]
    monkeypatch.setattr(subprocess, 'Popen', popen)
    with media_server(destination.read_bytes()) as (client, calls):
        with VideoFrames(client, {'paths': {'stream': '/stream'}}, load(root=tmp_path)) as decoder:
            for requested in (.5, .7):
                image, metadata = decoder.frame(requested)
                with image:
                    assert image.size == (160, 160)
                assert requested - .000002 <= metadata['actual_seconds'] <= requested + .11
                assert metadata['source_pts_seconds'] - metadata['timeline_origin_seconds'] == pytest.approx(
                    metadata['actual_seconds'])
                assert metadata['time_coordinate_system'] == 'scene-relative'
                assert metadata['decoder'] == 'ffmpeg-copyts-showinfo-scene-time-v2'
                assert metadata['timeline_origin_seconds'] == pytest.approx(
                    offset if container == 'mp4' else offset + 1.5, abs=.11)
            assert calls and all(call[1] == 'private-key' for call in calls)
            assert any(call[2] for call in calls)
        assert not decoder.thread.is_alive() and decoder.process is None
    assert len(processes) == 2 and all(process.poll() == 0 for _, process in processes)
    assert all('-copyts' in command and '-ss' in command for command, _ in processes)


@pytest.mark.parametrize('log', [
    b'pts_time:12.5',
    b'Duration: 00:00:01.00, start: 12.000000, bitrate: 100 kb/s',
    b'Duration: 00:00:01.00, start: nan, bitrate: 100 kb/s\npts_time:12.5',
    b'Duration: 00:00:01.00, start: 1e999, bitrate: 100 kb/s\npts_time:12.5',
    b'Duration: 00:00:01.00, start: 12, bitrate: 100 kb/s\npts_time:1e999',
    b'Duration: 00:00:01.00, start: 12, bitrate: 100 kb/s\npts_time:11',
    b'Duration: 00:00:01.00, start: 12, bitrate: 100 kb/s\npts_time:18',
])
def test_missing_nonfinite_and_inconsistent_timestamps_are_rejected(log):
    with pytest.raises(ValueError):
        scene_timing(log, .5)


def test_negative_timeline_origin_and_only_rounding_at_zero_are_supported():
    timing = scene_timing(b'  Duration: N/A, start: -2.0, bitrate: N/A\npts_time:-1.5', .5)
    assert timing['actual_seconds'] == .5 and timing['timeline_origin_seconds'] == -2
    assert scene_timing(b'Duration: N/A, start: 0.000001, bitrate: N/A\npts_time:0', 0)[
        'actual_seconds'] == 0


def test_changed_timeline_origin_between_seeks_leaves_no_workers(tmp_path, video, monkeypatch):
    from stash_ai.recognition import video as module
    with media_server(video) as (client, _):
        with VideoFrames(client, {'paths': {'stream': '/stream'}}, load(root=tmp_path)) as decoder:
            image, _ = decoder.frame(.5)
            image.close()
            def changed(log, seconds):
                timing = scene_timing(log, seconds)
                timing['timeline_origin_seconds'] += 1
                return timing
            monkeypatch.setattr(module, 'scene_timing', changed)
            with pytest.raises(ValueError, match='origin changed'):
                decoder.frame(.7)
        assert not decoder.thread.is_alive() and decoder.process is None


from dataclasses import replace
from types import SimpleNamespace
from io import BytesIO
import json
import time
import sqlite3

import numpy as np
import pytest

from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_identity_merges import changed, fragments
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.clustering import cluster_scene
from stash_ai.models import ModelSettings, SOURCES


def stub_video_owner(tmp_path):
    from stash_ai.recognition.video import VideoFrames
    owner = VideoFrames(SimpleNamespace(url='http://stash/'), {'paths': {'stream': '/video'}}, setup(tmp_path))
    owner.server = SimpleNamespace(server_port=8000, server_close=lambda: None)
    return owner

@pytest.mark.parametrize('failed_start', [1, 2])
def test_decoder_reader_start_failure_reaps_owned_process(tmp_path, monkeypatch, failed_start):
    from stash_ai.recognition import video as module
    owner = stub_video_owner(tmp_path)
    calls = []
    def process(number):
        result = SimpleNamespace(stdout=BytesIO(), stderr=BytesIO(), returncode=None)
        result.poll = lambda: result.returncode
        result.kill = lambda: calls.append(('kill', number))
        def wait(**kw):
            calls.append(('wait', number))
            result.returncode = 0
        result.wait = wait
        return result
    first, second = process(1), process(2)
    processes = iter([first, second])
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: next(processes))
    class Thread:
        starts = 0
        def __init__(self, *a, **kw): pass
        def start(self):
            Thread.starts += 1
            if Thread.starts == failed_start: raise RuntimeError('reader startup')
        def join(self, **kw): calls.append('join')
    monkeypatch.setattr(module.threading, 'Thread', Thread)
    with pytest.raises(RuntimeError, match='reader startup'):
        owner.frame(0)
    assert owner.process is None and ('kill', 1) in calls and ('wait', 1) in calls
    assert calls.count('join') == failed_start - 1
    assert first.stderr.closed
    if failed_start == 1: assert first.stdout.closed
    with pytest.raises(RuntimeError, match='no image was chosen'): owner.frame(1)
    assert ('wait', 2) in calls and ('kill', 1) in calls and owner.process is None
    for value in (first, second): value.stdout.close(); value.stderr.close()
    owner.close()



from dataclasses import replace
from types import SimpleNamespace
from io import BytesIO
import json
import time
import sqlite3

import numpy as np
import pytest

from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_identity_merges import changed, fragments
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.clustering import cluster_scene
from stash_ai.models import ModelSettings, SOURCES


def stub_video_owner(tmp_path):
    from stash_ai.recognition.video import VideoFrames
    owner = VideoFrames(SimpleNamespace(url='http://stash/'), {'paths': {'stream': '/video'}}, setup(tmp_path))
    owner.server = SimpleNamespace(server_port=8000, server_close=lambda: None)
    return owner

def test_decoder_kill_failure_still_completes_other_teardown(tmp_path):
    owner = stub_video_owner(tmp_path)
    calls = []
    def kill(): raise OSError('kill failure')
    owner.process = SimpleNamespace(poll=lambda: None, kill=kill, wait=lambda **kw: calls.append('wait'))
    owner.cancellation = SimpleNamespace(cancel=lambda: calls.append('cancel'))
    owner.server = SimpleNamespace(server_close=lambda: calls.append('server_close'))
    owner.owned_client = True
    owner.client = SimpleNamespace(close=lambda: calls.append('client_close'))
    with pytest.raises(OSError, match='kill failure'): owner.close()
    assert calls == ['cancel', 'wait', 'server_close', 'client_close']



@pytest.mark.parametrize('fault', ['kill', 'wait', 'response'])
def test_video_exit_keeps_primary_and_attempts_all_teardown(tmp_path, monkeypatch, fault):
    from stash_ai.recognition import video as module
    owner = stub_video_owner(tmp_path)
    calls = []
    def action(name):
        def call(*args, **kwargs):
            calls.append(name)
            if name == fault:
                raise OSError(name + ' cleanup failure')
        return call
    owner.process = SimpleNamespace(poll=lambda: None, kill=action('kill'), wait=action('wait'))
    owner.active = object()
    monkeypatch.setattr(module, 'abort_response', lambda active: action('response')())
    owner.cancellation = SimpleNamespace(cancel=action('cancel'))
    owner.server = SimpleNamespace(server_close=action('server_close'))
    owner.owned_client = True
    owner.client = SimpleNamespace(close=action('client_close'))
    with pytest.raises(KeyboardInterrupt, match='primary interruption') as caught:
        try:
            raise KeyboardInterrupt('primary interruption')
        except KeyboardInterrupt as error:
            owner.__exit__(type(error), error, error.__traceback__)
            raise
    assert all(name in calls for name in ('cancel', 'response', 'kill', 'wait', 'server_close', 'client_close'))
    assert any('cleanup failure' in note for note in caught.value.__notes__)


def test_rejected_video_response_close_keeps_status_error(tmp_path):
    owner = stub_video_owner(tmp_path)
    def status(): raise ValueError('primary HTTP status')
    def close(): raise OSError('response close failure')
    response = SimpleNamespace(status_code=500, raise_for_status=status, close=close)
    owner.client.media = SimpleNamespace(get=lambda *a, **kw: response)
    with pytest.raises(ValueError, match='primary HTTP status') as caught:
        owner._response(None)
    assert any('response close failure' in note for note in caught.value.__notes__)
    owner.close()
