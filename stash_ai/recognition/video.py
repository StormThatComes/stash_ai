"""Bounded authenticated range bridge: FFmpeg sees only an owned loopback URL."""
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import math
import os
import re
import subprocess
import threading
import time
from urllib.parse import urljoin, urlsplit
import uuid

from stash_ai.stash_api.transport import absolute_stash_url, origin
from stash_ai.stash_api.deadlines import call_with_deadline, abort_response
from stash_ai.stash_api.read_connections import shutdown
from stash_ai.runtime.cancellation import Cancellation, use
from stash_ai.inference.image_io import open_image
from stash_ai.runtime.cleanup import cleanup


def scene_timing(log, requested):
    """Input -ss seeks relative to the demuxer start; copyts preserves raw PTS."""
    origin = re.search(rb'(?m)^\s*Duration:[^\r\n]*?\bstart:\s*([-+0-9.eE]+)(?:,|\s|$)', log)
    times = re.findall(rb'pts_time:([-+0-9.eE]+)', log)
    if origin is None or not times:
        raise ValueError('Decoder did not report the source timeline origin and frame presentation time.')
    start, raw = float(origin[1]), float(times[0])
    if not math.isfinite(start) or not math.isfinite(raw):
        raise ValueError('Decoder reported non-finite source timestamps.')
    actual = raw - start
    # Both log fields are rounded decimal seconds. Tolerate only microsecond
    # rounding at timeline zero; retain the existing large-seek rejection.
    if -0.000002 <= actual < 0:
        actual = 0.
    if actual < 0 or abs(actual - requested) > 5:
        raise ValueError('Decoded timestamp is inconsistent with the requested portrait time.')
    return dict(actual_seconds=actual, source_pts_seconds=raw, timeline_origin_seconds=start,
                time_coordinate_system='scene-relative')


class VideoFrames(AbstractContextManager):
    def __init__(self, client, scene, config):
        self.config = config
        self.url = absolute_stash_url(scene['paths'].get('stream') or '', client.url)
        if not scene['paths'].get('stream') or origin(self.url) != origin(client.url):
            raise ValueError('Selected scene has no same-origin video stream.')
        self.client = client.new_media_reader() if hasattr(client, 'new_media_reader') else client
        self.owned_client = self.client is not client
        self.deadline = time.monotonic() + config.data['portraits']['timeout_seconds']
        self.remaining = config.data['portraits']['max_video_bytes']
        self.error, self.active, self.process = None, None, None
        self.last_error = None
        self.server = self.thread = None
        self.guard = threading.Lock()
        self.cancellation = Cancellation()
        self.path = '/' + uuid.uuid4().hex
        self.validator = None
        self.timeline_origin = None

    def _failed_request(self, error):
        self.last_error = error
        if (not isinstance(error, Exception) or isinstance(error, ValueError)
                or self.remaining <= 0 or time.monotonic() >= self.deadline):
            self.error = error

    def _response(self, byte_range):
        url = self.url
        for _ in range(6):
            parts = urlsplit(url)
            if parts.username or parts.password or origin(url) != origin(self.client.url):
                raise ValueError('Video redirect left the configured Stash origin.')
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.remaining <= 0:
                raise TimeoutError('Portrait video byte/time budget exhausted.')
            response = call_with_deadline(lambda: self.client.media.get(url,
                    headers={'Accept-Encoding': 'identity', **({'Range': byte_range} if byte_range else {})}, stream=True, allow_redirects=False,
                    timeout=(min(5., remaining), remaining)), self.deadline)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                try:
                    if not location:
                        raise ValueError('Video redirect has no destination.')
                finally:
                    cleanup([('close portrait redirect response', response.close)])
                url = urljoin(url, location)
                continue
            try:
                response.raise_for_status()
                if response.status_code not in (200, 206):
                    raise ValueError('Unsupported video response.')
            except BaseException:
                cleanup([('close rejected portrait response', response.close)])
                raise
            return response
        raise ValueError('Too many portrait video redirects.')

    def __enter__(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != owner.path:
                    self.send_error(404)
                    return
                byte_range = self.headers.get('Range')
                if byte_range and not re.fullmatch(r'bytes=\d+-\d*', byte_range):
                    self.send_error(400)
                    return
                try:
                    with use(owner.cancellation):
                        call_with_deadline(lambda: self.forward(byte_range), owner.deadline,
                                           cancel=lambda: shutdown(self.connection))
                except BaseException as error:
                    owner._failed_request(error)
                    self.close_connection = True

            def forward(self, byte_range):
                response = None
                try:
                    response = owner._response(byte_range)
                    with owner.guard:
                        owner.active = response
                    # A stable validator is checked across seek requests when
                    # supplied; face identity is independently checked later.
                    validator = (response.headers.get('ETag'), response.headers.get('Last-Modified'))
                    if owner.validator is not None and validator != owner.validator:
                        raise ValueError('Video source changed during portrait extraction.')
                    owner.validator = validator
                    encoded = response.headers.get('Content-Encoding', '').lower() not in ('', 'identity')
                    if encoded and response.status_code == 206:
                        raise ValueError('Compressed ranged video response cannot preserve byte offsets.')
                    self.send_response(response.status_code)
                    for key in ('Content-Length', 'Content-Range', 'Accept-Ranges', 'Content-Type'):
                        if encoded and key != 'Content-Type':
                            continue  # iter_content yields decoded bytes; compressed lengths do not apply.
                        if key in response.headers:
                            self.send_header(key, response.headers[key])
                    if encoded:
                        self.send_header('Connection', 'close')
                        self.close_connection = True
                    self.end_headers()
                    for chunk in response.iter_content(65536):
                        if len(chunk) > owner.remaining or time.monotonic() >= owner.deadline:
                            raise ValueError('Portrait video exceeds its byte/time budget.')
                        owner.remaining -= len(chunk)
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # FFmpeg stopped reading after obtaining its frame.
                except Exception as error:
                    owner._failed_request(error)
                    self.close_connection = True
                finally:
                    try:
                        if response is not None:
                            cleanup([('close portrait response', response.close)], primary=owner.error)
                    finally:
                        with owner.guard:
                            owner.active = None
        try:
            self.server = HTTPServer(('127.0.0.1', 0), Handler)
            self.thread = threading.Thread(target=self.server.serve_forever,
                                           kwargs={'poll_interval': .05}, name='recognition-video')
            self.thread.start()
            return self
        except BaseException:
            cleanup([('close portrait video startup', self.close)])
            raise

    def frame(self, seconds):
        if self.process is not None:
            raise RuntimeError('Earlier portrait decoder has not been reaped.')
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid portrait seek time.')
        timeout = self.deadline - time.monotonic()
        if timeout <= 0 or self.error:
            raise RuntimeError('Portrait source budget exhausted or source failed.')
        self.last_error = None
        address = f'http://127.0.0.1:{self.server.server_port}{self.path}'
        command = [self.config.data['portraits']['ffmpeg'], '-hide_banner', '-nostdin', '-nostats',
                   '-loglevel', 'info',
                   '-protocol_whitelist', 'http,tcp',
                   '-format_whitelist', 'mov,matroska,webm,avi,mpegts,mpeg',
                   '-rw_timeout', str(int(timeout * 1000000)),
                   '-threads', '1', '-copyts', '-ss', str(seconds), '-i', address,
                   '-map', '0:v:0', '-an', '-sn', '-dn', '-frames:v', '1', '-fps_mode', 'passthrough',
                   '-filter_threads', '1', '-vf', 'showinfo', '-threads', '1',
                   '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1']
        environment = {k: v for k, v in os.environ.items()
                       if 'proxy' not in k.lower() and k not in ('STASH_API_KEY', 'STASH_URL')}
        output, errors, failures = [], [], []
        process = self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        def read(stream, result, limit):
            size = 0
            try:
                while chunk := stream.read(65536):
                    size += len(chunk)
                    if limit is not None and size > limit:
                        raise ValueError('Portrait decoder output/log limit exceeded.')
                    result.append(chunk)
            except BaseException as error:
                failures.append(error)
                cleanup([('stop failed portrait decoder', process.kill)], primary=error)
            finally:
                try:
                    stream.close()
                except BaseException as error:
                    failures.append(error)
        workers = []
        streams = (process.stdout, process.stderr)
        try:
            # Process ownership already exists before either thread allocation
            # or start can fail. Only successfully started readers may be joined.
            for stream, result, limit in ((process.stdout, output, None), (process.stderr, errors, 131072)):
                worker = threading.Thread(target=read, args=(stream, result, limit))
                worker.start()
                workers.append(worker)
            process.wait(timeout=max(.01, self.deadline - time.monotonic()))
        finally:
            def release_process():
                if process.poll() is not None:
                    self.process = None
            actions = [('stop portrait decoder', lambda: process.kill() if process.poll() is None else None),
                       ('reap portrait decoder', lambda: process.wait(timeout=5))]
            actions.extend(('join portrait pipe reader', lambda worker=worker: worker.join(timeout=5)) for worker in workers)
            actions.extend(('close portrait pipe', stream.close) for stream in streams)
            actions.append(('release portrait decoder', release_process))
            cleanup(actions)
        if failures:
            raise failures[0]
        if self.error:
            raise RuntimeError('Portrait video source failed (byte/time/validator/origin check).') from self.error
        if process.returncode or not output:
            if self.last_error is not None:
                from .diagnostics import message
                raise RuntimeError('Portrait video request failed; later candidates may retry: ' +
                                   message(self.last_error, getattr(self.client, 'api_key', ''))[:500]) from self.last_error
            log = b''.join(errors).decode('utf-8', errors='replace')
            diagnostic = re.findall(r'(?m)^(?:Unrecognized option|Option |Error )[^\r\n]*', log)
            hint = re.sub(r'https?://\S+', '[media URL]', '; '.join(diagnostic[-2:]))[:400]
            raise RuntimeError('Portrait decoding is unavailable for this source; no image was chosen.'
                               + (' Decoder: ' + hint if hint else ''))
        timing = scene_timing(b''.join(errors), seconds)
        start = timing['timeline_origin_seconds']
        if self.timeline_origin is not None and abs(self.timeline_origin - start) > .000002:
            raise ValueError('Video source timeline origin changed between portrait seeks.')
        self.timeline_origin = start
        with open_image(BytesIO(b''.join(output))) as raw:
            image = raw.convert('RGB')
        return image, {'requested_seconds': seconds, **timing,
                       'decoder': 'ffmpeg-copyts-showinfo-scene-time-v2', 'pixels': [image.width, image.height],
                       'source_validator': list(self.validator or (None, None))}

    def close(self):
        with self.guard:
            active, self.active = self.active, None
        process = self.process
        server, self.server = self.server, None
        thread = self.thread
        owned, self.owned_client = self.owned_client, False
        actions = [('cancel portrait bridge', self.cancellation.cancel)]
        if active is not None:
            actions.append(('abort portrait response', lambda: abort_response(active)))
        if process is not None:
            actions.extend([
                ('stop portrait decoder', lambda: process.kill() if process.poll() is None else None),
                ('reap portrait decoder', lambda: process.wait(timeout=5)),
            ])
        if server is not None:
            if thread is not None and thread.is_alive():
                actions.extend([('stop portrait server', server.shutdown),
                                ('join portrait server', thread.join)])
            actions.append(('close portrait server', server.server_close))
        if owned:
            actions.append(('close portrait media client', self.client.close))
        def release_process():
            if process is not None and process.poll() is not None:
                self.process = None
        actions.append(('release portrait decoder', release_process))
        cleanup(actions)

    def __exit__(self, exc_type, exc, traceback):
        cleanup([('close portrait video', self.close)], primary=exc)
