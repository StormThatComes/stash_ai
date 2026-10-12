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
        self.server = self.thread = None
        self.guard = threading.Lock()
        self.path = '/' + uuid.uuid4().hex
        self.validator = None

    def _response(self, byte_range):
        url = self.url
        for _ in range(6):
            parts = urlsplit(url)
            if parts.username or parts.password or origin(url) != origin(self.client.url):
                raise ValueError('Video redirect left the configured Stash origin.')
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.remaining <= 0:
                raise TimeoutError('Portrait video byte/time budget exhausted.')
            response = self.client.media.get(url, headers={'Range': byte_range} if byte_range else {},
                    stream=True, allow_redirects=False, timeout=(min(5., remaining), remaining))
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                response.close()
                if not location:
                    raise ValueError('Video redirect has no destination.')
                url = urljoin(url, location)
                continue
            try:
                response.raise_for_status()
            except BaseException:
                response.close()
                raise
            if response.status_code not in (200, 206):
                response.close()
                raise ValueError('Unsupported video response.')
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
                    self.send_response(response.status_code)
                    for key in ('Content-Length', 'Content-Range', 'Accept-Ranges', 'Content-Type'):
                        if key in response.headers:
                            self.send_header(key, response.headers[key])
                    self.end_headers()
                    for chunk in response.iter_content(65536):
                        if len(chunk) > owner.remaining or time.monotonic() >= owner.deadline:
                            raise ValueError('Portrait video exceeds its byte/time budget.')
                        owner.remaining -= len(chunk)
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # FFmpeg stopped reading after obtaining its frame.
                except Exception as error:
                    owner.error = error
                    self.close_connection = True
                finally:
                    if response is not None:
                        response.close()
                    with owner.guard:
                        owner.active = None
        try:
            self.server = HTTPServer(('127.0.0.1', 0), Handler)
            self.thread = threading.Thread(target=self.server.serve_forever,
                                           kwargs={'poll_interval': .05}, name='recognition-video')
            self.thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def frame(self, seconds):
        from PIL import Image
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid portrait seek time.')
        timeout = self.deadline - time.monotonic()
        if timeout <= 0 or self.error:
            raise RuntimeError('Portrait source budget exhausted or source failed.')
        pixels = self.config.data['execution']['max_image_pixels']
        address = f'http://127.0.0.1:{self.server.server_port}{self.path}'
        # Downscale only when the source exceeds the output pixel budget.
        # There is no upscaling or copied thumbnail rectangle.
        scale = f"scale=w='min(iw,floor(sqrt({pixels}*iw/ih)/2)*2)':h=-2:flags=area,showinfo"
        command = [self.config.data['portraits']['ffmpeg'], '-hide_banner', '-nostdin', '-nostats',
                   '-loglevel', 'info', '-max_alloc', str(max(16777216, pixels * 8)),
                   '-protocol_whitelist', 'http,tcp',
                   '-format_whitelist', 'mov,matroska,webm,avi,mpegts',
                   '-rw_timeout', str(int(timeout * 1000000)),
                   '-threads', '1', '-copyts', '-ss', str(seconds), '-i', address,
                   '-map', '0:v:0', '-an', '-sn', '-dn', '-frames:v', '1', '-fps_mode', 'passthrough',
                   '-filter_threads', '1', '-vf', scale, '-threads', '1',
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
                    if size > limit:
                        raise ValueError('Portrait decoder output/log limit exceeded.')
                    result.append(chunk)
            except BaseException as error:
                failures.append(error)
                process.kill()
            finally:
                stream.close()
        workers = [threading.Thread(target=read, args=(process.stdout, output, pixels * 4 + 65536)),
                   threading.Thread(target=read, args=(process.stderr, errors, 131072))]
        for worker in workers:
            worker.start()
        try:
            process.wait(timeout=max(.01, self.deadline - time.monotonic()))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            for worker in workers:
                worker.join()
            self.process = None
        if failures:
            raise failures[0]
        if self.error:
            raise RuntimeError('Portrait video source failed (byte/time/validator/origin check).') from self.error
        if process.returncode or not output:
            log = b''.join(errors).decode('utf-8', errors='replace')
            diagnostic = re.findall(r'(?m)^(?:Unrecognized option|Option |Error )[^\r\n]*', log)
            hint = re.sub(r'https?://\S+', '[media URL]', '; '.join(diagnostic[-2:]))[:400]
            raise RuntimeError('Portrait decoding is unavailable for this source; no image was chosen.'
                               + (' Decoder: ' + hint if hint else ''))
        matches = re.findall(rb'pts_time:([-+0-9.eE]+)', b''.join(errors))
        if not matches:
            raise ValueError('Decoder did not report actual frame presentation time.')
        actual = float(matches[0])
        if not math.isfinite(actual) or actual < 0 or abs(actual - seconds) > 5:
            raise ValueError('Decoded timestamp is inconsistent with the requested portrait time.')
        with Image.open(BytesIO(b''.join(output))) as raw:
            if raw.width * raw.height > pixels:
                raise ValueError('Decoded portrait exceeds the pixel budget.')
            image = raw.convert('RGB')
        return image, {'requested_seconds': seconds, 'actual_seconds': actual,
                       'decoder': 'ffmpeg-copyts-showinfo-v1', 'pixels': [image.width, image.height],
                       'source_validator': list(self.validator or (None, None))}

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        with self.guard:
            if self.active is not None:
                self.active.close()
        if self.server is not None:
            if self.thread is not None and self.thread.is_alive():
                self.server.shutdown()
                self.thread.join()
            self.server.server_close()
        if self.owned_client:
            self.client.close()

    def __exit__(self, *args):
        self.close()
