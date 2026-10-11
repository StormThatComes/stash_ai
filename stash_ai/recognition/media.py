"""Bounded same-origin media input; credentials never enter stored manifests."""
from collections import OrderedDict
from io import BytesIO
import hashlib
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

from stash_ai.vtt import parse_vtt_cues
from stash_ai.domain import Cue
from stash_ai.stash_api.transport import origin, absolute_stash_url
from .config import digest


def safe_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def bounded_media(client, url, limit, timeout):
    deadline = time.monotonic() + timeout
    for _ in range(6):
        parts = urlsplit(url)
        if parts.username or parts.password or origin(url) != origin(client.url):
            raise ValueError('Recognition media must use the configured Stash origin.')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Recognition media deadline exceeded.')
        with client.media.get(url, stream=True, allow_redirects=False,
                              timeout=(min(5., remaining), remaining)) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if not location:
                    raise ValueError('Recognition media redirect has no destination.')
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            parts, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > limit or time.monotonic() >= deadline:
                    raise ValueError('Recognition media exceeds its byte/time budget.')
                parts.append(chunk)
            return b''.join(parts)
    raise ValueError('Too many recognition media redirects.')


class SpriteInput:
    def __init__(self, client, scene, config):
        self.client, self.config = client, config
        settings, paths = config.data['execution'], scene['paths']
        self.cache = OrderedDict()
        if not paths.get('vtt'):
            raise ValueError('Scene has no VTT; generate Stash sprites first.')
        vtt = absolute_stash_url(paths['vtt'], client.url)
        text = self.read(vtt, settings['max_vtt_bytes']).decode('utf-8-sig')
        cues = parse_vtt_cues(text)
        if len(cues) > 10000:
            raise ValueError('Recognition VTT exceeds the supported cue limit.')
        single = len({cue.image for cue in cues}) == 1
        cues = [Cue(c.start, c.end, absolute_stash_url(paths['sprite'], client.url)
                    if single and paths.get('sprite') else urljoin(vtt, c.image), c.rect) for c in cues]
        total = len(cues)
        count = settings['max_cues']
        if count and total > count:
            from stash_ai.inference.sprites import evenly_sample
            cues = evenly_sample(cues, count)
        urls = list(dict.fromkeys(c.image for c in cues))
        if len(urls) > settings['max_sheets']:
            raise ValueError('Recognition VTT exceeds the sprite-sheet limit.')
        self.hashes = {}
        for url in urls:
            data = self._sheet(url)
            self.hashes[url] = hashlib.sha256(data).hexdigest()
        self.cues = cues
        self.coverage = {'total_cues': total, 'selected_cues': len(cues), 'complete_sampling': total == len(cues)}
        self.manifest = {'cues': [[c.start, c.end, safe_url(c.image), list(c.rect)] for c in cues],
                         'sheets': [[safe_url(url), self.hashes[url]] for url in urls]}
        self.fingerprint = digest(self.manifest)

    def read(self, url, limit):
        return bounded_media(self.client, url, limit, self.config.data['execution']['timeout_seconds'])

    def _sheet(self, url):
        if url not in self.cache:
            self.cache[url] = self.read(url, self.config.data['execution']['max_media_bytes'])
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(url)
        return self.cache[url]

    def frames(self, start=0):
        from PIL import Image
        sheets = OrderedDict()
        try:
            for index in range(start, len(self.cues)):
                cue = self.cues[index]
                if cue.image not in sheets:
                    data = self._sheet(cue.image)
                    if hashlib.sha256(data).hexdigest() != self.hashes[cue.image]:
                        raise ValueError('Sprite changed during recognition; rerun with stable inputs.')
                    with Image.open(BytesIO(data)) as raw:
                        if raw.width * raw.height > self.config.data['execution']['max_image_pixels']:
                            raise ValueError('Sprite exceeds recognition decoded-pixel limit.')
                        sheets[cue.image] = raw.convert('RGB')
                    if len(sheets) > 2:
                        _, removed = sheets.popitem(last=False)
                        removed.close()
                sheets.move_to_end(cue.image)
                sheet = sheets[cue.image]
                x, y, width, height = cue.rect
                if x + width > sheet.width or y + height > sheet.height:
                    raise ValueError('Recognition VTT rectangle exceeds sprite bounds.')
                with sheet.crop((x, y, x + width, y + height)) as image:
                    yield index, cue, image
        finally:
            for sheet in sheets.values():
                sheet.close()
            self.cache.clear()

    def close(self):
        self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
