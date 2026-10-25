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
from stash_ai.inference.image_io import open_image
from stash_ai.runtime.cleanup import closing, cleanup


def safe_url(url):
    try:
        parts = urlsplit(url)
    except ValueError:
        # urllib can include credential-bearing netloc text in its exception.
        raise ValueError('Invalid recognition source URL.') from None
    return urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1], parts.path, '', ''))


def bounded_media(client, url, limit, timeout, *, cache=None):
    from stash_ai.runtime.cleanup import closing
    from stash_ai.stash_api.deadlines import call_with_deadline, abort_response
    deadline = time.monotonic() + timeout
    retained=cache.peek(url) if cache is not None else None

    def acquire():
        nonlocal retained
        current = url
        for _ in range(6):
            parts = urlsplit(current)
            if parts.username or parts.password or origin(current) != origin(client.url):
                raise ValueError('Recognition media must use the configured Stash origin.')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Recognition media deadline exceeded.')
            headers={'If-None-Match':retained[0]} if retained is not None and current==url else {}
            options=dict(headers=headers) if headers else {}
            response = call_with_deadline(lambda: client.media.get(current, stream=True, allow_redirects=False,
                                         timeout=(min(5., remaining), remaining),**options), deadline)
            with closing(response, name='close recognition media response'):
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location:
                        raise ValueError('Recognition media redirect has no destination.')
                    current = urljoin(current, location)
                    continue
                if response.status_code==304:
                    if retained is None or current!=url:
                        raise ValueError('Recognition media returned an unsolicited cache acknowledgement.')
                    if response.headers.get('ETag',retained[0])!=retained[0]:
                        cache.discard(url)
                        raise ValueError('Recognition media cache acknowledgement changed its validator.')
                    body=cache.get(url)
                    if body is None or body[0]!=retained[0]:
                        # Corruption/eviction between metadata and 304: retry a
                        # full read under the original request deadline.
                        retained=None
                        continue
                    data=body[1]
                    if 'no-store' in response.headers.get('Cache-Control','').lower():cache.discard(url)
                    if limit is not None and len(data)>limit:raise ValueError('Recognition media exceeds its byte budget.')
                    return data
                response.raise_for_status()

                def body():
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if limit is not None and size > limit:
                            raise ValueError('Recognition media exceeds its byte budget.')
                        chunks.append(chunk)
                    return b''.join(chunks)
                data=call_with_deadline(body, deadline, cancel=lambda: abort_response(response))
                if cache is not None and current==url:
                    etag=None if 'no-store' in response.headers.get('Cache-Control','').lower() else response.headers.get('ETag')
                    cache.put(url,etag,data)
                return data
        raise ValueError('Too many recognition media redirects.')

    return call_with_deadline(acquire, deadline)


class SpriteInput:
    # A portrait source owns at most one additional decoded RGB sheet.
    # Oversized sheets keep the original crop-before-conversion path.
    portrait_sheet_bytes = 64 * 1048576

    def __init__(self, client, scene, config):
        self.client, self.config = client, config
        from .media_cache import MediaCache
        self.media_cache=MediaCache(config.database,config.data['execution']['media_cache_bytes'],client)
        settings, paths = config.data['execution'], scene['paths']
        from .sprite_cache import current, SpriteCache
        shared = current()
        self._owns_sprites = shared is None
        self.sprites = shared or SpriteCache(settings['sprite_cache_bytes'])
        self.cache = OrderedDict()
        self._portrait_sheet = None
        if not paths.get('vtt'):
            raise ValueError('Scene has no VTT; generate Stash sprites first.')
        vtt = absolute_stash_url(paths['vtt'], client.url)
        text = self.read(vtt).decode('utf-8-sig')
        cues = parse_vtt_cues(text)
        single = len({cue.image for cue in cues}) == 1
        cues = [Cue(c.start, c.end, absolute_stash_url(paths['sprite'], client.url)
                    if single and paths.get('sprite') else urljoin(vtt, c.image), c.rect) for c in cues]
        total = len(cues)
        count = settings['max_cues']
        if count and total > count:
            from stash_ai.inference.sprites import evenly_sample
            cues = evenly_sample(cues, count)
        urls = list(dict.fromkeys(c.image for c in cues))
        self.hashes = {}
        for url in urls:
            data = self._sheet(url)
            self.hashes[url] = hashlib.sha256(data).hexdigest()
        self.cues = cues
        self.coverage = {'total_cues': total, 'selected_cues': len(cues), 'complete_sampling': total == len(cues)}
        self.manifest = {'cues': [[c.start, c.end, safe_url(c.image), list(c.rect)] for c in cues],
                         'sheets': [[safe_url(url), self.hashes[url]] for url in urls]}
        if len({safe_url(url) for url in urls}) != len(urls):
            # Bind each ambiguous sanitized URL to its actual pixels, without
            # retaining credentials or invalidating ordinary historical scans.
            self.manifest['cue_sheets'] = [self.hashes[cue.image] for cue in cues]
        self.fingerprint = digest(self.manifest)

    def read(self, url):
        return bounded_media(self.client, url, None, self.config.data['execution']['timeout_seconds'],cache=self.media_cache)

    def _sheet(self, url):
        expected = getattr(self, 'hashes', {}).get(url)
        override = self.cache.get(url)
        data = override if isinstance(override, bytes) else self.sprites.compressed(expected) if expected is not None else None
        if isinstance(override, bytes) and hashlib.sha256(override).hexdigest() != expected:
            raise ValueError('Sprite changed during portrait selection.')
        if data is None:
            data = self.read(url)
            actual = hashlib.sha256(data).hexdigest()
            if expected is not None and actual != expected:
                raise ValueError('Sprite changed during recognition; rerun with stable inputs.')
            self.sprites.remember(actual, data)
        self.cache[url] = expected or actual  # Only tiny source-local ownership markers.
        if len(self.cache) > 2:
            self.cache.popitem(last=False)
        return data

    def frames(self, start=0):
        try:
            for index in range(start, len(self.cues)):
                with closing(self._cue_image(index), name='close recognition source crop') as image:
                    yield index, self.cues[index], image
        finally:
            # Shared immutable sprites can serve portraits later in this run.
            cleanup([self.cache.clear])

    def frames_indices(self, indices):
        try:
            for index in indices:
                with closing(self._cue_image(index),name='close selected recognition source crop') as image:
                    yield index,self.cues[index],image
        finally:self.cache.clear()

    @property
    def cached_bytes(self):
        return [data for digest in self.cache.values() if isinstance(digest,str)
                if (data := self.sprites.compressed(digest)) is not None]

    def thumbnail(self, index):
        return self._cue_image(index, maximum=self.portrait_sheet_bytes)

    def _cue_image(self, index, *, maximum=None):
        cue = self.cues[index]
        if not isinstance(self.cache.get(cue.image), bytes):
            cached = self.sprites.cached_crop(self.hashes[cue.image], cue, self._thumbnail_crop)
            if cached is not None:
                return cached
        data = self._sheet(cue.image)
        self._portrait_sheet = cue.image, None, None, self.hashes[cue.image]
        return self.sprites.crop(self.hashes[cue.image], data, cue, self._thumbnail_crop, maximum=maximum)

    @staticmethod
    def _thumbnail_crop(sheet, cue):
        x, y, width, height = cue.rect
        if x + width > sheet.width or y + height > sheet.height:
            raise ValueError('Recognition VTT rectangle exceeds sprite bounds.')
        with sheet.crop((x, y, x + width, y + height)) as crop:
            return crop.convert('RGB')

    def _close_portrait_sheet(self):
        cached, self._portrait_sheet = self._portrait_sheet, None
        if cached is not None and cached[2] is not None:
            cached[2].close()

    def close(self):
        from stash_ai.runtime.shutdown import release
        release(self, lambda: cleanup([self._close_portrait_sheet, self.cache.clear,
            self.sprites.close if self._owns_sprites else lambda: None]))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        cleanup([self.close], primary=args[1])
