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


def bounded_media(client, url, limit, timeout):
    from stash_ai.runtime.cleanup import closing
    from stash_ai.stash_api.deadlines import call_with_deadline, abort_response
    deadline = time.monotonic() + timeout

    def acquire():
        current = url
        for _ in range(6):
            parts = urlsplit(current)
            if parts.username or parts.password or origin(current) != origin(client.url):
                raise ValueError('Recognition media must use the configured Stash origin.')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Recognition media deadline exceeded.')
            response = call_with_deadline(lambda: client.media.get(current, stream=True, allow_redirects=False,
                                         timeout=(min(5., remaining), remaining)), deadline)
            with closing(response, name='close recognition media response'):
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location:
                        raise ValueError('Recognition media redirect has no destination.')
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()

                def body():
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if limit is not None and size > limit:
                            raise ValueError('Recognition media exceeds its byte budget.')
                        chunks.append(chunk)
                    return b''.join(chunks)
                return call_with_deadline(body, deadline, cancel=lambda: abort_response(response))
        raise ValueError('Too many recognition media redirects.')

    return call_with_deadline(acquire, deadline)


class SpriteInput:
    # A portrait source owns at most one additional decoded RGB sheet.
    # Oversized sheets keep the original crop-before-conversion path.
    portrait_sheet_bytes = 64 * 1048576

    def __init__(self, client, scene, config):
        self.client, self.config = client, config
        settings, paths = config.data['execution'], scene['paths']
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
        return bounded_media(self.client, url, None, self.config.data['execution']['timeout_seconds'])

    def _sheet(self, url):
        if url not in self.cache:
            self.cache[url] = self.read(url)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(url)
        return self.cache[url]

    def frames(self, start=0):
        sheets = OrderedDict()
        try:
            for index in range(start, len(self.cues)):
                cue = self.cues[index]
                if cue.image not in sheets:
                    data = self._sheet(cue.image)
                    if hashlib.sha256(data).hexdigest() != self.hashes[cue.image]:
                        raise ValueError('Sprite changed during recognition; rerun with stable inputs.')
                    with open_image(BytesIO(data)) as raw:
                        sheets[cue.image] = raw.convert('RGB')
                    if len(sheets) > 2:
                        _, removed = sheets.popitem(last=False)
                        removed.close()
                sheets.move_to_end(cue.image)
                sheet = sheets[cue.image]
                x, y, width, height = cue.rect
                if x + width > sheet.width or y + height > sheet.height:
                    raise ValueError('Recognition VTT rectangle exceeds sprite bounds.')
                with closing(sheet.crop((x, y, x + width, y + height)), name='close recognition source crop') as image:
                    yield index, cue, image
        finally:
            cleanup([*(sheet.close for sheet in sheets.values()), self.cache.clear])

    def thumbnail(self, index):
        """Read one full cue image, sharing the bounded verified sheet cache."""
        cue = self.cues[index]
        data = self._sheet(cue.image)
        cached = self._portrait_sheet
        if (cached is None or cached[0] != cue.image or cached[1] is not data
                or cached[3] != self.hashes[cue.image]):
            if hashlib.sha256(data).hexdigest() != self.hashes[cue.image]:
                raise ValueError('Sprite changed during portrait selection.')
            self._close_portrait_sheet()
            with open_image(BytesIO(data)) as raw:
                pixels = raw.convert('RGB') if raw.width * raw.height * 3 <= self.portrait_sheet_bytes else None
                self._portrait_sheet = cue.image, data, pixels, self.hashes[cue.image]
                if pixels is None:
                    return self._thumbnail_crop(raw, cue)
        sheet = self._portrait_sheet[2]
        if sheet is not None:
            return self._thumbnail_crop(sheet, cue)
        with open_image(BytesIO(data)) as raw:
            return self._thumbnail_crop(raw, cue)

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
        release(self, lambda: cleanup([self._close_portrait_sheet, self.cache.clear]))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        cleanup([self.close], primary=args[1])
