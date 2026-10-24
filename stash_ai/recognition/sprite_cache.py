"""Thread-safe, content-addressed compressed/decoded sprite LRU for one run."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO
import threading

from stash_ai.inference.image_io import open_image
from stash_ai.runtime.cleanup import cleanup

_active = ContextVar('recognition_sprite_cache', default=None)


class SpriteCache:
    def __init__(self, limit):
        self.limit, self.size, self.peak = limit, 0, 0
        self.entries, self.lock = OrderedDict(), threading.RLock()
        self.hits = self.misses = self.evictions = 0
        self.decoders = {}
        self.generation = 0

    def _get(self, key):
        row = self.entries.pop(key, None)
        if row is None:
            self.misses += 1
            return None
        self.entries[key] = row
        self.hits += 1
        return row[0]

    def _put(self, key, value, size):
        size += 512
        if size > self.limit:
            return False
        removed = self.entries.pop(key, None)
        if removed is not None:
            self.size -= removed[1]
            if key[0] == 'pixels':
                removed[0].close()
        while self.entries and self.size + size > self.limit:
            old_key, old = self.entries.popitem(last=False)
            self.size -= old[1]
            self.evictions += 1
            if old_key[0] == 'pixels':
                old[0].close()
        self.entries[key] = value, size
        self.size += size
        self.peak = max(self.peak, self.size)
        return True

    def compressed(self, digest):
        with self.lock:
            return self._get(('bytes', digest))

    def remember(self, digest, data):
        with self.lock:
            self._put(('bytes', digest), data, len(data))

    def cached_crop(self, digest, cue, crop):
        with self.lock:
            sheet = self._get(('pixels', digest))
            return crop(sheet, cue) if sheet is not None else None

    @contextmanager
    def decoding(self, digest):
        with self.lock:
            lock, users = self.decoders.get(digest, (threading.Lock(), 0))
            self.decoders[digest] = lock, users + 1
        try:
            with lock:
                yield
        finally:
            with self.lock:
                _, users = self.decoders[digest]
                if users == 1:
                    del self.decoders[digest]
                else:
                    self.decoders[digest] = lock, users - 1

    def crop(self, digest, data, cue, crop, *, maximum=None):
        with self.decoding(digest):
            with self.lock:
                generation = self.generation
                sheet = self._get(('pixels', digest))
                if sheet is not None:
                    return crop(sheet, cue)
            # Different sprites may decode concurrently. Only one decoder for
            # a digest runs at once; decoded owners are published atomically.
            with open_image(BytesIO(data)) as raw:
                size = raw.width * raw.height * 3
                if size + 512 > self.limit or maximum is not None and size > maximum:
                    return crop(raw, cue)
                sheet = raw.convert('RGB')
            retained = False
            try:
                with self.lock:
                    if self.generation == generation:
                        retained = self._put(('pixels', digest), sheet, size)
                    return crop(sheet, cue)
            finally:
                if not retained:
                    sheet.close()

    def close(self):
        with self.lock:
            self.generation += 1
            sheets = [v[0] for k,v in self.entries.items() if k[0] == 'pixels']
            self.entries.clear()
            self.size = 0
            cleanup([sheet.close for sheet in sheets])


def current():
    return _active.get()


@contextmanager
def use(cache):
    token = _active.set(cache)
    try:
        yield cache
    finally:
        _active.reset(token)


@contextmanager
def scope(config):
    cache = SpriteCache(config.data['execution']['sprite_cache_bytes'])
    with use(cache):
        try:
            yield cache
        finally:
            from stash_ai.runtime.shutdown import release
            release(cache, cache.close)
