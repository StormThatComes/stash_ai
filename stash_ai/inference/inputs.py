"""Verified VTT/sprite inputs and shared media caches for model passes, RAM only."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from threading import Lock, local
from dataclasses import dataclass
from stash_ai.storage.segments import FrozenDict, FrozenList
import hashlib
import json

from stash_ai.inference import sprites
from stash_ai.signatures import digest
from .input_identity import cue_manifest, frame_signature, inference_signature


@dataclass(frozen=True, slots=True)
class VerifiedInput:
    cues: FrozenList
    manifest: FrozenList
    hashes: FrozenDict
    frame_signature: str
    cue_digest: str
    legacy_cue_sha256: str

    @classmethod
    def capture(cls, cues, hashes):
        manifest = FrozenList(cue_manifest(cues))
        hashes = FrozenDict(hashes)
        legacy = hashlib.sha256(json.dumps(manifest, ensure_ascii=False).encode()).hexdigest()
        return cls(FrozenList(cues), manifest, hashes, frame_signature(manifest, hashes), digest(manifest), legacy)

    def inference_signature(self, identity):
        return inference_signature(identity, self.manifest, self.hashes)


@dataclass(frozen=True, slots=True)
class InspectedInput:
    snapshot: VerifiedInput
    media: object
    signature: str

    def __iter__(self):
        return iter((self.snapshot.cues, self.media, self.signature))

    def __getitem__(self, index):
        return (self.snapshot.cues, self.media, self.signature)[index]

    def __len__(self):
        return 3


class VerifiedMedia:
    """Hash input bytes; retain at most two compressed sheets within a RAM limit.

    Never writes media to disk. A second fetch during inference verifies the
    preflight hash so a regenerated sprite cannot silently reuse a checkpoint.
    """
    def __init__(self, stash, limit=16 * 1024 * 1024, expected=None):
        self.stash, self.limit, self.size = stash, limit, 0
        self.cache, self.hashes = OrderedDict(), dict(expected or {})

    def get_media(self, url):
        if url in self.cache:
            self.cache.move_to_end(url)
            return self.cache[url]
        if isinstance(self.stash, RunMedia):
            data, checksum = self.stash.get_verified_media(url)
        else:
            data = self.stash.get_media(url)
            checksum = hashlib.sha256(data).hexdigest()
        if url in self.hashes and checksum != self.hashes[url]:
            raise RuntimeError("Input media changed during this scene; rerun to use the new input.")
        self.hashes[url] = checksum
        if len(data) <= self.limit:
            while self.cache and (len(self.cache) >= 2 or self.size + len(data) > self.limit):
                self.size -= len(self.cache.popitem(last=False)[1])
            self.cache[url] = data
            self.size += len(data)
        return data


def inspect_input(app, stash, paths, identity):
    media = VerifiedMedia(stash)
    cues, total = sprites.fetch_cues(app, media, paths)
    if not cues:
        raise ValueError("No VTT cues to analyze.")
    for url in dict.fromkeys(cue.image for cue in cues):
        media.get_media(url)
    snapshot = VerifiedInput.capture(cues, media.hashes)
    return InspectedInput(snapshot, media, snapshot.inference_signature(identity))


class RunMedia:
    """One verified media snapshot for sequential model passes, RAM only."""
    def __init__(self, stash, cache, report=None):
        self.stash, self.cache, self.report = stash, cache, report
        self.token, self.hashes = object(), {}
        self._lock, self._readers = Lock(), local()

    @contextmanager
    def read_with(self, reader):
        previous = getattr(self._readers, 'reader', None)
        self._readers.reader = reader
        try:
            yield
        finally:
            self._readers.reader = previous

    def get_media(self, url):
        key = ('media', self.token, url)
        saved = self.cache.get(key)
        if saved is not None:
            if self.report is not None:
                with self._lock:
                    self.report.counts['media_cache_hits'] += 1
            return saved
        reader = getattr(self._readers, 'reader', None) or self.stash
        data = reader.get_media(url)
        checksum = hashlib.sha256(data).hexdigest()
        with self._lock:
            if url in self.hashes and checksum != self.hashes[url]:
                raise RuntimeError('Stash media changed during model passes; rerun to use one consistent input.')
            self.hashes[url] = checksum
            self.cache.put(key, data)
            if self.report is not None:
                self.report.counts['media_downloads'] += 1
                self.report.counts['media_download_bytes'] += len(data)
        return data

    def get_verified_media(self, url):
        data = self.get_media(url)
        with self._lock:
            return data, self.hashes[url]


class RunInputs:
    """Share verified cue manifests without changing any inference signatures."""
    def __init__(self, stash, cache, report=None):
        self.cache, self.report = cache, report
        self.media = RunMedia(stash, cache, report)
        self.token, self.identities = object(), {}
        self._lock = Lock()

    def open_background_reader(self):
        factory = getattr(self.media.stash, 'new_media_reader', None)
        return factory() if factory is not None else None

    def read_with(self, reader):
        return self.media.read_with(reader)

    def inspect(self, app, paths, model_identity):
        key = digest({'paths': paths, 'sample_limit': app.MAX_FRAMES_PER_SCENE})
        cached = self.cache.get(('input', self.token, key))
        if cached is None:
            result = inspect_input(app, self.media, paths, model_identity)
            snapshot = result.snapshot
            with self._lock:
                if key in self.identities and snapshot.frame_signature != self.identities[key]:
                    raise RuntimeError('Stash VTT changed during model passes; rerun the analysis.')
                self.identities[key] = snapshot.frame_signature
                self.cache.put(('input', self.token, key), snapshot)
            return result
        media = VerifiedMedia(self.media, limit=0, expected=cached.hashes)
        signature = cached.inference_signature(model_identity)
        if self.report is not None:
            self.report.counts['input_manifest_hits'] += 1
        return InspectedInput(cached, media, signature)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        # Called only after all ordinary/stale-peer passes and producer joins.
        self.cache.discard_matching(lambda key: isinstance(key, tuple) and len(key) >= 2 and
            ((key[0] == 'input' and key[1] is self.token) or
             (key[0] == 'media' and key[1] is self.media.token)))
        self.identities.clear()
        with self.media._lock:
            self.media.hashes.clear()
