"""Verified VTT/sprite inputs and shared media caches for model passes, RAM only."""
from __future__ import annotations

from collections import OrderedDict
import hashlib

from stash_ai.inference import sprites
from stash_ai.signatures import digest


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
    cue_identity = [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]
    signature = digest({"model": identity, "cues": cue_identity, "media": media.hashes})
    return cues, media, signature


class RunMedia:
    """One verified media snapshot for sequential model passes, RAM only."""
    def __init__(self, stash, cache, report=None):
        self.stash, self.cache, self.report = stash, cache, report
        self.token, self.hashes = object(), {}

    def get_media(self, url):
        key = ('media', self.token, url)
        saved = self.cache.get(key)
        if saved is not None:
            if self.report is not None:
                self.report.counts['media_cache_hits'] += 1
            return saved
        data = self.stash.get_media(url)
        checksum = hashlib.sha256(data).hexdigest()
        if url in self.hashes and checksum != self.hashes[url]:
            raise RuntimeError('Stash media changed during model passes; rerun to use one consistent input.')
        self.hashes[url] = checksum
        self.cache.put(key, data)
        if self.report is not None:
            self.report.counts['media_downloads'] += 1
            self.report.counts['media_download_bytes'] += len(data)
        return data


class RunInputs:
    """Share verified cue manifests without changing any inference signatures."""
    def __init__(self, stash, cache, report=None):
        self.cache, self.report = cache, report
        self.media = RunMedia(stash, cache, report)
        self.token, self.identities = object(), {}

    def inspect(self, app, paths, model_identity):
        key = digest({'paths': paths, 'sample_limit': app.MAX_FRAMES_PER_SCENE})
        cached = self.cache.get(('input', self.token, key))
        if cached is None:
            cues, media, signature = inspect_input(app, self.media, paths, model_identity)
            cue_identity = [[c.start, c.end, c.image, list(c.rect)] for c in cues]
            frozen = digest({'cues': cue_identity, 'media': media.hashes})
            if key in self.identities and frozen != self.identities[key]:
                raise RuntimeError('Stash VTT changed during model passes; rerun the analysis.')
            self.identities[key] = frozen
            self.cache.put(('input', self.token, key), (cues, dict(media.hashes), cue_identity))
            return cues, media, signature
        cues, hashes, cue_identity = cached
        media = VerifiedMedia(self.media, limit=0, expected=hashes)
        signature = digest({'model': model_identity, 'cues': cue_identity, 'media': hashes})
        if self.report is not None:
            self.report.counts['input_manifest_hits'] += 1
        return cues, media, signature
