"""Model identity used by retained evidence and checkpoint signatures."""
from pathlib import Path
import hashlib
import os

from stash_ai.models import ModelSettings
from stash_ai.signatures import digest
from stash_ai.backends.vocabulary import metadata_files, weight_files

PREDICTION_FORMAT = 4


def _stamp(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def asset_stamps(config):
    """Transient change guards; never include file timestamps in provenance."""
    result = {}
    for name in (*metadata_files(config.source), *weight_files(config)):
        path = (config.model_dir / name).resolve()
        try:
            result[str(path)] = _stamp(path.stat())
        except FileNotFoundError:
            result[str(path)] = None
    return result


def require_assets_unchanged(expected, actual, *, allow_download=False):
    for path, stamp in expected.items():
        if actual.get(path) != stamp and not (allow_download and stamp is None):
            raise RuntimeError(f'Model asset changed during initialization: {path}; retry with stable model files.')


def model_identity(app, config: ModelSettings, vocabulary, *, known_files=(),
                   known_stamps=None, require_unchanged=False):
    """Reuse in-run hashes only with verified transient file-change guards.

    Every new run hashes content. ctime/inode guards also detect same-size
    replacements with restored mtime; they do not change retained signatures.
    """
    known = {path: checksum for path, checksum in known_files if checksum is not None}
    current = asset_stamps(config)
    if require_unchanged:
        require_assets_unchanged(known_stamps or {}, current)
    def stamp(path):
        path = Path(path).resolve()
        before = current[str(path)]
        if before is None:
            return [str(path), None]
        try:
            checksum = (known.get(str(path)) if known_stamps is not None
                        and known_stamps.get(str(path)) == before else None)
            if checksum is None:
                with path.open('rb') as stream:
                    if _stamp(os.fstat(stream.fileno())) != before:
                        raise RuntimeError(f'Model asset changed while hashing: {path}')
                    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if _stamp(os.fstat(stream.fileno())) != before:
                        raise RuntimeError(f'Model asset changed while hashing: {path}')
            if _stamp(path.stat()) != before:
                raise RuntimeError(f'Model asset changed while hashing: {path}')
            return [str(path), checksum]
        except FileNotFoundError as error:
            raise RuntimeError(f'Model asset disappeared while hashing: {path}') from error
    files = [stamp(config.model_dir / name) for name in (*metadata_files(config.source), *weight_files(config))]
    loader = {}
    if config.source.key == "joytag":
        from stash_ai.backends.joytag import LOADER_REVISION, LOADER_SHA256
        loader = {"loader": {"revision": LOADER_REVISION, "sha256": LOADER_SHA256},
                  "preprocessing": {"mean": list(app.JOYTAG_CLIP_MEAN), "std": list(app.JOYTAG_CLIP_STD)}}
    identity = {"format": PREDICTION_FORMAT, "source": config.source.key, "repo": config.source.repo,
            "revision": config.source.revision, "files": files, **loader,
            # Checkpoint score columns depend on label order, not just the set.
            "vocabulary": digest(vocabulary.labels), "batch_size": config.batch_size,
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k": app.TOP_K_FRAMES,
            "retention": {"marker_ratio": app.TEMPORAL_POLICY.marker_continue_ratio,
                          "scene_ratio": app.TEMPORAL_POLICY.scene_support_ratio}}
    # Preserve historical signatures. Verified state provenance separates
    # execution settings and checks actual retained coverage before reuse.
    for name in ('scene_cache_floor', 'marker_cache_floor'):
        value = getattr(config, name)
        if value is not None:
            identity['retention'][name] = value
    return identity
