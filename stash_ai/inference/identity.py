"""Model identity used by retained evidence and checkpoint signatures."""
from pathlib import Path
import hashlib

from stash_ai.models import ModelSettings
from stash_ai.signatures import digest
from stash_ai.backends.vocabulary import metadata_files, weight_files

PREDICTION_FORMAT = 4


def model_identity(app, config: ModelSettings, vocabulary, *, known_files=()):
    """Hash assets once per run; only retry missing files after model download.

    Callers must not reuse known_files across runs: same-size replacements with
    preserved timestamps still need a new content hash.
    """
    known = {path: checksum for path, checksum in known_files if checksum is not None}
    def stamp(path):
        path = Path(path).resolve()
        try:
            checksum = known.get(str(path))
            if checksum is None:
                with path.open('rb') as stream:
                    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            return [str(path), checksum]
        except FileNotFoundError:
            return [str(path), None]
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
    # Enabling or changing retention requests fresh inference to populate it;
    # publication-only threshold edits can still reuse the retained evidence.
    for name in ('scene_cache_floor', 'marker_cache_floor'):
        value = getattr(config, name)
        if value is not None:
            identity['retention'][name] = value
    return identity
