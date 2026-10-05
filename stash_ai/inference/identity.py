"""Model identity used by retained evidence and checkpoint signatures."""
from pathlib import Path

from stash_ai.models import ModelSettings
from stash_ai.signatures import digest
from stash_ai.backends.vocabulary import metadata_files, weight_files

PREDICTION_FORMAT = 2


def model_identity(app, config: ModelSettings, vocabulary):
    def stamp(path):
        path = Path(path).resolve()
        try:
            stat = path.stat()
            return [str(path), stat.st_size, stat.st_mtime_ns]
        except FileNotFoundError:
            return [str(path), None, None]
    files = [stamp(config.model_dir / name) for name in (*metadata_files(config.source), *weight_files(config))]
    loader = {}
    if config.source.key == "joytag":
        from stash_ai.backends.joytag import LOADER_REVISION, LOADER_SHA256
        loader = {"loader": {"revision": LOADER_REVISION, "sha256": LOADER_SHA256}}
    identity = {"format": PREDICTION_FORMAT, "source": config.source.key, "repo": config.source.repo,
            "revision": config.source.revision, "files": files, **loader,
            # Checkpoint score columns depend on label order, not just the set.
            "vocabulary": digest(vocabulary.labels), "batch_size": config.batch_size,
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k": app.TOP_K_FRAMES,
            "retention": {"temporal": True,
                          "marker_ratio": app.TEMPORAL_POLICY.marker_continue_ratio,
                          "scene_ratio": app.TEMPORAL_POLICY.scene_support_ratio},
            # Keep the old empty value so default-policy caches/checkpoints match.
            "frame_blocklist": []}
    # Omitted/null settings preserve legacy inference/checkpoint signatures.
    # Enabling or changing retention requests fresh inference to populate it;
    # publication-only threshold edits can still reuse the retained evidence.
    for name in ('scene_cache_floor', 'marker_cache_floor'):
        value = getattr(config, name)
        if value is not None:
            identity['retention'][name] = value
    return identity
