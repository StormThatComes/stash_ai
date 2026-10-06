"""Explicit per-run configuration, independent of CLI imports and model services."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from urllib.parse import urlparse

from stash_ai.evidence.temporal.policy import TemporalPolicy

INSTALL_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RuntimeSettings:
    """Own one invocation's settings; evidence workers use this data-only context."""

    INSTALL_ROOT: Path = INSTALL_ROOT
    DEFAULT_RESUME: bool = True
    CHECKPOINT_BATCHES: int = 8
    PREFETCH_BATCHES: int = 4
    RAM_CACHE_GB: float = 24.0
    RAM_BUDGET_GB: float = 32.0
    PREPARE_WORKERS: int = 4
    MERGE_STATE_FILE: Path = INSTALL_ROOT / "tagger_sources.sqlite3"
    TEMPORAL_POLICY: TemporalPolicy = field(default_factory=TemporalPolicy)
    TEMPORAL_POLICY_FILE: Path = INSTALL_ROOT / "temporal_policy.yml"
    STASH_URL: str = field(default_factory=lambda: os.getenv("STASH_URL", "http://localhost:9999"))
    STASH_API_KEY: str = field(default_factory=lambda: os.getenv("STASH_API_KEY", ""), repr=False)
    APPLY_SCENE_TAGS: bool = True
    MAX_TAGS: int = 0
    AGGREGATION: str = "topk_mean"
    TOP_K_FRAMES: int = 2
    CREATE_MARKERS: bool = True
    MARKER_MAX_TAGS_PER_FRAME: int = 0
    MARKER_MIN_HITS: int = 2
    MARKER_MERGE_GAP_SECONDS: float = 0.0
    MAX_MARKERS_PER_SCENE: int = 0
    MARKER_MATCH_TOLERANCE_SECONDS: float = 0.25
    CREATE_MISSING: bool = True
    DRY_RUN: bool = False
    MAX_FRAMES_PER_SCENE: int = 0
    SCENE_PAGE_SIZE: int = 100
    REQUEST_TIMEOUT: tuple[int, int] = (5, 600)
    HTTP_RETRIES: int = 3
    HIERARCHY_FILE: Path = INSTALL_ROOT / "tag_hierarchy.yml"
    MARKER_CORE_EXCLUSIONS_FILE: Path = INSTALL_ROOT / "marker_core_exclusions.yml"
    SCENE_TAG_EXCLUSIONS_FILE: Path = INSTALL_ROOT / "scene_tag_exclusions.yml"
    MODELS_CONFIG_FILE: Path = INSTALL_ROOT / "tagger_models.yml"
    JOYTAG_CLIP_MEAN: list[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    JOYTAG_CLIP_STD: list[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])

def validate_config(app, scene_work: bool = True) -> None:
    parsed = urlparse(app.STASH_URL)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("STASH_URL must be an http(s) base URL without credentials/query/fragment.")
    _ = parsed.port
    if scene_work and not app.APPLY_SCENE_TAGS and not app.CREATE_MARKERS:
        raise ValueError("Enable scene tags, markers, or both.")
    for name, value in (("TOP_K_FRAMES", app.TOP_K_FRAMES),
                        ("MARKER_MIN_HITS", app.MARKER_MIN_HITS), ("SCENE_PAGE_SIZE", app.SCENE_PAGE_SIZE)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("MAX_TAGS", app.MAX_TAGS), ("MAX_FRAMES_PER_SCENE", app.MAX_FRAMES_PER_SCENE),
                        ("MARKER_MAX_TAGS_PER_FRAME", app.MARKER_MAX_TAGS_PER_FRAME),
                        ("MAX_MARKERS_PER_SCENE", app.MAX_MARKERS_PER_SCENE)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0 (0 = unlimited).")
    for value in (app.MARKER_MERGE_GAP_SECONDS, app.MARKER_MATCH_TOLERANCE_SECONDS):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Marker time gaps/tolerances must be finite and nonnegative.")
    if app.AGGREGATION not in {"max", "topk_mean"}:
        raise ValueError("AGGREGATION must be max or topk_mean.")


EVIDENCE_SETTINGS = (
    "APPLY_SCENE_TAGS", "CREATE_MARKERS", "MAX_TAGS", "MARKER_MIN_HITS",
    "MARKER_MERGE_GAP_SECONDS", "MARKER_MAX_TAGS_PER_FRAME", "MAX_MARKERS_PER_SCENE",
    "TEMPORAL_POLICY",
)


def evidence_settings(app):
    """Capture every mutable setting used by evidence replay in spawned workers."""
    return {name: getattr(app, name) for name in EVIDENCE_SETTINGS}
