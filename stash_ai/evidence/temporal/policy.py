"""Temporal configuration, frozen snapshots and publication floors."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class TemporalPolicy:
    enabled: bool = True
    shared_marker_frames: bool = True
    shared_marker_extras: bool = True
    short_marker_extras: bool = True
    extra_short_min_hits: int = 3
    shared_scene_tags: bool = True
    occurrence_markers: bool = True
    marker_continue_ratio: float = 0.85
    marker_max_weak_seconds: float = 15.0
    extra_min_hits: int = 2
    extra_min_fraction: float = 0.20
    consensus_min_models: int = 2
    scene_support_ratio: float = 0.85
    scene_min_hits: int = 2
    scene_repeat_hits: int = 4
    scene_repeat_run: int = 3
    scene_repeat_fraction: float = 0.10
    scene_strong_margin: float = 0.50

    @classmethod
    def from_snapshot(cls, values):
        # Old published markers may retain retired video options. Never use them.
        saved = {key: value for key, value in values.items() if not key.startswith("refine_")}
        # Cleanup must retain the algorithm used by older published snapshots.
        saved.setdefault("shared_marker_frames", False)
        saved.setdefault("shared_marker_extras", False)
        saved.setdefault("short_marker_extras", False)
        saved.setdefault("shared_scene_tags", False)
        saved.setdefault("occurrence_markers", False)
        policy = cls(**saved)
        policy.validate()
        return policy

    def validate(self):
        for name in ("enabled", "shared_marker_frames", "shared_marker_extras", "short_marker_extras", "shared_scene_tags", "occurrence_markers"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false.")
        for name in ("marker_continue_ratio", "scene_support_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        for name in ("extra_min_fraction", "scene_repeat_fraction", "scene_strong_margin"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in ("extra_min_hits", "extra_short_min_hits", "scene_min_hits", "scene_repeat_hits", "scene_repeat_run"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        if type(self.consensus_min_models) is not int or not 2 <= self.consensus_min_models <= 4:
            raise ValueError("consensus_min_models must be in 2..4.")
        value = self.marker_max_weak_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid marker_max_weak_seconds.")


def load_policy(app, path, enabled=True):
    if not enabled:
        return TemporalPolicy(enabled=False)
    data = app.load_unique_yaml(path, "Temporal policy")
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Temporal policy requires version: 1.")
    allowed = {f.name for f in fields(TemporalPolicy)}
    if any(isinstance(key, str) and key.startswith("refine_") for key in data):
        raise ValueError("Video-file analysis has been removed. Remove refine_* settings from the temporal policy; only Stash VTT/sprites are supported.")
    if set(data) - allowed - {"version"}:
        raise ValueError("Unknown temporal policy keys: " + ", ".join(sorted(set(data) - allowed - {"version"})))
    policy = TemporalPolicy(**{k: v for k, v in data.items() if k != "version"})
    policy.validate()
    return policy


def marker_floor(policy, threshold):
    return threshold * policy.marker_continue_ratio if policy.enabled else threshold


def build_options(app):
    return {"policy": asdict(app.TEMPORAL_POLICY),
        "marker_thresholds": dict(getattr(app, "MARKER_THRESHOLDS", {})),
        "settings": {name: getattr(app, name) for name in (
        "MARKER_MIN_HITS", "MARKER_MERGE_GAP_SECONDS", "MARKER_MAX_TAGS_PER_FRAME", "MAX_MARKERS_PER_SCENE")}}
