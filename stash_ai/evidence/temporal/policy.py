"""Temporal configuration, frozen snapshots and publication floors."""
from dataclasses import asdict, dataclass, fields
import math
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps


@dataclass(frozen=True)
class TemporalPolicy:
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
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(cls)}:
            raise ValueError('Incomplete temporal snapshot policy; run --db-update.')
        policy = cls(**values)
        policy.validate()
        return policy

    def validate(self):
        for name in ("shared_marker_frames", "shared_marker_extras", "short_marker_extras", "shared_scene_tags", "occurrence_markers"):
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


def load_policy(path):
    from stash_ai.config import load_unique_yaml
    data = load_unique_yaml(path, "Temporal policy")
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
    return threshold * policy.marker_continue_ratio


def build_options(app, records=()):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.storage.thresholds import effective
    from stash_ai.runtime.dependencies import capture
    return {"policy": asdict(app.TEMPORAL_POLICY),
        "marker_thresholds": {marker_segment_signature(record): effective(record) for record in dict(records).values()},
        "settings": capture(app, 'marker_snapshot')}


def frozen_policy(options):
    """Replay only the current temporal engine with its saved settings."""
    return _cached('policy', options['policy'], _POLICY_FIELDS, TemporalPolicy.from_snapshot)


# Reuse validated immutable policy/settings values only within one operation.
# Include types in keys: bool/int/float equality must never bypass type checks.

_VALIDATED = ContextVar('validated_temporal_options', default=None)
_POLICY_FIELDS = tuple(field.name for field in fields(TemporalPolicy))
_MARKER_FIELDS = ('MARKER_MIN_HITS', 'MARKER_MAX_TAGS_PER_FRAME', 'MARKER_MERGE_GAP_SECONDS')


def _key(values, names):
    if not isinstance(values, dict) or set(values) != set(names):
        return None
    key = tuple((type(values[name]), values[name]) for name in names)
    try:
        hash(key)
    except TypeError:
        return None
    return key


def _cached(kind, values, names, validate):
    cache, key = _VALIDATED.get(), _key(values, names)
    key = (kind, key) if key is not None else None
    if cache is not None and key is not None and key in cache:
        return cache[key]
    result = validate(values)
    if cache is not None and key is not None and len(cache) < 128:
        cache[key] = result
    return result


@contextmanager
def validation_scope():
    if _VALIDATED.get() is not None:
        yield
        return
    token = _VALIDATED.set({})
    try:
        yield
    finally:
        _VALIDATED.reset(token)


def validated_operation(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        with validation_scope():
            return operation(*args, **kwargs)
    return wrapped


def validate_marker_settings(settings):
    def validate(values):
        if not isinstance(values, dict) or set(values) != set(_MARKER_FIELDS):
            raise ValueError('Incomplete frozen marker settings; run --db-update.')
        for name in _MARKER_FIELDS[:2]:
            if type(values[name]) is not int or values[name] < (1 if name == 'MARKER_MIN_HITS' else 0):
                raise ValueError(f'Invalid frozen marker setting {name}.')
        gap = values['MARKER_MERGE_GAP_SECONDS']
        if type(gap) not in (int, float) or not 0 <= gap < float('inf'):
            raise ValueError('Invalid frozen marker setting MARKER_MERGE_GAP_SECONDS.')
        return tuple((name, values[name]) for name in _MARKER_FIELDS)
    return _cached('marker_settings', settings, _MARKER_FIELDS, validate)
