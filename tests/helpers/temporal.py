"""Policies for focused threshold tests using the current temporal engine."""
from dataclasses import asdict
from pathlib import Path
import yaml
from stash_ai.evidence.temporal.policy import TemporalPolicy


def threshold_policy():
    return TemporalPolicy(shared_marker_frames=False, shared_marker_extras=False,
        short_marker_extras=False, shared_scene_tags=False, occurrence_markers=False,
        marker_continue_ratio=1.0, marker_max_weak_seconds=0,
        extra_min_hits=1, extra_min_fraction=0, scene_support_ratio=1.0,
        scene_min_hits=1, scene_strong_margin=1.0,
        scene_repeat_hits=1000, scene_repeat_run=1000, scene_repeat_fraction=1.0)


def policy_args(directory, extended=False):
    if extended:
        return ['--temporal-policy', str(Path(__file__).resolve().parents[2] / 'temporal_policy.yml')]
    path = Path(directory) / 'threshold_policy.yml'
    values = asdict(threshold_policy())
    path.write_text(yaml.safe_dump({'version': 1, **values}))
    return ['--temporal-policy', str(path)]


def install_threshold_policy(monkeypatch, runtime, directory):
    monkeypatch.setattr(runtime, 'TEMPORAL_POLICY', threshold_policy())
    monkeypatch.setattr(runtime, 'TEMPORAL_POLICY_FILE', Path(policy_args(directory)[1]))
