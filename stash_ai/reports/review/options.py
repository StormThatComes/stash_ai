"""Review thresholds and documented signal meanings."""
from dataclasses import dataclass


REASONS = {
    "weak_or_borderline": "Best retained score is at most the applicable threshold plus the review margin.",
    "isolated_peak": "A scene tag qualifies through a strong peak with only one high sample.",
    "model_disagreement": "A compatible model explicitly retained this label but it does not qualify under that model's own scene rule.",
    "long_marker": "The merged marker range reaches the configured duration.",
}


@dataclass(frozen=True)
class ReviewOptions:
    margin: float = .05
    long_marker_seconds: float = 300.0
    examples: int = 3
