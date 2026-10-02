"""Shared scene, prediction, marker and hierarchy value types."""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_HIERARCHY_DESCRIPTION = "[AI hierarchy v1]"


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    image: str
    rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class FramePrediction:
    cue: Cue
    tags: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MarkerSpec:
    tag: str
    start: float
    end: float
    score: float
    hits: int
    extra_tags: tuple[str, ...] = ()
    extra_evidence: tuple[tuple[str, int, float], ...] = ()
    support_sources: tuple[str, ...] = ()
    strong_hits: int = 0
    extra_support: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TagHierarchy:
    """Leaf keys are original model labels; category keys are Stash tag names."""
    leaf_parents: dict[str, tuple[str, ...]]
    category_parents: dict[str, tuple[str, ...]]
    category_prefix: str
    description_prefix: str = DEFAULT_HIERARCHY_DESCRIPTION

    def is_managed_category(self, tag: dict) -> bool:
        return (tag.get("name", "").startswith(self.category_prefix)
                and (tag.get("description") or "").startswith(self.description_prefix))
