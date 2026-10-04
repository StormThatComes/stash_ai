"""Merged publication types."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MergeSyncPlan:
    """Concrete Stash mutations and durable snapshots for one scene."""

    snapshots: dict
    existing_scene_tag_ids: set[str]
    desired_scene_tag_ids: set[str]
    final_scene_tag_ids: set[str]
    migrated_scene_tag_ids: set[str]
    wanted_markers: list[tuple[dict, dict]] = field(default_factory=list)
    creates: list[dict] = field(default_factory=list)
    updates: list[dict] = field(default_factory=list)
    deletes: set[str] = field(default_factory=set)
    unchanged: int = 0

    @property
    def scene_changed(self):
        return self.final_scene_tag_ids != self.existing_scene_tag_ids
