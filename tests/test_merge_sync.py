"""One-scene merged reconciliation plans."""
from stash_ai.publication import merged as merge_sync
import stash_ai.publication.planning.types as merge_sync_types


def test_merge_sync_plan_reports_scene_change():
    unchanged = merge_sync_types.MergeSyncPlan({}, {"1"}, set(), {"1"}, set())
    changed = merge_sync_types.MergeSyncPlan({}, {"1"}, {"2"}, {"1", "2"}, set())

    assert not unchanged.scene_changed
    assert changed.scene_changed
