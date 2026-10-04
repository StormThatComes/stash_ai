"""Model sources are explicit in separate-output reconciliation."""
from stash_ai.publication import markers as marker_ops
from stash_ai.publication import separate as separate_sync
import stash_ai.cleanup.markers as separate_sync_markers
import stash_ai.publication.planning.separate_markers as separate_sync_separate_markers


def test_generic_sync_helpers_require_explicit_source():
    import inspect

    assert inspect.signature(marker_ops.owned_marker).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(marker_ops.marker_has_excluded_core).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync_markers.cleanup_scene_marker_cores).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(marker_ops.marker_payload).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync_separate_markers.plan_marker_changes).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.sync_scene).parameters["source"].default is inspect.Parameter.empty
