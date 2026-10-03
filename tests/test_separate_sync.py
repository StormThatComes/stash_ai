"""Model sources are explicit in separate-output reconciliation."""
from stash_ai.publication import markers as marker_ops
from stash_ai.publication import separate as separate_sync


def test_generic_sync_helpers_require_explicit_source():
    import inspect

    assert inspect.signature(marker_ops.owned_marker).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(marker_ops.marker_has_excluded_core).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.cleanup_scene_marker_cores).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(marker_ops.marker_payload).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.plan_marker_changes).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.sync_scene).parameters["source"].default is inspect.Parameter.empty
