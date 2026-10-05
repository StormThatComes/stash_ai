"""Historical source ownership and cleanup require an explicit source."""
import inspect
from stash_ai.publication import markers as marker_ops
from stash_ai.cleanup.markers import cleanup_scene_marker_cores


def test_historical_marker_helpers_require_explicit_source():
    for helper in (marker_ops.owned_marker, marker_ops.marker_has_excluded_core,
                   cleanup_scene_marker_cores):
        assert inspect.signature(helper).parameters['source'].default is inspect.Parameter.empty
