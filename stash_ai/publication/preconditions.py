"""Required source state and native range contract for current publication."""
from stash_ai.tools import MIGRATION_BRIDGE_HINT
from stash_ai.evidence.labels import AI
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG
from stash_ai.publication.markers import owned_marker


def ordinary_points(scene):
    return [marker for marker in scene.get('scene_markers', [])
            if owned_marker(marker, AI) and marker.get('end_seconds') is None
            and not any(tag['name'] == OCCURRENCE_TAG for tag in marker.get('tags', []))]


def require_native_ranges(scene):
    if ordinary_points(scene):
        raise RuntimeError(f"Scene {scene['id']} has AI markers without native ranges; "
                           + MIGRATION_BRIDGE_HINT)


def require_source_state(app, store, scene):
    from stash_ai.storage.recovery import publications
    if app.CREATE_MARKERS and any(owned_marker(marker, AI) for marker in scene.get('scene_markers', [])):
        known = (store.sources(scene['id'], 'markers') or
                 any(value is not None for value in store.snapshot(scene['id'], 'markers', borrow=True)) or
                 publications(store, scene['id'], 'markers'))
        if not known:
            raise RuntimeError('Merged markers exist without their source state. Restore the original --state-db before synchronization.')
        require_native_ranges(scene)
