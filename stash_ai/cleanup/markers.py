"""Delete excluded marker cores and extras without inferring replacements."""
from collections import Counter
from stash_ai.models import ModelSource
from stash_ai.publication import markers as marker_ops


def cleanup_scene_marker_cores(app, stash, scene: dict,
                               core_exclusions: frozenset[str], source: ModelSource,
                               report=None,
                               tag_exclusions: frozenset[str] = frozenset()) -> Counter:
    """Remove excluded cores and global extras; never infer replacement runs."""
    from stash_ai.vtt import format_time

    core_exclusions = core_exclusions | tag_exclusions
    excluded = [
        marker
        for marker in scene.get("scene_markers") or []
        if (
            marker_ops.owned_marker(marker, source)
            and marker_ops.marker_has_excluded_core(marker, core_exclusions, source)
        )
        or marker_ops.marker_has_excluded_core(marker, tag_exclusions, source)
    ]
    stats = Counter(
        markers_deleted=len(excluded),
        markers_core_cleaned=len(excluded),
    )
    for marker in excluded:
        if report:
            report(
                f"{'[DRY] Would delete' if stash.dry_run else 'Deleting'} "
                f"marker {marker['id']} on scene {scene['id']}: "
                f"core={marker['primary_tag']['name']!r}, "
                f"start={format_time(marker['seconds'])}"
            )
        if not stash.dry_run:
            stash.delete_marker(str(marker["id"]))

    excluded_ids = {str(marker["id"]) for marker in excluded}
    blocked_names = {
        source.prefix + tag
        for tag in tag_exclusions
    }
    for marker in scene.get("scene_markers") or []:
        if str(marker["id"]) in excluded_ids:
            continue
        names = (
            blocked_names
            if marker_ops.owned_marker(marker, source)
            else {source.prefix + tag for tag in tag_exclusions}
        )
        payload = marker_ops.marker_extra_cleanup_payload(
            marker,
            str(scene["id"]),
            names,
        )
        if payload is None:
            continue
        stats["markers_updated"] += 1
        stats["markers_extras_cleaned"] += 1
        if report:
            report(
                f"{'[DRY] ' if stash.dry_run else ''}"
                f"Remove excluded extras from marker {marker['id']} "
                f"on scene {scene['id']}."
            )
        if not stash.dry_run:
            stash.update_marker(payload)
    return stats
