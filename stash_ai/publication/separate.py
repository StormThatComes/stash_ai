"""Separate-output reconciliation with explicit runtime settings and model sources."""
from __future__ import annotations

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


def plan_marker_changes(app, existing: list[dict], wanted: list[dict],
                        core_exclusions: frozenset[str], source: ModelSource
                        ) -> tuple[list[dict], list[dict], list[str], int]:
    available = {
        str(marker["id"]): marker
        for marker in existing
        if marker_ops.owned_marker(marker, source)
    }
    creates, updates = [], []
    unchanged = 0
    for payload in wanted:
        matches = [
            marker
            for marker in available.values()
            if str(marker["primary_tag"]["id"]) == payload["primary_tag_id"]
            and abs(marker["seconds"] - payload["seconds"])
            <= app.MARKER_MATCH_TOLERANCE_SECONDS
        ]
        if not matches:
            creates.append(payload)
            continue
        current = min(
            matches,
            key=lambda marker: (
                abs(marker["seconds"] - payload["seconds"]),
                str(marker["id"]),
            ),
        )
        marker_id = str(current["id"])
        del available[marker_id]
        if marker_ops.marker_differs(current, payload):
            updates.append({**payload, "id": marker_id})
        else:
            unchanged += 1

    # The exclusion policy always wins over retaining obsolete generated runs.
    deletes = sorted(
        marker_id
        for marker_id, marker in available.items()
        if app.REPLACE_MARKERS
        or marker_ops.marker_has_excluded_core(marker, core_exclusions, source)
    )
    return creates, updates, deletes, unchanged


def _resolved_names(app, resolver, source, scene_predictions, specs, scene_exclusions):
    scene_names = (
        {
            source.prefix + tag
            for tag, _ in scene_predictions
            if tag not in scene_exclusions
        }
        if app.APPLY_SCENE_TAGS
        else set()
    )
    names = set(scene_names)
    if app.CREATE_MARKERS and specs:
        names.add(source.owner_tag)
        for spec in specs:
            names.add(source.prefix + spec.tag)
            names.update(source.prefix + tag for tag in spec.extra_tags)
    ids = {name: resolver.resolve(name) for name in sorted(names)}
    return scene_names, ids


def _marker_plan(app, stash, source, scene_id, existing_markers, specs, ids,
                 core_exclusions, scene_exclusions):
    wanted = (
        [
            marker_ops.marker_payload(app, spec, scene_id, ids, source)
            for spec in specs
            if ids.get(source.prefix + spec.tag)
        ]
        if app.CREATE_MARKERS
        else []
    )
    plan = (
        plan_marker_changes(
            app,
            existing_markers,
            wanted,
            core_exclusions,
            source,
        )
        if app.CREATE_MARKERS
        else ([], [], [], 0)
    )
    creates, updates, deletes, unchanged = plan

    if app.CREATE_MARKERS:
        deletes = sorted(
            set(deletes)
            | {
                str(marker["id"])
                for marker in existing_markers
                if marker_ops.marker_has_excluded_core(marker, scene_exclusions, source)
            }
        )

    changed_ids = set(deletes) | {payload["id"] for payload in updates}
    wanted_keys = {
        (payload["primary_tag_id"], payload["seconds"])
        for payload in wanted
    }
    extra_updates = []
    if app.CREATE_MARKERS and scene_exclusions:
        blocked_names = {source.prefix + tag for tag in scene_exclusions}
        for marker in existing_markers:
            if str(marker["id"]) in changed_ids:
                continue
            if (
                str(marker["primary_tag"]["id"]),
                marker["seconds"],
            ) in wanted_keys:
                continue
            payload = marker_ops.marker_extra_cleanup_payload(
                marker,
                scene_id,
                blocked_names,
            )
            if payload is not None:
                extra_updates.append(payload)
        updates.extend(extra_updates)

    return wanted, creates, updates, deletes, unchanged, extra_updates


def _scene_tag_plan(app, source, existing_tags, scene_names, ids, scene_exclusions):
    existing_ids = {str(tag["id"]) for tag in existing_tags}
    preserved = {
        str(tag["id"])
        for tag in existing_tags
        if not (
            tag["name"].startswith(source.prefix)
            and (
                app.REPLACE_SCENE_TAGS
                or tag["name"][len(source.prefix):] in scene_exclusions
            )
        )
    }
    final_ids = preserved | {
        ids[name]
        for name in scene_names
        if ids[name] is not None
    }
    scene_changed = app.APPLY_SCENE_TAGS and final_ids != existing_ids
    cleaned = (
        sum(
            tag["name"].startswith(source.prefix)
            and tag["name"][len(source.prefix):] in scene_exclusions
            for tag in existing_tags
        )
        if app.APPLY_SCENE_TAGS
        else 0
    )
    return final_ids, scene_changed, cleaned


def _sync_statistics(existing_markers, source, core_exclusions, wanted,
                     creates, updates, deletes, unchanged, extra_updates, spec_count):
    excluded_ids = {
        str(marker["id"])
        for marker in existing_markers
        if marker_ops.owned_marker(marker, source)
        and marker_ops.marker_has_excluded_core(marker, core_exclusions, source)
    }
    return Counter(
        markers_created=len(creates),
        markers_updated=len(updates),
        markers_deleted=len(deletes),
        markers_unchanged=unchanged,
        markers_skipped=spec_count - len(wanted),
        markers_extras_cleaned=len(extra_updates),
        markers_core_cleaned=len(excluded_ids.intersection(deletes)),
    )


def _publish_plan(stash, scene_id, creates, updates, deletes,
                  scene_changed, final_ids):
    if stash.dry_run:
        return
    # No destructive cleanup until all new/replacement marker writes succeed.
    for payload in creates:
        stash.create_marker(payload)
    for payload in updates:
        stash.update_marker(payload)
    if scene_changed:
        stash.update_scene_tags(scene_id, sorted(final_ids))
    for marker_id in deletes:
        stash.delete_marker(marker_id)


def sync_scene(app, stash, resolver, scene: dict,
               scene_predictions: list[tuple[str, float]], specs: list,
               core_exclusions: frozenset[str], source: ModelSource,
               scene_exclusions: frozenset[str] = frozenset()) -> Counter:
    if resolver.source != source:
        raise ValueError(
            "Resolver/source mismatch; refusing to mix model namespaces."
        )

    from stash_ai.exclusions import filter_spec

    spec_count = len(specs)
    core_exclusions = core_exclusions | scene_exclusions
    specs = [
        filter_spec(spec, scene_exclusions)
        for spec in specs
        if spec.tag not in core_exclusions
    ]
    scene_id = str(scene["id"])
    existing_tags = scene.get("tags") or []
    existing_markers = scene.get("scene_markers") or []

    scene_names, ids = _resolved_names(
        app, resolver, source, scene_predictions, specs, scene_exclusions,
    )
    if app.CREATE_MARKERS and specs and not ids.get(source.owner_tag):
        raise ValueError(
            f"Missing marker ownership tag {source.owner_tag!r}; "
            "enable CREATE_MISSING or create it in Stash."
        )

    wanted, creates, updates, deletes, unchanged, extra_updates = _marker_plan(
        app, stash, source, scene_id, existing_markers, specs, ids,
        core_exclusions, scene_exclusions,
    )
    stats = _sync_statistics(
        existing_markers, source, core_exclusions, wanted,
        creates, updates, deletes, unchanged, extra_updates, spec_count,
    )
    final_ids, scene_changed, scene_cleaned = _scene_tag_plan(
        app, source, existing_tags, scene_names, ids, scene_exclusions,
    )
    stats["scene_tags_updated"] = int(scene_changed)
    stats["scene_tags_cleaned"] = scene_cleaned

    _publish_plan(
        stash, scene_id, creates, updates, deletes, scene_changed, final_ids,
    )
    return stats
