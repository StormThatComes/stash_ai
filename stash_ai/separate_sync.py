"""Separate-output reconciliation helpers.

Public compatibility entry points remain in stash_ai.cli. Helpers that depend on
live CLI policy flags receive the app module explicitly so existing monkeypatch
surfaces remain valid.
"""
from __future__ import annotations

from collections import Counter

from stash_ai.taggers import JOYTAG


def owned_marker(marker: dict, source=JOYTAG) -> bool:
    return bool(
        (marker.get("title") or "").startswith(source.title_prefix)
        and any(tag.get("name") == source.owner_tag for tag in marker.get("tags") or [])
    )


def marker_has_excluded_core(marker: dict, core_exclusions: frozenset[str],
                             source=JOYTAG) -> bool:
    name = (marker.get("primary_tag") or {}).get("name") or ""
    return name.startswith(source.prefix) and name[len(source.prefix):] in core_exclusions


def marker_extra_cleanup_payload(marker: dict, scene_id: str,
                                 blocked_names: set[str], has_end: bool):
    """Preserve marker identity, primary, title and timing while stripping extras."""
    primary = str(marker["primary_tag"]["id"])
    extras = {
        str(tag["id"])
        for tag in marker.get("tags", [])
        if str(tag["id"]) != primary
    }
    kept = {
        str(tag["id"])
        for tag in marker.get("tags", [])
        if str(tag["id"]) != primary and tag["name"] not in blocked_names
    }
    if extras == kept:
        return None
    payload = dict(
        id=str(marker["id"]),
        scene_id=scene_id,
        title=marker["title"],
        seconds=marker["seconds"],
        primary_tag_id=primary,
        tag_ids=sorted(kept),
    )
    if has_end:
        payload["end_seconds"] = marker.get("end_seconds")
    return payload


def cleanup_scene_marker_cores(app, stash, scene: dict,
                               core_exclusions: frozenset[str], report=None,
                               source=JOYTAG,
                               tag_exclusions: frozenset[str] = frozenset()) -> Counter:
    """Remove excluded cores and global extras; never infer replacement runs."""
    from stash_ai.vtt import format_time

    core_exclusions = core_exclusions | tag_exclusions
    excluded = [
        marker
        for marker in scene.get("scene_markers") or []
        if (
            owned_marker(marker, source)
            and marker_has_excluded_core(marker, core_exclusions, source)
        )
        or marker_has_excluded_core(marker, tag_exclusions, source)
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
        for tag in tag_exclusions | app.MARKER_TAG_BLOCKLIST
    }
    for marker in scene.get("scene_markers") or []:
        if str(marker["id"]) in excluded_ids:
            continue
        names = (
            blocked_names
            if owned_marker(marker, source)
            else {source.prefix + tag for tag in tag_exclusions}
        )
        payload = marker_extra_cleanup_payload(
            marker,
            str(scene["id"]),
            names,
            stash.has_end_time,
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


def marker_payload(app, spec, scene_id: str, ids: dict[str, str | None],
                   has_end: bool, source=JOYTAG) -> dict:
    primary = ids[source.prefix + spec.tag]
    extras = {ids[source.prefix + tag] for tag in spec.extra_tags}
    extras.add(ids[source.owner_tag])
    extras.discard(primary)
    extras.discard(None)
    payload = {
        "scene_id": scene_id,
        "title": source.title_prefix + spec.tag,
        "seconds": round(spec.start, 3),
        "primary_tag_id": primary,
        "tag_ids": sorted(extras),
    }
    if has_end:
        payload["end_seconds"] = (
            round(spec.end, 3) if app.MARKER_USE_END_TIME else None
        )
    return payload


def marker_differs(existing: dict, wanted: dict) -> bool:
    if (
        existing["title"] != wanted["title"]
        or abs(existing["seconds"] - wanted["seconds"]) > 0.0005
    ):
        return True
    if str(existing["primary_tag"]["id"]) != wanted["primary_tag_id"]:
        return True
    # Some API versions include the primary tag in the returned tags list.
    extra_ids = {
        str(tag["id"])
        for tag in existing["tags"]
    } - {wanted["primary_tag_id"]}
    if extra_ids != set(wanted["tag_ids"]):
        return True
    if "end_seconds" in wanted:
        old_end, new_end = existing.get("end_seconds"), wanted["end_seconds"]
        if old_end is None or new_end is None:
            return old_end != new_end
        return abs(old_end - new_end) > 0.0005
    return False


def plan_marker_changes(app, existing: list[dict], wanted: list[dict],
                        core_exclusions: frozenset[str], source=JOYTAG
                        ) -> tuple[list[dict], list[dict], list[str], int]:
    available = {
        str(marker["id"]): marker
        for marker in existing
        if owned_marker(marker, source)
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
        if marker_differs(current, payload):
            updates.append({**payload, "id": marker_id})
        else:
            unchanged += 1

    # The exclusion policy always wins over retaining obsolete generated runs.
    deletes = sorted(
        marker_id
        for marker_id, marker in available.items()
        if app.REPLACE_MARKERS
        or marker_has_excluded_core(marker, core_exclusions, source)
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
            marker_payload(app, spec, scene_id, ids, stash.has_end_time, source)
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
                if marker_has_excluded_core(marker, scene_exclusions, source)
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
            payload = marker_extra_cleanup_payload(
                marker,
                scene_id,
                blocked_names,
                stash.has_end_time,
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


def sync_scene(app, stash, resolver, scene: dict,
               scene_predictions: list[tuple[str, float]], specs: list,
               core_exclusions: frozenset[str], source=JOYTAG,
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
        app,
        resolver,
        source,
        scene_predictions,
        specs,
        scene_exclusions,
    )
    if app.CREATE_MARKERS and specs and not ids.get(source.owner_tag):
        raise ValueError(
            f"Missing marker ownership tag {source.owner_tag!r}; "
            "enable CREATE_MISSING or create it in Stash."
        )

    wanted, creates, updates, deletes, unchanged, extra_updates = _marker_plan(
        app,
        stash,
        source,
        scene_id,
        existing_markers,
        specs,
        ids,
        core_exclusions,
        scene_exclusions,
    )

    excluded_ids = {
        str(marker["id"])
        for marker in existing_markers
        if owned_marker(marker, source)
        and marker_has_excluded_core(marker, core_exclusions, source)
    }
    stats = Counter(
        markers_created=len(creates),
        markers_updated=len(updates),
        markers_deleted=len(deletes),
        markers_unchanged=unchanged,
        markers_skipped=spec_count - len(wanted),
        markers_extras_cleaned=len(extra_updates),
        markers_core_cleaned=len(excluded_ids.intersection(deletes)),
    )

    final_ids, scene_changed, scene_cleaned = _scene_tag_plan(
        app,
        source,
        existing_tags,
        scene_names,
        ids,
        scene_exclusions,
    )
    stats["scene_tags_updated"] = int(scene_changed)
    stats["scene_tags_cleaned"] = scene_cleaned

    if stash.dry_run:
        return stats

    # No destructive cleanup until all new/replacement marker writes succeed.
    for payload in creates:
        stash.create_marker(payload)
    for payload in updates:
        stash.update_marker(payload)
    if scene_changed:
        stash.update_scene_tags(scene_id, sorted(final_ids))
    for marker_id in deletes:
        stash.delete_marker(marker_id)
    return stats
