"""Separate-source marker payloads, matching and exclusion updates."""
from stash_ai.models import ModelSource
from stash_ai.publication import markers as marker_ops
from .types import MarkerChanges, SeparateMarkerPlan


def plan_marker_changes(app, existing: list[dict], wanted: list[dict],
                        core_exclusions: frozenset[str], source: ModelSource
                        ) -> MarkerChanges:
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
    return MarkerChanges(creates, updates, deletes, unchanged)


def plan_separate_markers(app, source, scene_id, existing_markers, specs, ids,
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
        else MarkerChanges()
    )
    creates, updates, deletes, unchanged = plan.creates, plan.updates, plan.deletes, plan.unchanged

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

    return SeparateMarkerPlan(creates, updates, deletes, unchanged, wanted, extra_updates)

