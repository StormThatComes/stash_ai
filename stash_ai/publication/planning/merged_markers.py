"""Merged publication markers."""
from __future__ import annotations
from stash_ai.publication import markers as marker_ops
from stash_ai.publication.index import MarkerIntervalIndex
from stash_ai.commands import legacy as legacy_commands
import stash_ai.evidence.labels as merge_labels
from stash_ai.runtime.timing import stage


def _wanted_markers(app, scene_id, markers, catalog, blocked_names):
    from stash_ai.evidence.marker_identity import OCCURRENCE_TAG, is_occurrence

    ids = catalog.ids

    def allowed_id(identifier):
        name = catalog.names_by_id.get(str(identifier))
        return name is not None and name not in blocked_names and str(identifier) not in catalog.managed_group_ids

    wanted, resolved_marker_labels = [], set()
    for marker in markers:
        primary = ids.get(merge_labels.AI.prefix + marker["tag"])
        if not primary:
            continue
        resolved_marker_labels.add(marker["tag"])
        extras = {ids[merge_labels.AI.prefix + label] for label in marker["extra_tags"]
                  if ids.get(merge_labels.AI.prefix + label)}
        extras.update(identifier for identifier in marker["extra_ids"] if allowed_id(identifier))
        extras.add(ids[merge_labels.AI.owner_tag])
        extras.add(catalog.group_ids[marker['tag'], is_occurrence(marker)])
        if is_occurrence(marker) and ids.get(OCCURRENCE_TAG):
            extras.add(ids[OCCURRENCE_TAG])
        extras.discard(primary)
        payload = {"scene_id": scene_id, "primary_tag_id": primary, "tag_ids": sorted(extras),
                   "title": merge_labels.AI.title_prefix + marker["tag"] + " [" +
                            "+".join(merge_labels.source_codes(marker["sources"])) + "]",
                   "seconds": round(marker["start"], 3)}
        if is_occurrence(marker):
            payload['title'] += ' · wykrycie'
        payload["end_seconds"] = (round(marker["end"], 3)
                                  if marker["end"] is not None else None)
        wanted.append((payload, {"seconds": marker["start"], "end_seconds": marker["end"]}))
    return wanted, resolved_marker_labels


def plan_markers(app, store, resolved, catalog, plan, *, migrate, report):
    """Match wanted markers to owned markers and populate create/update/delete actions."""
    scene, markers, blocked_names = resolved.scene, resolved.markers, resolved.blocked_names
    scene_id = str(scene['id'])
    with stage(report, 'marker_matching'):
        wanted, resolved_marker_labels = _wanted_markers(
            app, scene_id, markers, catalog, blocked_names)
        plan.wanted_markers = wanted
        current_markers = scene.get("scene_markers", [])
        owned = [marker for marker in current_markers if marker_ops.owned_marker(marker, merge_labels.AI)]
        existing = MarkerIntervalIndex(owned if wanted else [])
        available = {str(marker["id"]): marker for marker in owned}

        def manual_extra(tag):
            return (not tag['name'].startswith(merge_labels.AI.prefix) and tag['name'] not in blocked_names
                    and str(tag['id']) not in catalog.managed_group_ids)

        for payload, interval in wanted:
            related = existing.overlapping(payload["primary_tag_id"], interval)
            # Keep manual extras from every marker absorbed by the union, including
            # duplicates that will be deleted. A split carries them to both pieces.
            payload["tag_ids"] = sorted(set(payload["tag_ids"]) | {str(tag["id"]) for marker in related
                for tag in marker.get("tags", []) if manual_extra(tag)})
            matches = [marker for marker in existing.overlapping(
                payload["primary_tag_id"], interval, app.MARKER_MATCH_TOLERANCE_SECONDS)
                if str(marker["id"]) in available]
            if not matches:
                plan.creates.append(payload)
                continue

            def match_key(marker):
                old_end = marker.get("end_seconds") if marker.get("end_seconds") is not None else marker["seconds"]
                new_end = interval["end_seconds"] if interval["end_seconds"] is not None else interval["seconds"]
                shared = max(0, min(old_end, new_end) - max(marker["seconds"], payload["seconds"]))
                return (-shared, abs(marker["seconds"] - payload["seconds"]), str(marker["id"]))

            current = min(matches, key=match_key)
            marker_id = str(current["id"])
            del available[marker_id]
            # Additional manually attached tags outside the generated namespace survive.
            payload["tag_ids"] = sorted(set(payload["tag_ids"]) | {
                str(tag["id"]) for tag in current.get("tags", []) if manual_extra(tag)})
            if marker_ops.marker_differs(current, payload):
                plan.updates.append({**payload, "id": marker_id})
            else:
                plan.unchanged += 1

        plan.deletes = set(available) if app.CREATE_MARKERS else set()
        if migrate and app.CREATE_MARKERS:
            from stash_ai.signatures import freeze
            legacy_markers = list(legacy_commands.legacy_objects(scene, 'markers'))
            imported = {freeze(item) for item in store.imported_legacy(scene_id, 'markers', borrow=True)} if legacy_markers else set()
            predicted_labels = {marker["tag"] for marker in markers}
            for _, label, marker in legacy_markers:
                if freeze(marker) in imported and (label not in predicted_labels or label in resolved_marker_labels):
                    plan.deletes.add(str(marker["id"]))
        if app.CREATE_MARKERS and blocked_names:
            plan.deletes.update(str(marker['id']) for marker in current_markers
                                if marker['primary_tag']['name'] in blocked_names)
            changed_ids = plan.deletes | {payload['id'] for payload in plan.updates}
            for marker in current_markers:
                if str(marker['id']) in changed_ids:
                    continue
                payload = marker_ops.marker_extra_cleanup_payload(
                    marker, scene_id, blocked_names)
                if payload is not None:
                    plan.updates.append(payload)

