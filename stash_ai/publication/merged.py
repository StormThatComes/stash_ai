"""One-scene merged-output planning and publication.

Call sync_merged directly for one-scene reconciliation.
This module separates evidence preparation, tag planning, marker planning and
publication while keeping merge algorithms function-oriented.
"""
from __future__ import annotations

from stash_ai.publication import markers as marker_ops
from collections import Counter
from dataclasses import dataclass, field

from stash_ai.commands import legacy as legacy_commands
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
import stash_ai.models as merge_models
from stash_ai.marker_index import MarkerIntervalIndex, PreviousMarkerRanges
from stash_ai.runtime import stage


@dataclass
class MergeSyncPlan:
    """Concrete Stash mutations and durable snapshots for one scene."""

    snapshots: dict
    existing_scene_tag_ids: set[str]
    desired_scene_tag_ids: set[str]
    final_scene_tag_ids: set[str]
    migrated_scene_tag_ids: set[str]
    wanted_markers: list[tuple[dict, dict]] = field(default_factory=list)
    creates: list[dict] = field(default_factory=list)
    updates: list[dict] = field(default_factory=list)
    deletes: set[str] = field(default_factory=set)
    unchanged: int = 0

    @property
    def scene_changed(self):
        return self.final_scene_tag_ids != self.existing_scene_tag_ids


def _previous_ranges(app, store, scene_id):
    if not app.CREATE_MARKERS or app.MARKER_USE_END_TIME:
        return []
    from stash_ai.recovery import publications
    result = []
    for snapshot in (*store.snapshot(scene_id, "markers", borrow=True),
                     *publications(store, scene_id, 'markers')):
        if snapshot:
            result.extend(merge_intervals.union_intervals(snapshot.get("contributions", []), borrow=True))
    return result


def prepare_sync_evidence(app, stash, store, scene, scene_hits, spans, *, cleanup, report):
    """Build merged markers/scene labels and apply cleanup-only retention rules."""
    scene_id = str(scene["id"])
    previous_ranges = PreviousMarkerRanges(_previous_ranges(app, store, scene_id))
    with stage(report, 'marker_union'):
        markers = merge_intervals.union_intervals(spans, borrow=True) if app.CREATE_MARKERS else []
    if report is not None:
        report.record_counts(marker_contributions=len(spans), union_markers=len(markers),
                             scene_contributions=len(scene_hits))
    scene_labels = {merge_labels.raw_label(hit["tag"]) for hit in scene_hits} if app.APPLY_SCENE_TAGS else set()
    if cleanup:
        # Cleanup never restores outputs that the user removed since the last sync.
        scene_labels.intersection_update(tag["name"][len(merge_labels.AI.prefix):] for tag in scene.get("tags", [])
                                         if tag["name"].startswith(merge_labels.AI.prefix))
        scene_hits = [hit for hit in scene_hits if hit["tag"] in scene_labels]
        present = [marker for marker in scene.get("scene_markers", [])
                   if marker_ops.owned_marker(marker, merge_labels.AI)]
        existing = MarkerIntervalIndex(present if markers else [], previous_ranges, by_name=True)
        markers = [marker for marker in markers if existing.overlapping(
            merge_labels.AI.prefix + marker["tag"], {"seconds": marker["start"], "end_seconds": marker["end"]})]
        spans = [item for marker in markers for item in marker["contributors"]]
    return scene_hits, spans, markers, scene_labels, previous_ranges


def resolve_sync_tags(app, stash, resolver, scene_id, scene, markers, spans, scene_labels,
                      tag_policies, previous_ranges, report):
    """Resolve generated tags, refresh the scene and suppress manual occurrences."""
    from stash_ai.exclusions import excluded_names
    from stash_ai.marker_groups import for_resolver
    from stash_ai.evidence.occurrences import OCCURRENCE_TAG, is_occurrence

    groups = for_resolver(app, resolver)
    blocked_names = excluded_names(tag_policies)
    names = {merge_labels.AI.prefix + label for label in scene_labels}
    for marker in markers:
        names.update(merge_labels.AI.prefix + label for label in [marker["tag"], *marker["extra_tags"]])
    if markers:
        names.add(merge_labels.AI.owner_tag)
    if any(is_occurrence(marker) for marker in markers):
        names.add(OCCURRENCE_TAG)

    with stage(report, 'tag_resolution'):
        ids = {name: resolver.resolve(name) for name in sorted(names)}
    if markers and not ids.get(merge_labels.AI.owner_tag):
        raise ValueError("Missing AI marker ownership tag; enable CREATE_MISSING.")

    # Refresh after tag creation/reparenting, preserving newly added manual tags.
    with stage(report, 'scene_refresh'):
        scene = stash.find_scene_for_merge(scene_id)
    if report is not None:
        report.record_counts(existing_markers=len(scene.get('scene_markers', [])),
                             existing_scene_tags=len(scene.get('tags', [])))

    # A manually maintained ai: marker already supplies a location. It remains
    # untouched, and must not receive a duplicate automatically generated point.
    manual_labels = {marker['primary_tag']['name'] for marker in scene.get('scene_markers', [])
                     if not marker_ops.owned_marker(marker, merge_labels.AI)}
    suppressed = {marker['tag'] for marker in markers
                  if is_occurrence(marker) and merge_labels.AI.prefix + marker['tag'] in manual_labels}
    if suppressed:
        markers = [marker for marker in markers
                   if not (is_occurrence(marker) and marker['tag'] in suppressed)]
        spans = [span for span in spans
                 if not (span.get('derived') == 'occurrence' and span['tag'] in suppressed)]

    # Imported/manual extras may retain IDs in old namespaces. Resolve each
    # unknown ID at most once through the shared resolver, not per marker.
    for tag in scene.get('tags', []):
        resolver.names_by_id[str(tag['id'])] = tag['name']
    for marker in scene.get('scene_markers', []):
        for tag in [marker['primary_tag'], *marker.get('tags', [])]:
            resolver.names_by_id[str(tag['id'])] = tag['name']

    return scene, markers, spans, ids, groups, blocked_names, suppressed


def plan_scene_tags(app, store, scene_id, scene, scene_hits, scene_labels, ids, blocked_names, *, migrate):
    """Plan scene associations and the durable scene snapshot."""
    from stash_ai.recovery import publications

    current_tags = scene.get("tags", [])
    migrated_scene_ids = ({str(tag["id"]) for tag in current_tags if legacy_commands.legacy_label(tag["name"])}
                          if migrate else set())
    published, pending = store.snapshot(scene_id, "scene", borrow=True)
    owned_before = set((published or {}).get("owned_ids", [])) | set((pending or {}).get("owned_ids", []))
    owned_before.update(identifier for snapshot in publications(store, scene_id, 'scene')
                        for identifier in snapshot.get('owned_ids', []))
    existing_ids = {str(tag["id"]) for tag in current_tags}
    desired_ids = {ids[merge_labels.AI.prefix + label] for label in scene_labels
                   if ids.get(merge_labels.AI.prefix + label)}
    unresolved_labels = {label for label in scene_labels if not ids.get(merge_labels.AI.prefix + label)}

    # CREATE_MISSING=False must not remove a legacy association lacking a replacement.
    migrated_scene_ids.difference_update(str(tag["id"]) for tag in current_tags
        if (parsed := legacy_commands.legacy_label(tag["name"])) and parsed[1] in unresolved_labels)
    forbidden_ids = {str(tag['id']) for tag in current_tags if tag['name'] in blocked_names}
    preserved = existing_ids - owned_before - migrated_scene_ids - forbidden_ids
    final_ids = preserved | desired_ids
    snapshots = {}
    if app.APPLY_SCENE_TAGS:
        snapshots["scene"] = {"owned_ids": sorted(desired_ids - preserved), "contributions": scene_hits}

    return MergeSyncPlan(snapshots, existing_ids, desired_ids, final_ids, migrated_scene_ids)


def _wanted_markers(app, stash, resolver, scene_id, markers, ids, groups, blocked_names):
    from stash_ai.evidence.occurrences import OCCURRENCE_TAG, is_occurrence

    def allowed_id(identifier):
        name = resolver.name_for_id(str(identifier))
        return name is not None and name not in blocked_names and not groups.owns_id(identifier)

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
        extras.add(groups.resolve(marker['tag'], is_occurrence(marker)))
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
                                  if marker["end"] is not None and app.MARKER_USE_END_TIME else None)
        wanted.append((payload, {"seconds": marker["start"], "end_seconds": marker["end"]}))
    return wanted, resolved_marker_labels


def plan_markers(app, stash, resolver, scene_id, scene, markers, ids, groups, blocked_names,
                 previous_ranges, plan, *, migrate, report):
    """Match wanted markers to owned markers and populate create/update/delete actions."""
    with stage(report, 'marker_matching'):
        wanted, resolved_marker_labels = _wanted_markers(
            app, stash, resolver, scene_id, markers, ids, groups, blocked_names)
        plan.wanted_markers = wanted
        current_markers = scene.get("scene_markers", [])
        owned = [marker for marker in current_markers if marker_ops.owned_marker(marker, merge_labels.AI)]
        existing = MarkerIntervalIndex(owned if wanted else [], previous_ranges)
        available = {str(marker["id"]): marker for marker in owned}

        def manual_extra(tag):
            return (not tag['name'].startswith(merge_labels.AI.prefix) and tag['name'] not in blocked_names
                    and not groups.owns(tag))

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
                marker = existing.temporal(marker)
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
            predicted_labels = {marker["tag"] for marker in markers}
            for marker in current_markers:
                parsed = legacy_commands.legacy_label((marker.get("primary_tag") or {}).get("name", ""))
                if parsed and marker_ops.owned_marker(marker, merge_models.SOURCES[parsed[0]]):
                    if parsed[1] not in predicted_labels or parsed[1] in resolved_marker_labels:
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


def sync_statistics(app, plan, markers, suppressed):
    from stash_ai.evidence.occurrences import is_occurrence

    stats = Counter(
        scene_tags_updated=int(app.APPLY_SCENE_TAGS and plan.scene_changed),
        legacy_scene_tags_migrated=len(plan.migrated_scene_tag_ids) if app.APPLY_SCENE_TAGS else 0,
        markers_created=len(plan.creates),
        markers_updated=len(plan.updates),
        markers_deleted=len(plan.deletes),
        markers_unchanged=plan.unchanged,
        merged_markers=len(plan.wanted_markers),
        merged_scene_tags=len(plan.desired_scene_tag_ids),
    )
    stats['short_extras'] = sum(item.get('qualification') == 'short_episode'
        for marker in markers for item in marker.get('extra_evidence', {}).values())
    stats['occurrence_markers'] = sum(is_occurrence(marker) for marker in markers)
    stats['occurrence_manual_locations'] = len(suppressed)
    stats['cross_model_extras'] = sum(bool(set(item.get('source_evidence', {})) - set(marker['sources']))
        for marker in markers for tag, item in marker.get('extra_evidence', {}).items()
        if tag in marker['extra_tags'])
    return stats


def publish_sync_plan(app, stash, resolver, store, scene_id, scene, markers, scene_labels,
                      plan, stats, *, report, publication_key):
    """Apply or preview a completed sync plan and durably commit its snapshot."""
    if stash.dry_run:
        from stash_ai.marker_groups import for_resolver
        for_resolver(app, resolver).record_preview(scene, plan.creates, plan.updates, plan.deletes)
        write = report.detail if report is not None else print
        write(f"[DRY] scene {scene_id}: ai scene tags={len(plan.desired_scene_tag_ids)}, "
              f"markers create={len(plan.creates)} update={len(plan.updates)} delete={len(plan.deletes)}; "
              f"migrate scene tags={stats['legacy_scene_tags_migrated']}.")
        if app.APPLY_SCENE_TAGS and (report is None or report.verbose):
            write("  Scene tags: " + (", ".join(merge_labels.AI.prefix + label for label in sorted(scene_labels)) or "(none)"))
        for marker in markers if report is None or report.verbose else ():
            end = app.format_time(marker["end"]) if marker["end"] is not None else "point (end unknown)"
            write(f"  {app.format_time(marker['start'])} -> {end}: ai:{marker['tag']} "
                  f"[{'+'.join(merge_labels.source_codes(marker['sources']))}]")
        store.publish(scene_id, plan.snapshots)  # temporary preview overlay only
        return stats

    with stage(report, 'publication_prepare'):
        store.prepare(scene_id, plan.snapshots)
    created = []
    with stage(report, 'publication_mutations'):
        for payload in plan.creates:
            created.append((payload, stash.create_marker(payload)))
        for payload in plan.updates:
            stash.update_marker(payload)
        if app.APPLY_SCENE_TAGS and plan.scene_changed:
            stash.update_scene_tags(scene_id, sorted(plan.final_scene_tag_ids))
        for marker_id in sorted(plan.deletes):
            stash.delete_marker(marker_id)

    publication = None
    with stage(report, 'publication_fingerprint'):
        if publication_key is not None and not resolver.missing:
            from stash_ai.performance import projected_scene, scene_fingerprint
            observed = projected_scene(
                scene, resolver, created, plan.updates, plan.deletes,
                plan.final_scene_tag_ids if app.APPLY_SCENE_TAGS else plan.existing_scene_tag_ids)
            if observed is None:
                observed = stash.find_scene(scene_id)
            publication = publication_key, scene_fingerprint(observed)
    with stage(report, 'publication_commit'):
        store.commit_prepared(scene_id, plan.snapshots, publication=publication)
    return stats


def sync_merged(app, stash, resolver, store, scene, scene_hits, spans, *, migrate=True, cleanup=False,
                report=None, tag_policies=None, publication_key=None):
    """Reconcile only script-owned markers and recorded scene associations."""
    scene_id = str(scene["id"])
    report = report or getattr(store, 'report', None)
    scene_hits, spans, markers, scene_labels, previous_ranges = prepare_sync_evidence(
        app, stash, store, scene, scene_hits, spans, cleanup=cleanup, report=report)
    scene, markers, spans, ids, groups, blocked_names, suppressed = resolve_sync_tags(
        app, stash, resolver, scene_id, scene, markers, spans, scene_labels,
        tag_policies, previous_ranges, report)
    plan = plan_scene_tags(
        app, store, scene_id, scene, scene_hits, scene_labels, ids, blocked_names, migrate=migrate)
    plan_markers(
        app, stash, resolver, scene_id, scene, markers, ids, groups, blocked_names,
        previous_ranges, plan, migrate=migrate, report=report)
    if app.CREATE_MARKERS:
        plan.snapshots["markers"] = {"contributions": spans}
    stats = sync_statistics(app, plan, markers, suppressed)
    return publish_sync_plan(
        app, stash, resolver, store, scene_id, scene, markers, scene_labels, plan, stats,
        report=report, publication_key=publication_key)
