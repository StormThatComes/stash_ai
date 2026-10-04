"""Merged publication tags."""
from __future__ import annotations
from stash_ai.publication import markers as marker_ops
from stash_ai.commands import legacy as legacy_commands
import stash_ai.evidence.labels as merge_labels
from stash_ai.runtime.timing import stage
from .types import MergeSyncPlan


def resolve_sync_tags(app, stash, resolver, scene_id, scene, markers, spans, scene_labels,
                      tag_policies, previous_ranges, report):
    """Resolve generated tags, refresh the scene and suppress manual occurrences."""
    from stash_ai.exclusions import excluded_names
    from stash_ai.publication.groups import for_resolver
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
    from stash_ai.storage.recovery import publications

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
