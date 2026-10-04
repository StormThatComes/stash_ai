"""One-scene merged-output planning and publication.

Call sync_merged directly for one-scene reconciliation.
Mode-specific planning lives in publication.planning; this coordinator owns
publication sequencing and durable snapshot promotion.
"""
from __future__ import annotations

from .planning.merged_evidence import prepare_sync_evidence
from .planning.merged_tags import resolve_sync_tags, plan_scene_tags
from .planning.merged_markers import plan_markers

from collections import Counter

import stash_ai.evidence.labels as merge_labels
from stash_ai.runtime.timing import stage


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
        from stash_ai.publication.groups import for_resolver
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
