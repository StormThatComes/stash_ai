"""One-scene merged-output planning and publication.

Call sync_merged directly for one-scene reconciliation.
Mode-specific planning lives in publication.planning; this coordinator owns
publication sequencing and durable snapshot promotion.
"""
from __future__ import annotations

from .planning.merged_evidence import prepare_sync_evidence
from .planning.merged_tags import resolve_sync_tags, resolve_marker_catalog, plan_scene_tags
from .planning.merged_markers import plan_markers

from collections import Counter
from itertools import batched

import stash_ai.evidence.labels as merge_labels
from stash_ai.runtime.timing import stage
from stash_ai.runtime.console import activity, status
from stash_ai.runtime.mutations import write as confirmed_write


def sync_statistics(app, plan, markers, suppressed):
    from stash_ai.evidence.marker_identity import is_occurrence

    stats = Counter(
        scene_tags_updated=int(app.APPLY_SCENE_TAGS and plan.scene_changed),
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
    composition = dict(create=len(plan.creates), update=len(plan.updates), delete=len(plan.deletes),
                       scene_tags=int(app.APPLY_SCENE_TAGS and plan.scene_changed))
    status(report, scene=scene_id, planned=' '.join(f'{key}={value}' for key, value in composition.items()),
           write_outcome=None, local_commit='preview' if stash.dry_run else 'not prepared')
    if report is not None:
        report.counts['mutation_rows_planned'] += sum(composition.values())
    if stash.dry_run:
        if report is not None and report.dry_run_plan is not None:
            report.dry_run_plan.scene(scene_id, plan, resolver, app, stats)
        from stash_ai.publication.groups import for_resolver
        for_resolver(app, resolver).record_preview(scene, plan.creates, plan.updates, plan.deletes)
        write = report.detail if report is not None else print
        write(f"[DRY] scene {scene_id}: ai scene tags={len(plan.desired_scene_tag_ids)}, "
              f"markers create={len(plan.creates)} update={len(plan.updates)} delete={len(plan.deletes)}.")
        if app.APPLY_SCENE_TAGS and (report is None or report.verbose):
            write("  Scene tags: " + (", ".join(merge_labels.AI.prefix + label for label in sorted(scene_labels)) or "(none)"))
        for marker in markers if report is None or report.verbose else ():
            end = app.services.format_time(marker["end"]) if marker["end"] is not None else "point (end unknown)"
            write(f"  {app.services.format_time(marker['start'])} -> {end}: ai:{marker['tag']} "
                  f"[{'+'.join(merge_labels.source_codes(marker['sources']))}]")
        store.publish(scene_id, plan.snapshots)  # temporary preview overlay only
        return stats

    with activity(report, 'prepare publication intent'), stage(report, 'publication_prepare'):
        store.prepare(scene_id, plan.snapshots)
    if report is not None:
        report.counts['publication_pending'] += 1
    status(report, local_commit='pending', recovery='pending intent')
    created = []
    with stage(report, 'publication_mutations'):
        for payload in plan.creates:
            created.append((payload, confirmed_write(report, 'marker_create', lambda: stash.create_marker(payload))))
        for payload in plan.updates:
            confirmed_write(report, 'marker_update', lambda: stash.update_marker(payload))
        if app.APPLY_SCENE_TAGS and plan.scene_changed:
            confirmed_write(report, 'scene_tags', lambda: stash.update_scene_tags(scene_id, sorted(plan.final_scene_tag_ids)))
        for group in batched(sorted(plan.deletes), 200):
            confirmed_write(report, 'marker_delete', lambda: stash.delete_markers(list(group)), rows=len(group))
    status(report, write='idle', local_commit='Stash confirmed / local commit pending')

    publication = None
    with activity(report, 'publication fingerprint'), stage(report, 'publication_fingerprint'):
        if publication_key is not None and not resolver.missing:
            from stash_ai.publication.fingerprints import projected_scene, scene_fingerprint
            observed = projected_scene(
                scene, resolver, created, plan.updates, plan.deletes,
                plan.final_scene_tag_ids if app.APPLY_SCENE_TAGS else plan.existing_scene_tag_ids)
            if observed is None:
                observed = stash.find_scene(scene_id)
            publication = publication_key, scene_fingerprint(observed)
    with activity(report, 'commit publication'), stage(report, 'publication_commit'):
        store.commit_prepared(scene_id, plan.snapshots, publication=publication)
    if report is not None:
        report.counts['publication_pending'] -= 1
        report.counts['publication_committed'] += 1
    status(report, completed=1, local_commit='committed', recovery=None)
    return stats


def sync_merged(app, stash, resolver, store, scene, scene_hits, spans, *, cleanup=False,
                report=None, tag_policies, publication_key=None, prepared_markers=None):
    """Reconcile only script-owned markers and recorded scene associations."""
    scene_id = str(scene["id"])
    store.initialization.require_ready(scene_id)
    report = report or store.report
    with activity(report, 'marker union'):
        evidence = prepare_sync_evidence(
            app, stash, store, scene, scene_hits, spans, cleanup=cleanup, report=report,
            prepared_markers=prepared_markers)
    with activity(report, 'resolve tags / refresh scene'):
        resolved = resolve_sync_tags(app, stash, resolver, scene_id, evidence, tag_policies, report)
    with activity(report, 'resolve marker catalog'), stage(report, 'tag_resolution'):
        catalog = resolve_marker_catalog(resolver, resolved)
    scene, markers, spans = resolved.scene, resolved.markers, resolved.spans
    scene_labels, suppressed = evidence.scene_labels, resolved.suppressed
    plan = plan_scene_tags(
        app, store, scene_id, scene, evidence.scene_hits, scene_labels,
        resolved.ids, resolved.blocked_names)
    with activity(report, 'match markers'):
        plan_markers(app, store, resolved, catalog, plan, report=report)
    if app.CREATE_MARKERS:
        plan.snapshots["markers"] = {"contributions": spans}
    stats = sync_statistics(app, plan, markers, suppressed)
    return publish_sync_plan(
        app, stash, resolver, store, scene_id, scene, markers, scene_labels, plan, stats,
        report=report, publication_key=publication_key)
