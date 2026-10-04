"""Reconcile cleanup changes for one scene before moving to the next."""
from __future__ import annotations
from stash_ai.cleanup import scene_tags

from collections import Counter
from types import SimpleNamespace

import stash_ai.evidence.labels as merge_labels
from stash_ai.publication import markers as marker_ops
from stash_ai.publication import separate as separate_sync
import stash_ai.cleanup.markers as separate_sync_markers
from stash_ai.publication import merged as merge_sync
from stash_ai.cleanup import evidence


def cleanup_scene(app, stash, resolver, store, current, settings, selected, args,
                  core_policies, scene_policies, scoped_tags, report):
    """Reconcile one affected scene, rechecking prefetched data before writes."""
    from stash_ai.exclusions import excluded_names
    from stash_ai.evidence.scenes import cleanup_scene_support
    scene_id, stats = str(current["id"]), Counter()
    if args.output_mode == 'merged':
        published = {}
        for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)):
            if not enabled:
                continue
            snapshot, pending = store.snapshot(scene_id, kind, borrow=True)
            if pending is not None:
                raise RuntimeError("An interrupted merge is pending; run --merge-only before cleanup.")
            from stash_ai.publication.recovery import reconcile
            published[kind], recovered = reconcile(store, current, kind, snapshot)
            stats['publications_reconciled'] += int(recovered)
        if app.CREATE_MARKERS and published.get("markers") is None and any(
                marker_ops.owned_marker(marker, merge_labels.AI) for marker in current.get("scene_markers", [])):
            raise RuntimeError("AI marker source state is missing; restore the original --state-db.")
        hits = (published.get("scene") or {}).get("contributions", [])
        spans = (published.get("markers") or {}).get("contributions", [])
        clean_hits = cleanup_scene_support(app, store, scene_id, hits, selected, scene_policies)
        clean_spans = evidence.cleanup_support(app, store, scene_id, spans, selected, core_policies, scene_policies)
        blocked_names = excluded_names(scoped_tags)
        forbidden_present = (app.APPLY_SCENE_TAGS and any(tag['name'] in blocked_names for tag in current.get('tags', []))) or (
            app.CREATE_MARKERS and any(tag['name'] in blocked_names for marker in current.get('scene_markers', [])
                                      for tag in [marker['primary_tag'], *marker.get('tags', [])]))
        if clean_hits != hits or clean_spans != spans or forbidden_present:
            result = merge_sync.sync_merged(app, stash, resolver, store, current, clean_hits, clean_spans,
                                       migrate=False, cleanup=True, report=report, tag_policies=scoped_tags)
            stats.update(result)
            if not stash.dry_run and any(result[name] for name in (
                    'scene_tags_updated', 'markers_created', 'markers_updated', 'markers_deleted')):
                current = stash.find_scene(scene_id)
    # Most scenes have no old source-prefixed outputs. Reading the same scene
    # once for every configured model previously dominated no-op cleanup.
    for config in settings:
        source = config.source
        if not stash.dry_run and stash.merge_only_reads:
            # Batch snapshots can age while earlier scenes are processed. Reuse
            # the dry-run planner to avoid extra reads for unchanged outputs.
            preview = SimpleNamespace(dry_run=True)
            planned = _cleanup_source(app, preview, current, source, core_policies, scene_policies)
            if not any(planned.values()):
                stats.update(planned)
                continue
            current = stash.find_scene(scene_id)
        result = _cleanup_source(app, stash, current, source, core_policies, scene_policies, report.detail)
        stats.update(result)
        if not stash.dry_run and any(result[name] for name in ('scene_tags_cleaned', 'markers_deleted', 'markers_updated')):
            current = stash.find_scene(scene_id)
    return stats

def _cleanup_source(app, stash, current, source, core_policies, scene_policies, report=None):
    result = Counter()
    if app.APPLY_SCENE_TAGS:
        result.update(scene_tags.cleanup_scene_tags(stash, current, scene_policies[source.key], source, report))
    if app.CREATE_MARKERS:
        result.update(separate_sync_markers.cleanup_scene_marker_cores(
            app, stash, current, core_policies[source.key], source, report, scene_policies[source.key]))
    return result

