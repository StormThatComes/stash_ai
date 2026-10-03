"""Cleanup-specific merge reconciliation and stateful cleanup orchestration."""
from __future__ import annotations

from collections import Counter
import json
from types import SimpleNamespace

from tqdm import tqdm

from stash_ai import separate_sync
from stash_ai import merge_sync
from stash_ai import merge
from stash_ai.runtime import RunReport
from stash_ai.taggers import SOURCES


def cleanup_support(app, store, scene_id, spans, selected, core_policies, tag_policies=None):
    """Re-evaluate revoked votes, bounded by the already-published source ranges."""
    from stash_ai.temporal import TemporalPolicy
    from stash_ai.fusion import prune_blocked_extras, refresh_clipped_evidence
    from stash_ai.exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    records, rebuilt, available_by_source = None, {}, {}
    cleaned = []
    for span in spans:
        if span.get('derived') == 'occurrence':
            from stash_ai.occurrences import cleanup_occurrence
            point = cleanup_occurrence(app, store, scene_id, span, selected, core_policies, tag_policies)
            if point is not None:
                cleaned.append(point)
            continue
        own_blocked = tag_policies[span['source']] if span['source'] in selected else frozenset()
        if span.get('derived') != 'shared_frames' and span['source'] in selected and (
                span['tag'] in core_policies[span['source']] or span['tag'] in own_blocked):
            continue
        core_affected = any(source in selected and (span["tag"] in core_policies[source] or
                            span["tag"] in tag_policies[source])
                            for source in span.get("support_sources", []))
        extra_labels = set(span.get('extra_tags', [])) | set(span.get('extra_support', {}))
        extra_sources = set(span.get('frame_context', {})) | {span['source']}
        extras_affected = any(extra_labels & tag_policies[source] for source in extra_sources & selected)
        affected = core_affected or extras_affected
        if not affected:
            cleaned.append(span)
            continue
        if not span.get('build_options') or not span.get('evidence_signature'):
            # Imported ranges have no frame proofs. Their extras belong to the
            # captured source and can still be removed without loading assets.
            cleaned.append(prune_blocked_extras(span, own_blocked))
            continue
        records = store.records(scene_id, "markers", borrow=True) if records is None else records
        required = span.get("frame_context") or {**span.get("support_context", {}), span["source"]: [span["evidence_signature"]]}
        matching = {}
        for source, signatures in required.items():
            if source not in available_by_source:
                available_by_source[source] = {merge.marker_segment_signature(s): s for s in records.get(source, [])}
            available = available_by_source[source]
            if not set(signatures) <= set(available):
                raise RuntimeError("Consensus sources changed since publication; run --merge-only before cleanup.")
            matching[source] = [available[signature] for signature in signatures]
        options = span["build_options"]
        key = json.dumps([options, required], sort_keys=True)
        if key not in rebuilt:
            context = SimpleNamespace(**vars(app))
            context.TEMPORAL_POLICY = TemporalPolicy.from_snapshot(options["policy"])
            context.MARKER_THRESHOLDS = options.get("marker_thresholds", {})
            for name, value in options["settings"].items():
                setattr(context, name, value)
            # Preserve the retired option only as part of this published proof.
            legacy_blocklist = frozenset(options.get("blocklist", []))
            policies = {source: frozenset(options["core_exclusions"][source]) |
                        (core_policies[source] if source in selected else frozenset()) for source in SOURCES}
            frozen_tags = {source: frozenset(options.get('tag_exclusions', {}).get(source, [])) | legacy_blocklist |
                           (tag_policies[source] if source in selected else frozenset()) for source in SOURCES}
            # Cleanup can revoke old evidence but cannot add newly cached votes.
            # Older frames may lack blocklist or retention metadata. Verified
            # proofs are replayed with frozen exclusions, then clipped to published
            # ranges/extras below; this never admits previously missing labels.
            rebuilt[key] = (merge.marker_contributions(
                                context, matching, policies, tag_policies=frozen_tags,
                                allow_unknown_blocklist=True, allow_unknown_retention=True,
                                legacy_blocklist=legacy_blocklist),
                            context, policies, matching, frozen_tags)
        candidates, context, policies, matching, frozen_tags = rebuilt[key]
        for item in candidates:
            if item["tag"] != span["tag"]:
                continue
            if span.get("derived") != "shared_frames" and (item["source"] != span["source"] or item.get("derived")):
                continue
            start, end = max(span["start"], item["start"]), min(span["end"], item["end"])
            if end <= start:
                continue
            clipped = {**item, "start": start, "end": end}
            if options.get("blocklist"):
                # A later cleanup still needs these frozen bans to validate the
                # same legacy frames, even though new publications use none.
                clipped["build_options"] = {**item["build_options"], "blocklist": options["blocklist"]}
            if (start, end) != (item["start"], item["end"]):
                clipped.update(hits=None, strong_hits=None, extra_evidence=[], cleanup_clipped=True)
                if clipped.get("derived") == "shared_frames":
                    clipped.update(shared_hits=None, shared_strong_hits=None)
                # Legacy proofs can only be clipped; compatible frame records
                # below allow exact scores and hit proofs to be rebuilt.
                clipped["extra_tags"] = sorted(set(item.get("extra_tags", [])) & set(span.get("extra_tags", [])))
                clipped["extra_support"] = {label: dict(
                    intervals=[[max(a,start), min(b,end)] for a,b in evidence["intervals"] if min(b,end)>max(a,start)],
                    sample_times=[t for t in evidence["sample_times"] if start <= t < end])
                    for label,evidence in item.get("extra_support", {}).items()}
                if not refresh_clipped_evidence(context, clipped, matching, policies, frozen_tags):
                    continue
            # Cleanup only revokes published extras; a normal merge can admit
            # additional labels freed by a changed per-frame cap.
            new_extras = (set(clipped.get('extra_tags', [])) | set(clipped.get('extra_support', {}))) - extra_labels
            clipped = prune_blocked_extras(clipped, new_extras)
            if clipped not in cleaned:
                cleaned.append(clipped)
    return cleaned


def cleanup_scene(app, stash, resolver, store, current, settings, selected, args,
                  core_policies, scene_policies, scoped_tags, report):
    """Reconcile one affected scene, rechecking prefetched data before writes."""
    from stash_ai.exclusions import excluded_names
    from stash_ai.scene import cleanup_scene_support
    scene_id, stats = str(current["id"]), Counter()
    if getattr(args, 'output_mode', 'merged') == 'merged':
        published = {}
        for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)):
            if not enabled:
                continue
            snapshot, pending = store.snapshot(scene_id, kind, borrow=True)
            if pending is not None:
                raise RuntimeError("An interrupted merge is pending; run --merge-only before cleanup.")
            from stash_ai.recovery import reconcile
            published[kind], recovered = reconcile(app, stash, store, current, kind, snapshot)
            stats['publications_reconciled'] += int(recovered)
        if app.CREATE_MARKERS and published.get("markers") is None and any(
                separate_sync.owned_marker(marker, merge.AI) for marker in current.get("scene_markers", [])):
            raise RuntimeError("AI marker source state is missing; restore the original --state-db.")
        hits = (published.get("scene") or {}).get("contributions", [])
        spans = (published.get("markers") or {}).get("contributions", [])
        clean_hits = cleanup_scene_support(app, store, scene_id, hits, selected, scene_policies)
        clean_spans = cleanup_support(app, store, scene_id, spans, selected, core_policies, scene_policies)
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
        if not stash.dry_run and getattr(stash, 'merge_only_reads', False):
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
        result.update(app.cleanup_scene_tags(stash, current, scene_policies[source.key], source, report))
    if app.CREATE_MARKERS:
        result.update(separate_sync.cleanup_scene_marker_cores(
            app, stash, current, core_policies[source.key], source, report, scene_policies[source.key]))
    return result


class CleanupRun:
    """Own the resources and phases of one cleanup execution."""

    def __init__(self, app, settings, args, core_policies, scene_policies):
        from stash_ai.exclusions import selected_policies
        self.app, self.settings, self.args = app, settings, args
        self.core_policies, self.scene_policies = core_policies, scene_policies
        self.selected = {config.source.key for config in settings}
        self.scoped_tags = selected_policies(scene_policies, self.selected)
        self.stats = Counter()
        self.report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
        self.wide = not args.scene_id and self.selected == set(SOURCES)
        self.store = self.stash = self.cleanup_stash = self.resolver = self.sql = None
        self.scene_ids = None

    def run(self):
        app, args = self.app, self.args
        with merge.StateStore(args.state_db, merge.scope_for(app.STASH_URL), app.DRY_RUN,
                cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                allow_cleanup=self.wide or getattr(args, 'abandon_pending', False), cleanup_preview=True,
                memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as self.store:
            if getattr(args, 'abandon_pending', False):
                from stash_ai.recovery import abandon
                self.stats.update(abandon(self.store, self.report.write))
            self.stash = self.cleanup_stash = app.create_stash_client(app.DRY_RUN)
            try:
                self._prepare_strategy()
                self._cleanup_scenes()
                self._prune()
            except Exception as exc:
                self.stats['errors'] += 1
                self.report.issue('cleanup', '', None, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
            finally:
                self.stash.close()
            self.report.finish('Cleanup' + (' (planned)' if app.DRY_RUN else ''), self.stats, self.store.cache)
        return int(bool(self.stats["errors"]))

    def _prepare_strategy(self):
        app, args = self.app, self.args
        from stash_ai.cleanup import CleanupClient
        self.scene_ids = args.scene_id or None
        strategy = getattr(args, 'cleanup_strategy', 'auto')
        client = CleanupClient(self.stash) if strategy in ('auto', 'sql') else None
        if strategy in ('auto', 'sql'):
            from stash_ai.cleanup_sql import SQLCleanup
            candidate = SQLCleanup(client, args, self.stats, self.report)
            try:
                candidate.probe()
                self.sql = candidate
            except Exception as exc:
                if strategy == 'sql' or self.store.cleanup_pending:
                    raise
                self.report.write('SQL cleanup unavailable (' + app.safe_error(exc, api_key=app.STASH_API_KEY) +
                                  '); scanning scenes instead.')
        if self.sql is not None:
            from stash_ai.cleanup_sql import run_sql_cleanup
            self.report.write('Cleanup by indexed SELECT: ID cursors, no global marker counts; writes use Stash API.')
            if self.store.cleanup_pending:
                self.stats['cleanup_resumed'] = 1
            self.scene_ids, self.cleanup_stash = run_sql_cleanup(
                app, self.sql, self.store, self.settings, args, self.core_policies, self.scene_policies, self.stats)
        elif self.store.cleanup_pending:
            raise RuntimeError('Global cleanup is interrupted and the required Stash API is unavailable; '
                               'restore API access and rerun the same cleanup.')
        elif not args.scene_id:
            self.report.write('Cleanup scans scenes; global findSceneMarkers filtering is disabled. '
                              'SQL cleanup is faster for large collections.')

    def _cleanup_scenes(self):
        app = self.app
        # An empty candidate list must not be passed to iter_scenes: its empty
        # list convention means the whole library, not zero scenes.
        if self.scene_ids == []:
            return
        if self.sql is not None and hasattr(self.stash, 'configure_merge_reads'):
            # Cleanup uses the same snapshots as merge-only; media paths are unused.
            self.stash.configure_merge_reads(self.report, merge_only=True)
        self.resolver = app.TagResolver(self.cleanup_stash, source=merge.AI, report=self.report.detail,
                                        create_missing=app.CREATE_MISSING)
        bar = tqdm(self.cleanup_stash.iter_scenes(self.scene_ids), desc="Affected scene cleanup", unit="scene")
        for current in bar:
            scene_id = str(current["id"])
            try:
                self.stats.update(cleanup_scene(
                    app, self.cleanup_stash, self.resolver, self.store, current, self.settings, self.selected,
                    self.args, self.core_policies, self.scene_policies, self.scoped_tags, self.report))
                self.stats["processed"] += 1
            except Exception as exc:
                self.stats["errors"] += 1
                self.report.issue('cleanup', '', scene_id, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
            self.report.progress(bar, self.stats)

    def _prune(self):
        if self.stats['errors']:
            return
        app, args = self.app, self.args
        from stash_ai.exclusions import prune_excluded_definitions
        if getattr(args, 'prune_empty_tags', False):
            from stash_ai.prune import prune_tags
            self.stats.update(self.sql.prune(prefixes=args.prune_prefixes, skip_marker_groups=app.CREATE_MARKERS)
                              if self.sql is not None else prune_tags(
                                  self.stash, args.prune_prefixes, report=self.report.detail,
                                  skip_marker_groups=app.CREATE_MARKERS))
        elif not self.stats['cleanup_fast_path'] or len({frozenset(v) for v in self.scene_policies.values()}) > 1:
            if self.sql is not None and app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
                self.stats.update(self.sql.prune(self.scoped_tags))
            else:
                self.stats.update(prune_excluded_definitions(
                    app, self.stash, self.scoped_tags, self.report.detail))
        if app.CREATE_MARKERS:
            from stash_ai.marker_groups import for_resolver, prune_groups
            delta = Counter(getattr(self.cleanup_stash, 'marker_group_usage_delta', {}))
            if self.resolver is not None:
                delta.update(for_resolver(app, self.resolver).usage_delta)
            candidates = self.sql.marker_group_candidates(delta) if self.sql is not None else None
            self.stats.update(prune_groups(
                self.stash, self.report.detail, candidate_ids=candidates, usage_delta=delta))
