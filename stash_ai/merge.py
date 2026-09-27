"""Exact-label fusion, interval union and durable per-source evidence.

No inference adapter depends on this module. The caller supplies the application
module so CLI and imported use share the same configuration and Stash client.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math

from stash_ai.domain import Cue, FramePrediction, TagHierarchy
from stash_ai.taggers import ModelSource, SOURCES
from stash_ai.runtime import stage
from stash_ai.occurrences import OCCURRENCE_TAG
from stash_ai.state import APPLICATION_ID, KINDS, StateStore, scope_for
from stash_ai.state_codec import encode, decode  # compatibility re-export

AI = ModelSource("ai", "AI", "ai:", "ai:marker:stash-taggers", "[AI] ", "[AI hierarchy v1]")


def now():
    return datetime.now(timezone.utc).isoformat()


def raw_label(value):
    if (not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value
            or AI.prefix + value in (AI.owner_tag, OCCURRENCE_TAG) or value.startswith("category:")):
        raise ValueError(f"Invalid or reserved merged label: {value!r}")
    return value


def legacy_label(name):
    for key, source in SOURCES.items():
        if name.startswith(source.prefix) and name != source.owner_tag:
            label = name[len(source.prefix):]
            if label and not label.startswith(("category:", "marker:")):
                return key, raw_label(label)
    return None


def source_codes(sources):
    return [source.prefix.rstrip(":") for key, source in SOURCES.items() if key in sources]


def union_intervals(contributions, *, borrow=False):
    """One component per exact label. Positive gaps are never bridged here.

    Unknown legacy end times remain points, unless covered by a known interval.
    Scores are kept on contributors; there is no artificial ensemble probability.
    """
    grouped = defaultdict(list)
    memo = {}
    for contribution in contributions:
        item = contribution if borrow else deepcopy(contribution, memo)
        tag = raw_label(item["tag"])
        if item["source"] not in SOURCES:
            raise ValueError("Unknown interval source.")
        start, end = item["start"], item["end"]
        if (isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0
                or (end is not None and (isinstance(end, bool) or not isinstance(end, (int, float))
                    or not math.isfinite(end) or end < start))):
            raise ValueError(f"Invalid marker interval for {tag!r}.")
        grouped[tag].append(item)
    merged = []
    for tag, items in sorted(grouped.items()):
        items.sort(key=lambda item: (item["start"], item["end"] if item["end"] is not None else item["start"], item["source"]))
        components = []
        for item in items:
            end = item["end"] if item["end"] is not None else item["start"]
            if not components or item["start"] > components[-1]["bound"]:
                components.append(dict(tag=tag, start=item["start"], bound=end, contributors=[]))
            component = components[-1]
            component["bound"] = max(component["bound"], end)
            if item not in component["contributors"]:
                component["contributors"].append(item)
        for component in components:
            contributors = component["contributors"]
            bound = component.pop("bound")
            component["end"] = bound if any(item["end"] is not None for item in contributors) else None
            component["sources"] = [key for key in SOURCES if any(item["source"] == key or
                                    key in item.get("support_sources", []) for item in contributors)]
            component["extra_tags"] = sorted({raw_label(tag) for item in contributors for tag in item.get("extra_tags", [])}
                                             - {tag})
            component["extra_ids"] = sorted({str(i) for item in contributors for i in item.get("extra_ids", [])})
            from stash_ai.temporal import merged_extras
            merged_extras(component)
            component["extra_tags"] = sorted({raw_label(label) for label in component["extra_tags"]})
            merged.append(component)
    return sorted(merged, key=lambda item: (item["start"], item["tag"]))


def combined_hierarchy(app, args, tag_policies=None):
    """Compose the four local YAMLs into one stable ai: category namespace."""
    if args.no_hierarchy:
        return None
    from stash_ai.exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    prefix, root = "ai:category:", "ai:category:AI"
    leaves = defaultdict(set)
    categories = {root: ()}
    for key, source in SOURCES.items():
        path = args.hierarchy if key == "joytag" else getattr(args, key + "_hierarchy")
        data = app.load_unique_yaml(path, "Hierarchy")
        if not isinstance(data, dict) or not isinstance(data.get("groups"), dict):
            raise ValueError(f"Invalid hierarchy: {path}")
        groups = data["groups"].values()
        if any(not isinstance(group, list) or any(not isinstance(label, str) for label in group) for group in groups):
            raise ValueError(f"Invalid hierarchy labels: {path}")
        labels = sorted({label for group in groups for label in group})
        hierarchy = app.load_tag_hierarchy(path, labels, source)
        def converted(name):
            return root if not hierarchy.category_parents[name] else prefix + name[len(hierarchy.category_prefix):]
        for name, parents in hierarchy.category_parents.items():
            mapped, wanted = converted(name), tuple(converted(parent) for parent in parents)
            if mapped in categories and categories[mapped] != wanted:
                raise ValueError(f"Conflicting AI hierarchy category: {mapped}")
            categories[mapped] = wanted
        for label, parents in hierarchy.leaf_parents.items():
            raw_label(label)
            if label in tag_policies[key]:
                continue
            leaves[label].update(converted(parent) for parent in parents)
    fallbacks = {prefix + suffix for suffix in ("Other / Unclassified", "General", "Uncategorized")}
    for parents in leaves.values():
        if parents - fallbacks:
            parents.difference_update(fallbacks)
    if set(categories) & {AI.prefix + label for label in leaves}:
        raise ValueError("A merged hierarchy category collides with a predicted tag.")
    return TagHierarchy({label: tuple(sorted(parents)) for label, parents in leaves.items()},
                            categories, prefix, "replace_managed", AI.hierarchy_description)


def bootstrap_legacy(app, store, scene):
    """Capture all known source namespaces before migrating any associations."""
    scene_id = str(scene["id"])
    for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)):
        if not enabled:
            continue
        existing = store.sources(scene_id, kind)
        if kind == "markers" and not existing and any(app.owned_marker(marker, AI) for marker in scene.get("scene_markers", [])):
            raise RuntimeError("Merged markers exist without their source state. Restore the original --state-db before synchronization.")
        from stash_ai.performance import freeze
        imported = {freeze(item) for item in store.imported_legacy(scene_id, kind, borrow=True)}
        originals = []
        by_model = defaultdict(list)
        if kind == "scene":
            for tag in scene.get("tags", []):
                parsed = legacy_label(tag["name"])
                if parsed:
                    model, label = parsed
                    originals.append(tag)
                    if model not in existing or freeze(tag) not in imported:
                        by_model[model].append({"tag": label, "score": None, "legacy_tag_id": str(tag["id"])})
        else:
            for marker in scene.get("scene_markers", []):
                parsed = legacy_label((marker.get("primary_tag") or {}).get("name", ""))
                if not parsed or not app.owned_marker(marker, SOURCES[parsed[0]]):
                    continue
                model, label = parsed
                originals.append(marker)
                if model in existing and freeze(marker) in imported:
                    continue
                extras, manual_ids = set(), set()
                for tag in marker.get("tags", []):
                    extra = legacy_label(tag["name"])
                    if extra:
                        extras.add(extra[1])
                    elif tag["name"] not in {source.owner_tag for source in SOURCES.values()}:
                        manual_ids.add(str(tag["id"]))
                by_model[model].append(dict(tag=label, start=marker["seconds"], end=marker.get("end_seconds"),
                    score=None, hits=None, extra_tags=sorted(extras - {label}), extra_ids=sorted(manual_ids),
                    legacy_marker_id=str(marker["id"])))
        for model, items in by_model.items():
            segment = {"mode": "scene" if kind == "scene" else "spans", "items": items,
                       "meta": {"origin": "legacy", "captured_at": now(), "revision": None}}
            # Fresh separate-mode outputs augment the cache. Only a successful
            # merged-mode inference replaces that model's previous evidence.
            store.put(scene_id, model, {kind: segment}, {kind: False})
        store.remember_legacy(scene_id, kind, originals)


def evidence(app, store, scene_id, core_policies, scene_policies, metrics=None, *, selected_sources=None):
    """Retain source markers and enrich them with compatible shared observations.

    selected_sources is an analysis-only view used for counterfactual reports:
    omitted sources are removed before every fusion/consensus/occurrence stage.
    """
    from stash_ai.temporal import scene_reason
    from stash_ai.fusion import FrameIndex
    from stash_ai.scene import shared_scene_contributions
    report = getattr(store, 'report', None)
    scene_hits, spans = [], []
    need_scene = app.APPLY_SCENE_TAGS or (app.CREATE_MARKERS and app.TEMPORAL_POLICY.enabled
                                        and app.TEMPORAL_POLICY.occurrence_markers)
    scene_records = store.records(scene_id, 'scene', borrow=True) if need_scene else {}
    if selected_sources is not None:
        selected_sources = frozenset(selected_sources)
        scene_records = {source: segments for source, segments in scene_records.items()
                         if source in selected_sources}
    from stash_ai.scene_cache import views, frame_records
    app, scene_records = views(app, scene_records)
    for model, segments in scene_records.items():
        for segment in segments:
            eligible = []
            for entry in segment["items"]:
                label = raw_label(entry["tag"])
                reason = scene_reason(entry["score"], entry.get("statistics"),
                                      segment["meta"].get("scene_threshold", 0.0), app.TEMPORAL_POLICY)
                if label not in scene_policies[model] and reason:
                    eligible.append({**entry, "source": model, "meta": segment["meta"],
                                     **({"qualification": reason} if entry.get("statistics") else {})})
            # Cap each inference pass separately. With replacement disabled,
            # historical detections and their original confidence remain intact.
            eligible.sort(key=lambda item: (-(item["score"] if item["score"] is not None else -1), item["tag"]))
            scene_hits.extend(eligible[:app.MAX_TAGS] if app.MAX_TAGS and segment["meta"]["origin"] != "legacy" else eligible)
    records = store.records(scene_id, 'markers', borrow=True)
    if selected_sources is not None:
        records = {source: segments for source, segments in records.items()
                   if source in selected_sources}
    from stash_ai.thresholds import marker_context
    app = marker_context(app, records)
    if report is not None:
        segments = [segment for values in records.values() for segment in values]
        report.record_counts(input_models=len(records), input_segments=len(segments),
            frame_rows=sum(len(s['items']) for s in segments if s['mode'] == 'frames'),
            retained_detections=sum(len(row['tags']) for s in segments if s['mode'] == 'frames' for row in s['items']))
    with stage(report, 'frame_index'):
        index = FrameIndex(app, records, core_policies, cache=store.cache, tag_policies=scene_policies)
        index.report = report
    with index:
        if app.CREATE_MARKERS:
            spans = marker_contributions(app, records, core_policies, index=index, metrics=metrics, tag_policies=scene_policies)
        if need_scene:
            with stage(report, 'scene_fusion'):
                scene_frames = frame_records(scene_records, records)
                with FrameIndex(app, scene_frames, core_policies, cache=store.cache, tag_policies=scene_policies) as scene_index:
                    scene_hits.extend(shared_scene_contributions(app, scene_index, scene_records, scene_policies, scene_hits, metrics))
    if app.CREATE_MARKERS:
        from stash_ai.occurrences import occurrence_contributions
        report = getattr(store, 'report', None)
        with report.measure('occurrence_compute') if report is not None else nullcontext():
            spans.extend(occurrence_contributions(app, scene_hits, frame_records(scene_records, records), spans, core_policies, scene_policies, metrics))
    return sorted(scene_hits, key=lambda item: (item["tag"], item["source"])), spans


def marker_segment_signature(segment):
    # Cleanup must use exactly the evidence that supported published markers.
    from stash_ai.performance import StoredSegment, segment_digest
    from stash_ai.thresholds import OVERRIDE
    from stash_ai.scene_cache import SCENE_OVERRIDE
    if isinstance(segment, StoredSegment) and hasattr(segment, "_evidence_digest"):
        return segment._evidence_digest
    if OVERRIDE not in segment and SCENE_OVERRIDE not in segment:
        return segment_digest(segment)
    signature = segment_digest({k: v for k, v in segment.items() if k not in (OVERRIDE, SCENE_OVERRIDE)})
    if isinstance(segment, StoredSegment):
        segment._evidence_digest = signature
    return signature


def marker_contributions(app, records, core_policies, *, index=None, metrics=None, cache=None, tag_policies=None):
    from stash_ai.fusion import FrameIndex
    from stash_ai.thresholds import marker_context
    app = marker_context(app, records)
    if index is not None:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies)
    with FrameIndex(app, records, core_policies, cache=cache, tag_policies=tag_policies) as index:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies)


def _marker_contributions(app, records, core_policies, index, metrics, tag_policies):
    from stash_ai.temporal import consensus_seeds, build_options, build_markers as temporal_markers
    from stash_ai.exclusions import policies_or_empty, core_policies_with_global
    tag_policies = policies_or_empty(tag_policies if tag_policies is not None else getattr(index, 'tag_policies', None))
    effective_cores = core_policies_with_global(core_policies, tag_policies)
    report = getattr(index, 'report', None)
    with stage(report, 'marker_seeds'):
        seeds = consensus_seeds(records, effective_cores, app.MARKER_TAG_BLOCKLIST, app.TEMPORAL_POLICY,
                                thresholds=app.MARKER_THRESHOLDS)
    signatures = index.signatures
    spans = []
    with stage(report, 'marker_build'):
        for model, segments in records.items():
            for segment in segments:
                if segment["mode"] == "frames":
                    frames = [FramePrediction(Cue(frame["start"], frame["end"], "", (0, 0, 1, 1)),
                               tuple((raw_label(tag), score) for tag, score in frame["tags"]
                                     if tag not in app.MARKER_TAG_BLOCKLIST and tag not in tag_policies[model]))
                              for frame in segment["items"]]
                    from stash_ai.thresholds import marker_threshold
                    kwargs = dict(threshold=marker_threshold(segment, app.MARKER_THRESHOLDS), seeds=seeds,
                                  identity=segment["meta"].get("frame_input_signature"))
                    defer_extras = app.TEMPORAL_POLICY.shared_marker_extras and signatures[id(segment)] in index.by_signature
                    markers, _ = (temporal_markers(app, frames, effective_cores[model], **kwargs,
                                  collect_extras=not defer_extras,
                                  extra_group=index.by_signature.get(signatures[id(segment)], signatures[id(segment)]),
                                  extra_cache=index.cache) if app.TEMPORAL_POLICY.enabled else
                                  app.build_markers(frames, effective_cores[model], **kwargs))
                    options = {**build_options(app), "core_exclusions": {k: sorted(v) for k, v in core_policies.items()},
                               "tag_exclusions": {k: sorted(v) for k, v in tag_policies.items()},
                               "blocklist": sorted(app.MARKER_TAG_BLOCKLIST)}
                    values = [dict(tag=marker.tag, start=marker.start, end=marker.end, score=marker.score,
                                   hits=marker.hits, extra_tags=list(marker.extra_tags),
                                   **(dict(extra_evidence=list(marker.extra_evidence), strong_hits=marker.strong_hits,
                                           support_sources=list(marker.support_sources), extra_support=marker.extra_support, build_options=options,
                                           evidence_signature=signatures[id(segment)],
                                           support_context={source: sorted({signatures[id(s)] for s in records[source]
                                               if s["meta"].get("frame_input_signature") == segment["meta"].get("frame_input_signature")})
                                               for source in marker.support_sources})
                                      if app.TEMPORAL_POLICY.enabled else {})) for marker in markers]
                elif segment["mode"] == "spans":
                    values = segment["items"]
                else:
                    raise ValueError("Unsupported stored marker evidence.")
                for item in values:
                    if item["tag"] not in effective_cores[model] and item["tag"] not in app.MARKER_TAG_BLOCKLIST:
                        from stash_ai.fusion import prune_blocked_extras
                        spans.append(prune_blocked_extras({**item, "source": model, "meta": segment["meta"]},
                                                         app.MARKER_TAG_BLOCKLIST | tag_policies[model]))
    from stash_ai.fusion import enrich
    with stage(report, 'marker_fusion'):
        return enrich(app, records, core_policies, spans, signatures, index, metrics)


def cleanup_support(app, store, scene_id, spans, selected, core_policies, tag_policies=None):
    """Re-evaluate revoked votes, bounded by the already-published source ranges."""
    from types import SimpleNamespace
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
        own_blocked = app.MARKER_TAG_BLOCKLIST | (tag_policies[span['source']] if span['source'] in selected else frozenset())
        if span.get('derived') != 'shared_frames' and span['source'] in selected and (
                span['tag'] in core_policies[span['source']] or span['tag'] in own_blocked):
            continue
        core_affected = any(source in selected and (span["tag"] in core_policies[source] or
                            span["tag"] in tag_policies[source] or span["tag"] in app.MARKER_TAG_BLOCKLIST)
                            for source in span.get("support_sources", []))
        extra_labels = set(span.get('extra_tags', [])) | set(span.get('extra_support', {}))
        extra_sources = set(span.get('frame_context', {})) | {span['source']}
        extras_affected = any(extra_labels & tag_policies[source] for source in extra_sources & selected)
        affected = core_affected or extras_affected
        if not affected:
            cleaned.append(prune_blocked_extras(span, app.MARKER_TAG_BLOCKLIST))
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
                available_by_source[source] = {marker_segment_signature(s): s for s in records.get(source, [])}
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
            context.MARKER_TAG_BLOCKLIST = set(options["blocklist"]) | app.MARKER_TAG_BLOCKLIST
            policies = {source: frozenset(options["core_exclusions"][source]) |
                        (core_policies[source] if source in selected else frozenset()) for source in SOURCES}
            frozen_tags = {source: frozenset(options.get('tag_exclusions', {}).get(source, [])) |
                           (tag_policies[source] if source in selected else frozenset()) for source in SOURCES}
            # Cleanup can revoke old evidence but cannot add newly cached votes.
            rebuilt[key] = (marker_contributions(context, matching, policies, tag_policies=frozen_tags),
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
            clipped = prune_blocked_extras(clipped, app.MARKER_TAG_BLOCKLIST)
            if clipped not in cleaned:
                cleaned.append(clipped)
    return cleaned


def overlaps(marker, wanted, tolerance):
    start, end = marker["seconds"], marker.get("end_seconds")
    end = start if end is None else end
    target_end = wanted.get("end_seconds")
    target_end = wanted["seconds"] if target_end is None else target_end
    return (max(start, wanted["seconds"]) <= min(end, target_end)
            or abs(start - wanted["seconds"]) <= tolerance)


def sync_merged(app, stash, resolver, store, scene, scene_hits, spans, *, migrate=True, cleanup=False, report=None,
                tag_policies=None, publication_key=None):
    """Compatibility wrapper for one-scene merged reconciliation."""
    from stash_ai.merge_scene import MergedSceneSynchronizer
    return MergedSceneSynchronizer(
        app, stash, resolver, store, migrate=migrate, cleanup=cleanup,
        report=report, tag_policies=tag_policies,
        publication_key=publication_key,
    ).sync(scene, scene_hits, spans)

def capture_outputs(app, config, vocabulary, cues, predictions, frames, analysis=None):
    from stash_ai.temporal import marker_floor
    cue_data = [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]
    meta = {"origin": "inference", "captured_at": now(), "repo": config.source.repo,
            "revision": config.source.revision, "vocabulary_sha256": app.vocabulary_digest(vocabulary.labels),
            "input_cues_sha256": hashlib.sha256(json.dumps(cue_data, ensure_ascii=False).encode()).hexdigest(),
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k_frames": app.TOP_K_FRAMES}
    if app.TEMPORAL_POLICY.enabled:
        meta.update(analysis_format=3, marker_score_floor=marker_floor(app.TEMPORAL_POLICY, config.marker_threshold),
                    scene_support_floor=config.scene_threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    outputs = {}
    if app.APPLY_SCENE_TAGS:
        outputs["scene"] = {"mode": "scene", "meta": meta,
                            "items": [dict(tag=tag, score=score,
                                           **({"statistics": analysis["scene_statistics"][tag]} if analysis and
                                              tag in analysis.get("scene_statistics", {}) else {})) for tag, score in predictions]}
        if analysis is not None and 'scene_frames' in analysis:
            outputs['scene']['frames'] = analysis['scene_frames']
            outputs['scene']['meta'] = {**meta, 'scene_frames_format': 1,
                'scene_score_floor': analysis['scene_score_floor']}
    if app.CREATE_MARKERS:
        outputs["markers"] = {"mode": "frames", "meta": meta, "items": [
            dict(start=frame.cue.start, end=frame.cue.end, tags=list(frame.tags)) for frame in frames]}
    return outputs


def has_outputs(app, store, scene_id, model):
    from stash_ai.progress import completed, output_kinds
    return completed(store, scene_id, model, output_kinds(app))


def run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
    from stash_ai.diagnostics import MergeDiagnostics
    from stash_ai.runtime import RunReport
    from tqdm import tqdm
    report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
    args.report = report
    with MergeDiagnostics(report, getattr(args, 'profile_merge', None), dry_run=app.DRY_RUN,
                          merge_only=args.merge_only, force=getattr(args, 'force_merge', False)):
        return _run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies)


def _run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
    """Compatibility wrapper around the stateful merged-run orchestrator."""
    from stash_ai.merge_run import MergedRun
    return MergedRun(app, settings, args, catalogs, hierarchy, core_policies, scene_policies).run()

def run_hierarchy(app, args, catalogs, hierarchy, tag_policies=None):
    from stash_ai.exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    stash = app.create_stash_client(app.DRY_RUN)
    try:
        stash.detect_tag_hierarchy()
        resolver = app.TagResolver(stash, hierarchy, source=AI, report=args.report.detail, create_missing=app.CREATE_MISSING)
        resolver.preload()
        labels = sorted({label for source, vocabulary in catalogs.items() for label in vocabulary.labels
                         if label not in tag_policies[source]})
        for label in labels:
            if args.create_all_tags or AI.prefix + label in resolver.cache:
                resolver.resolve(AI.prefix + label)
        app.print_hierarchy_summary(resolver)
    finally:
        stash.close()
    return 0


def cleanup_scene(app, stash, resolver, store, current, settings, selected, args,
                  core_policies, scene_policies, scoped_tags, report):
    """Reconcile one affected scene; refresh only after actual writes."""
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
                app.owned_marker(marker, AI) for marker in current.get("scene_markers", [])):
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
            result = sync_merged(app, stash, resolver, store, current, clean_hits, clean_spans,
                                 migrate=False, cleanup=True, report=report, tag_policies=scoped_tags)
            stats.update(result)
            if not stash.dry_run and any(result[name] for name in (
                    'scene_tags_updated', 'markers_created', 'markers_updated', 'markers_deleted')):
                current = stash.find_scene(scene_id)
    # Most scenes have no old source-prefixed outputs. Reading the same scene
    # once for every configured model previously dominated no-op cleanup.
    for config in settings:
        result, source = Counter(), config.source
        if app.APPLY_SCENE_TAGS:
            result.update(app.cleanup_scene_tags(stash, current, scene_policies[source.key], source, report.detail))
        if app.CREATE_MARKERS:
            result.update(app.cleanup_scene_marker_cores(stash, current, core_policies[source.key], report.detail,
                                                        source, scene_policies[source.key]))
        stats.update(result)
        if not stash.dry_run and any(result[name] for name in ('scene_tags_cleaned', 'markers_deleted', 'markers_updated')):
            current = stash.find_scene(scene_id)
    return stats


def run_cleanup(app, settings, args, core_policies, scene_policies):
    """Global bans operate on tags; source-specific changes reconcile affected scenes."""
    from tqdm import tqdm
    from stash_ai.cleanup import CleanupClient
    from stash_ai.exclusions import selected_policies, prune_excluded_definitions
    from stash_ai.runtime import RunReport
    selected = {config.source.key for config in settings}
    scoped_tags = selected_policies(scene_policies, selected)
    stats = Counter()
    report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
    wide = not args.scene_id and selected == set(SOURCES)
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN,
                    cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                    allow_cleanup=wide or getattr(args, 'abandon_pending', False), cleanup_preview=True,
                    memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as store:
        if getattr(args, 'abandon_pending', False):
            from stash_ai.recovery import abandon
            stats.update(abandon(store, report.write))
        stash = app.create_stash_client(app.DRY_RUN)
        resolver = None
        try:
            scene_ids, cleanup_stash = args.scene_id or None, stash
            strategy = getattr(args, 'cleanup_strategy', 'auto')
            client = CleanupClient(stash) if strategy in ('auto', 'sql') else None
            sql = None
            if strategy in ('auto', 'sql') and client.sql_available:
                from stash_ai.cleanup_sql import SQLCleanup
                candidate = SQLCleanup(client, args, stats, report)
                try:
                    candidate.probe()
                    sql = candidate
                except Exception as exc:
                    if strategy == 'sql' or store.cleanup_pending:
                        raise
                    report.write('SQL cleanup unavailable (' + app.safe_error(exc, api_key=app.STASH_API_KEY) + '); scanning scenes instead.')
            if sql is not None:
                from stash_ai.cleanup_sql import run_sql_cleanup
                report.write('Cleanup by indexed SELECT: ID cursors, no global marker counts; writes use Stash API.')
                if store.cleanup_pending:
                    stats['cleanup_resumed'] = 1
                scene_ids, cleanup_stash = run_sql_cleanup(app, sql, store, settings, args, core_policies, scene_policies, stats)
            elif strategy == 'sql':
                raise RuntimeError('Stash does not expose the required SQL cleanup API. Use --cleanup-strategy scenes for compatibility.')
            elif store.cleanup_pending:
                raise RuntimeError('Global cleanup is interrupted and the required Stash API is unavailable; '
                                   'restore API access and rerun the same cleanup.')
            elif not args.scene_id:
                report.write('Cleanup scans scenes; global findSceneMarkers filtering is disabled. SQL cleanup is faster for large collections.')
            # An empty candidate list must not be passed to iter_scenes: its empty
            # list convention means the whole library, not zero scenes.
            if scene_ids != []:
                if app.CREATE_MARKERS:
                    stash.detect_marker_capabilities()
                    stash.detect_tag_hierarchy()
                resolver = app.TagResolver(cleanup_stash, source=AI, report=report.detail, create_missing=app.CREATE_MISSING)
                bar = tqdm(cleanup_stash.iter_scenes(scene_ids), desc="Affected scene cleanup", unit="scene")
                for current in bar:
                    scene_id = str(current["id"])
                    try:
                        stats.update(cleanup_scene(app, cleanup_stash, resolver, store, current, settings, selected,
                                                   args, core_policies, scene_policies, scoped_tags, report))
                        stats["processed"] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        report.issue('cleanup', '', scene_id, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
                    report.progress(bar, stats)
            if not stats['errors']:
                if getattr(args, 'prune_empty_tags', False):
                    from stash_ai.prune import prune_tags
                    stats.update(sql.prune(prefixes=args.prune_prefixes, skip_marker_groups=app.CREATE_MARKERS)
                                 if sql is not None else prune_tags(stash, args.prune_prefixes, report=report.detail,
                                                                  skip_marker_groups=app.CREATE_MARKERS))
                elif not stats['cleanup_fast_path'] or len({frozenset(v) for v in scene_policies.values()}) > 1:
                    if sql is not None and app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
                        stats.update(sql.prune(scoped_tags))
                    else:
                        stats.update(prune_excluded_definitions(app, stash, scoped_tags, report.detail))
                if app.CREATE_MARKERS:
                    from stash_ai.marker_groups import for_resolver, prune_groups
                    delta = Counter(getattr(cleanup_stash, 'marker_group_usage_delta', {}))
                    if resolver is not None:
                        delta.update(for_resolver(app, resolver).usage_delta)
                    candidates = sql.marker_group_candidates(delta) if sql is not None else None
                    stats.update(prune_groups(stash, report.detail, candidate_ids=candidates, usage_delta=delta))
        except Exception as exc:
            stats['errors'] += 1
            report.issue('cleanup', '', None, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
        finally:
            stash.close()
        report.finish('Cleanup' + (' (planned)' if app.DRY_RUN else ''), stats, store.cache)
    return int(bool(stats["errors"]))


def explain(app, args, core_policies, scene_policies):
    label = raw_label(args.explain_tag.removeprefix(AI.prefix))
    # Inspection always opens a read-only snapshot, even without --dry-run.
    with StateStore(args.state_db, scope_for(app.STASH_URL), True,
                    cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                    memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as store:
        reports = []
        for scene_id in dict.fromkeys(args.scene_id):
            diagnostics = Counter()
            hits, spans = evidence(app, store, scene_id, core_policies, scene_policies, diagnostics)
            matching = [span for span in spans if span["tag"] == label]
            recorded_sources = {}
            for kind in KINDS:
                sources = {}
                for source, segments in store.records(scene_id, kind).items():
                    matches = []
                    for segment in segments:
                        items = ([dict(start=frame["start"], end=frame["end"], score=score)
                                  for frame in segment["items"] for tag, score in frame["tags"] if tag == label]
                                 if segment["mode"] == "frames" else [entry for entry in segment["items"] if entry["tag"] == label])
                        if items:
                            matches.append({"meta": segment["meta"], "detections": items})
                    if matches:
                        sources[source] = {"excluded": label in scene_policies[source] or (kind == 'markers' and label in core_policies[source]),
                                           "records": matches}
                recorded_sources[kind] = sources
            reports.append({"scene_id": str(scene_id), "tag": AI.prefix + label, 'diagnostics': dict(diagnostics),
                            "scene_contributors": [hit for hit in hits if hit["tag"] == label],
                            "markers": union_intervals(matching), "recorded_sources": recorded_sources,
                            "as_marker_extra": [dict(tag=AI.prefix + marker["tag"], start=marker["start"], end=marker["end"],
                                                     evidence=marker["extra_evidence"][label])
                                                for marker in union_intervals(spans) if label in marker.get("extra_evidence", {})],
                            "pending_sync": any(store.snapshot(scene_id, kind, borrow=True)[1] is not None for kind in KINDS)})
    print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
