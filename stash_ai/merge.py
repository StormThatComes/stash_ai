"""Exact-label fusion, interval union and durable per-source evidence.

No inference adapter depends on this module. The caller supplies the application
module so CLI and imported use share the same configuration and Stash client.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math

from stash_ai.domain import Cue, FramePrediction
from stash_ai.models import ModelSource, SOURCES
from stash_ai.runtime import stage
from stash_ai.occurrences import OCCURRENCE_TAG
from stash_ai.state import KINDS, StateStore, scope_for

AI = ModelSource("ai", "AI", "ai:", "ai:marker:stash-taggers", "[AI] ", "[AI hierarchy v1]")


def now():
    return datetime.now(timezone.utc).isoformat()


def raw_label(value):
    if (not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value
            or AI.prefix + value in (AI.owner_tag, OCCURRENCE_TAG) or value.startswith("category:")):
        raise ValueError(f"Invalid or reserved merged label: {value!r}")
    return value


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


def marker_contributions(app, records, core_policies, *, index=None, metrics=None, cache=None, tag_policies=None,
                         allow_unknown_blocklist=False, allow_unknown_retention=False,
                         legacy_blocklist=frozenset()):
    from stash_ai.fusion import FrameIndex
    from stash_ai.thresholds import marker_context
    app = marker_context(app, records)
    if index is not None:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies,
                                     allow_unknown_blocklist=allow_unknown_blocklist,
                                     allow_unknown_retention=allow_unknown_retention,
                                     legacy_blocklist=legacy_blocklist)
    with FrameIndex(app, records, core_policies, cache=cache, tag_policies=tag_policies) as index:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies,
                                     allow_unknown_blocklist=allow_unknown_blocklist,
                                     allow_unknown_retention=allow_unknown_retention,
                                     legacy_blocklist=legacy_blocklist)


def _marker_contributions(app, records, core_policies, index, metrics, tag_policies, *,
                          allow_unknown_blocklist=False, allow_unknown_retention=False,
                          legacy_blocklist=frozenset()):
    from stash_ai.temporal import consensus_seeds, build_options, build_markers as temporal_markers
    from stash_ai.exclusions import policies_or_empty, core_policies_with_global
    from stash_ai.thresholds import validate_marker_cache
    validate_marker_cache(app, records, allow_unknown_blocklist=allow_unknown_blocklist,
                          allow_unknown_retention=allow_unknown_retention,
                          legacy_blocklist=legacy_blocklist)
    tag_policies = policies_or_empty(tag_policies if tag_policies is not None else getattr(index, 'tag_policies', None))
    effective_cores = core_policies_with_global(core_policies, tag_policies)
    report = getattr(index, 'report', None)
    with stage(report, 'marker_seeds'):
        seeds = consensus_seeds(records, effective_cores, app.TEMPORAL_POLICY,
                                thresholds=app.MARKER_THRESHOLDS)
    signatures = index.signatures
    spans = []
    with stage(report, 'marker_build'):
        for model, segments in records.items():
            for segment in segments:
                if segment["mode"] == "frames":
                    frames = [FramePrediction(Cue(frame["start"], frame["end"], "", (0, 0, 1, 1)),
                               tuple((raw_label(tag), score) for tag, score in frame["tags"]
                                     if tag not in tag_policies[model]))
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
                               "blocklist": []}
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
                    if item["tag"] not in effective_cores[model]:
                        from stash_ai.fusion import prune_blocked_extras
                        spans.append(prune_blocked_extras({**item, "source": model, "meta": segment["meta"]},
                                                         tag_policies[model]))
    from stash_ai.fusion import enrich
    with stage(report, 'marker_fusion'):
        return enrich(app, records, core_policies, spans, signatures, index, metrics)


def overlaps(marker, wanted, tolerance):
    start, end = marker["seconds"], marker.get("end_seconds")
    end = start if end is None else end
    target_end = wanted.get("end_seconds")
    target_end = wanted["seconds"] if target_end is None else target_end
    return (max(start, wanted["seconds"]) <= min(end, target_end)
            or abs(start - wanted["seconds"]) <= tolerance)


def capture_outputs(app, config, vocabulary, cues, predictions, frames, analysis=None):
    from stash_ai.retention import marker_cache_floor
    cue_data = [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]
    meta = {"origin": "inference", "captured_at": now(), "repo": config.source.repo,
            "revision": config.source.revision, "vocabulary_sha256": app.vocabulary_digest(vocabulary.labels),
            "input_cues_sha256": hashlib.sha256(json.dumps(cue_data, ensure_ascii=False).encode()).hexdigest(),
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k_frames": app.TOP_K_FRAMES}
    if app.TEMPORAL_POLICY.enabled:
        meta.update(analysis_format=3,
                    scene_support_floor=config.scene_threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    if app.TEMPORAL_POLICY.enabled or config.marker_cache_floor is not None:
        meta['marker_score_floor'] = marker_cache_floor(config, app.TEMPORAL_POLICY)
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
        # Empty legacy metadata proves no labels were discarded before caching.
        outputs["markers"] = {"mode": "frames", "meta": {
            **meta, "marker_tag_blocklist": []}, "items": [
            dict(start=frame.cue.start, end=frame.cue.end, tags=list(frame.tags)) for frame in frames]}
    return outputs


def has_outputs(app, store, scene_id, model):
    from stash_ai.progress import completed, output_kinds
    return completed(store, scene_id, model, output_kinds(app))
