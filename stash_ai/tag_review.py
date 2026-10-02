"""Offline review priorities from replayed scene tags and marker cores.

Signals are heuristics, not correctness judgments. Scores stay per model.
Only explicitly retained, compatible peer observations can prove disagreement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from tqdm import tqdm

from stash_ai.taggers import SOURCES

REASONS = {
    "weak_or_borderline": "Best retained score is at most the applicable threshold plus the review margin.",
    "isolated_peak": "A scene tag qualifies through a strong peak with only one high sample.",
    "model_disagreement": "A compatible model explicitly retained this label but it does not qualify under that model's own scene rule.",
    "long_marker": "The merged marker range reaches the configured duration.",
}


@dataclass(frozen=True)
class ReviewOptions:
    margin: float = .05
    long_marker_seconds: float = 300.0
    examples: int = 3


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _scene_ids(store, requested):
    table = store.table("results")
    present = {row[0] for row in store.connection.execute(
        f"SELECT DISTINCT scene FROM {table} WHERE scope=? ORDER BY scene", (store.scope,))}
    if requested:
        wanted = list(dict.fromkeys(map(str, requested)))
        missing = sorted(set(wanted) - present)
        if missing:
            raise ValueError("No saved evidence for scene(s): " + ", ".join(missing))
        return wanted
    return sorted(present)


def _candidate(rows, model, tag):
    if model not in SOURCES:
        raise ValueError("Unknown model in review evidence.")
    return rows.setdefault((model, tag), {"reasons": set(), "evidence": []})


def _flag(row, reason, detail):
    row["reasons"].add(reason)
    # Keep scene-local evidence bounded even when a tag has hundreds of ranges.
    key = reason, detail["kind"], detail.get("peer_model")
    if not any((item["reason"], item["kind"], item.get("peer_model")) == key
               for item in row["evidence"]) and len(row["evidence"]) < 8:
        row["evidence"].append({"reason": reason, **detail})


def _scene_signals(app, store, scene, hits, rows, options, scene_policies):
    from stash_ai.merge import marker_segment_signature
    from stash_ai.scene_cache import views
    from stash_ai.temporal import scene_reason

    _, projected = views(app, store.records(scene, "scene", borrow=True))
    # Index retained labels once per scene, rather than scanning the vocabulary
    # for every accepted tag. Absence from this map is never a negative vote.
    peers = defaultdict(list)
    metadata = {}
    for model, segments in projected.items():
        for segment in segments:
            meta = segment["meta"]
            metadata[model, marker_segment_signature(segment)] = meta
            identity = meta.get("frame_input_signature")
            if identity and meta.get("origin") == "inference":
                for entry in segment["items"]:
                    peers[identity, entry["tag"]].append((model, meta, entry))

    for hit in hits:
        shared = hit.get("derived") == "shared_scene_frames"
        sources = hit.get("source_evidence", {}) if shared else {hit["source"]: hit}
        for model, own in sources.items():
            row = _candidate(rows, model, hit["tag"])
            if shared:
                metas = [metadata[model, signature]
                         for signature in hit["scene_context"].get(model, ())
                         if (model, signature) in metadata]
                thresholds = {meta.get("scene_threshold") for meta in metas}
                threshold = next(iter(thresholds)) if len(thresholds) == 1 else None
                score, stats = own.get("max_score"), None
            else:
                threshold = hit["meta"].get("scene_threshold")
                stats = hit.get("statistics")
                score = stats.get("max_score") if stats else hit.get("score")
            cutoff = threshold
            if not shared and hit.get("qualification") == "strong_peak" and _score(threshold):
                cutoff += (1 - threshold) * app.TEMPORAL_POLICY.scene_strong_margin
            detail = dict(kind="scene", score=score, threshold=cutoff,
                          qualification=hit.get("qualification", "aggregate"))
            if stats and "peak_time" in stats:
                detail["time"] = stats["peak_time"]
            if _score(score) and _score(cutoff) and score <= cutoff + options.margin + 1e-12:
                _flag(row, "weak_or_borderline", detail)
            if (not shared and hit.get("qualification") == "strong_peak" and stats
                    and stats.get("high_hits") == 1 and stats.get("sampled_frames", 0) >= 2):
                _flag(row, "isolated_peak", {**detail, "high_hits": 1,
                                             "sampled_frames": stats["sampled_frames"]})
            if shared or hit["meta"].get("origin") != "inference":
                continue
            identity = hit["meta"].get("frame_input_signature")
            if not identity:
                continue
            comparisons = defaultdict(list)
            for peer, meta, entry in peers.get((identity, hit["tag"]), ()):
                if (peer != model and hit["tag"] not in scene_policies[peer]
                        and _score(entry.get("score")) and _score(meta.get("scene_threshold"))):
                    comparisons[peer].append((meta, entry))
            for peer, values in comparisons.items():
                if all(scene_reason(entry["score"], entry.get("statistics"),
                                    meta["scene_threshold"], app.TEMPORAL_POLICY) is None
                       for meta, entry in values):
                    meta, entry = max(values, key=lambda value: value[1]["score"])
                    _flag(row, "model_disagreement", {
                        **detail, "peer_model": peer, "peer_score": entry["score"],
                        "peer_threshold": meta["scene_threshold"]})


def _supporting_scores(app, contribution, tag, segments, postings):
    from stash_ai.temporal import marker_floor
    from stash_ai.thresholds import marker_threshold
    from stash_ai.timeline import Series

    identity = contribution["meta"].get("frame_input_signature")
    if not identity:
        return
    for model in contribution.get("support_sources", ()):
        if model == contribution["source"]:
            continue
        for signature in contribution.get("support_context", {}).get(model, ()):
            key = model, signature
            segment = segments.get(key)
            if (segment is None or segment["mode"] != "frames"
                    or segment["meta"].get("frame_input_signature") != identity):
                continue
            threshold = marker_threshold(segment)
            if not _score(threshold):
                continue
            if key not in postings:
                series = {}
                floor = marker_floor(app.TEMPORAL_POLICY, threshold)
                for position, frame in enumerate(sorted(segment["items"], key=lambda row: row["start"])):
                    for label, score in frame["tags"]:
                        if _score(score) and score >= floor:
                            if label not in series:
                                series[label] = Series()
                            series[label].append(position, frame["start"], frame["end"], score)
                postings[key] = series
            series = postings[key].get(tag)
            if series is None:
                continue
            lo = bisect_left(series.starts, contribution["start"])
            end = contribution["end"]
            hi = (bisect_right(series.starts, contribution["start"]) if end is None
                  else bisect_left(series.starts, end))
            if lo < hi:
                yield model, max(series.scores[lo:hi]), threshold


def _marker_signals(app, store, scene, spans, rows, options):
    from stash_ai.merge import marker_segment_signature, union_intervals
    from stash_ai.occurrences import is_occurrence
    from stash_ai.performance import digest
    from stash_ai.thresholds import marker_threshold

    effective, by_meta, segments_by_signature, postings = {}, defaultdict(set), {}, {}
    for model, segments in store.records(scene, "markers", borrow=True).items():
        for segment in segments:
            threshold = marker_threshold(segment)
            key = model, marker_segment_signature(segment)
            effective[key] = threshold
            segments_by_signature[key] = segment
            by_meta[model, digest(segment["meta"])].add(threshold)

    for marker in union_intervals(spans, borrow=True):
        if is_occurrence(marker):
            continue  # Its scene qualification is reviewed above.
        duration = (marker["end"] - marker["start"]) if marker["end"] is not None else None
        for contribution in marker["contributors"]:
            model, tag = contribution["source"], marker["tag"]
            threshold = effective.get((model, contribution.get("evidence_signature")))
            if threshold is None:
                values = by_meta.get((model, digest(contribution["meta"])), set())
                threshold = next(iter(values)) if len(values) == 1 else None
            score = contribution.get("score")
            witnesses = [(model, score, threshold, False)]
            witnesses.extend((peer, score, threshold, True) for peer, score, threshold in
                             _supporting_scores(app, contribution, tag, segments_by_signature, postings))
            for source, score, threshold, support_only in witnesses:
                row = _candidate(rows, source, tag)
                detail = dict(kind="marker", start=marker["start"], end=marker["end"],
                              score=score, threshold=threshold, support_only=support_only)
                if _score(score) and _score(threshold) and score <= threshold + options.margin + 1e-12:
                    _flag(row, "weak_or_borderline", detail)
                if duration is not None and duration >= options.long_marker_seconds:
                    _flag(row, "long_marker", {**detail, "duration_seconds": duration})


def build_report(app, store, scenes, core_policies, scene_policies, options=ReviewOptions()):
    from stash_ai import merge

    pending = store.pending_publication_scenes()
    refresh = {row[0] for row in store.connection.execute(
        f"SELECT DISTINCT scene FROM {store.table('refresh_requests')} WHERE scope=?", (store.scope,))}
    inspected = {model: set() for model in SOURCES}
    flagged = {model: set() for model in SOURCES}
    model_counts = {model: Counter() for model in SOURCES}
    model_reasons = {model: Counter() for model in SOURCES}
    tags, skipped, skip_counts = {}, [], Counter()
    totals = Counter(selected_scenes=len(scenes))
    all_tags, review_tags = set(), set()

    for scene in tqdm(scenes, desc="Tag review", unit="scene"):
        reason = ("pending_refresh" if scene in refresh else
                  "pending_publication" if scene in pending else None)
        message = None
        if reason is None:
            try:
                hits, spans = merge.evidence(app, store, scene, core_policies, scene_policies)
                rows = {}
                _scene_signals(app, store, scene, hits, rows, options, scene_policies)
                _marker_signals(app, store, scene, spans, rows, options)
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                reason, message = "unusable_evidence", str(exc)
        if reason is not None:
            skip_counts[reason] += 1
            if len(skipped) < 20:
                skipped.append({"scene_id": scene, "reason": reason,
                                **({"message": message} if message else {})})
            continue

        totals["analyzed_scenes"] += 1
        flagged_models, flagged_labels = set(), set()
        all_tags.update(tag for _, tag in rows)
        totals["scene_tag_pairs_reviewed"] += len({tag for _, tag in rows})
        for (model, tag), row in rows.items():
            inspected[model].add(tag)
            model_counts[model]["scene_tag_pairs_reviewed"] += 1
            if not row["reasons"]:
                continue
            flagged[model].add(tag)
            flagged_models.add(model)
            flagged_labels.add(tag)
            model_counts[model]["scene_tag_pairs_flagged"] += 1
            model_reasons[model].update(row["reasons"])
            entry = tags.setdefault((model, tag), dict(
                model=model, tag="ai:" + tag, scene_count=0, reasons=Counter(), examples=[]))
            entry["scene_count"] += 1
            entry["reasons"].update(row["reasons"])
            if len(entry["examples"]) < options.examples:
                entry["examples"].append(dict(scene_id=scene, reasons=sorted(row["reasons"]),
                                              evidence=row["evidence"]))
        for model in flagged_models:
            model_counts[model]["scenes_flagged"] += 1
        review_tags.update(flagged_labels)
        totals["scene_tag_pairs_flagged"] += len(flagged_labels)
        totals["scenes_flagged"] += bool(flagged_labels)

    total_keys = ("selected_scenes", "analyzed_scenes", "scenes_flagged",
                  "scene_tag_pairs_reviewed", "scene_tag_pairs_flagged")
    return {
        "format": 1,
        "mode": "tag-review",
        "semantics": {
            "scope": "Replayed scene labels and marker cores under saved effective thresholds and selected exclusion/temporal policies; no live Stash inspection.",
            "signals_are_not_errors": True,
            "model_scores_are_not_averaged": True,
            "pair_count": "One scene/tag per model, deduplicated across history, scene tags and marker ranges.",
            "global_counts": "Deduplicated across models; per-model counts must not be summed as unique global counts.",
            "reason_counts_overlap": True,
            "disagreement": "Requires explicit retained peer evidence for the same raw label and frame input; missing labels/models are not negative votes.",
            "examples_limit": options.examples,
        },
        "options": asdict(options),
        "reason_definitions": REASONS,
        "totals": {**{key: totals[key] for key in total_keys},
                   "unique_tags_reviewed": len(all_tags), "unique_tags_flagged": len(review_tags),
                   "skipped_scenes": sum(skip_counts.values())},
        "models": [
            dict(model=model, unique_tags_reviewed=len(inspected[model]),
                 unique_tags_flagged=len(flagged[model]),
                 scene_tag_pairs_reviewed=model_counts[model]["scene_tag_pairs_reviewed"],
                 scene_tag_pairs_flagged=model_counts[model]["scene_tag_pairs_flagged"],
                 scenes_flagged=model_counts[model]["scenes_flagged"],
                 reasons={reason: model_reasons[model][reason] for reason in REASONS})
            for model in SOURCES],
        "tags": sorted(tags.values(), key=lambda entry: (
            -entry["scene_count"], -sum(entry["reasons"].values()), entry["model"], entry["tag"])),
        "skipped_by_reason": dict(sorted(skip_counts.items())),
        "skipped_scene_examples": skipped,
        "warnings": ["Pending global cleanup; report replays the selected policies."]
                    if store.cleanup_pending else [],
    }


def format_summary(report):
    totals = report["totals"]
    lines = [
        f"Tag review: {totals['unique_tags_flagged']:,} distinct tags in "
        f"{totals['scene_tag_pairs_flagged']:,} scene/tag pairs; "
        f"{totals['analyzed_scenes']:,} scenes analyzed, {totals['skipped_scenes']:,} skipped.",
        "Heuristic review priorities, not confirmed tagging errors.",
        "model     tags flagged  scene/tag pairs  scenes flagged",
    ]
    for row in report["models"]:
        lines.append(f"{row['model']:<9} {row['unique_tags_flagged']:>12,} "
                     f"{row['scene_tag_pairs_flagged']:>16,} {row['scenes_flagged']:>15,}")
    if report["skipped_by_reason"]:
        lines.append("Skipped: " + ", ".join(
            f"{reason}={count}" for reason, count in report["skipped_by_reason"].items()))
    lines.extend("Warning: " + warning for warning in report["warnings"])
    return "\n".join(lines)


def run(app, args):
    from stash_ai.state import StateStore, scope_for
    from stash_ai.temporal import load_policy

    try:
        options = ReviewOptions(
            margin=.05 if args.tag_review_margin is None else args.tag_review_margin,
            long_marker_seconds=300.0 if args.tag_review_long_marker_seconds is None
            else args.tag_review_long_marker_seconds,
            examples=3 if args.tag_review_examples is None else args.tag_review_examples)
        local = SimpleNamespace(**{**vars(app), "APPLY_SCENE_TAGS": True, "CREATE_MARKERS": True,
                                   "TEMPORAL_POLICY": load_policy(app, args.temporal_policy, not args.no_temporal)})
        core = app.load_exclusion_policy(args.marker_core_exclusions, "excluded_core_tags")
        tags = app.load_exclusion_policy(args.scene_tag_exclusions, "excluded_scene_tags")
        if not Path(args.state_db).expanduser().is_file():
            raise FileNotFoundError(f"State database does not exist: {args.state_db}")
        with StateStore(args.state_db, scope_for(app.STASH_URL), True,
                        cache_bytes=int(args.ram_cache_gb * 2**30),
                        memory_budget_bytes=int(args.ram_budget_gb * 2**30),
                        allow_cleanup=True) as store:
            scenes = _scene_ids(store, args.scene_id)
            if not args.scene_id:
                store.preload_sources()
            report = build_report(local, store, scenes, core, tags, options)
        destination = Path(args.tag_review_report).expanduser().resolve()
        payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        print(format_summary(report))
        print(f"Tag review report written to {destination}. No Stash requests or inference were performed.")
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print("Tag review report failed: " + str(exc))
        return 1
