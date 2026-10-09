"""Offline review priorities from replayed scene tags and marker cores.

Signals are heuristics, not correctness judgments. Scores stay per model.
Only explicitly retained, compatible peer observations can prove disagreement.
"""
from __future__ import annotations

from .options import REASONS, ReviewOptions
from .scenes import _scene_signals
from .markers import _marker_signals

from dataclasses import asdict
import sqlite3

from tqdm import tqdm

from stash_ai.reports.review.accumulator import ReviewAccumulator
from stash_ai.reports.io import write_json_report
from stash_ai.reports.runtime import report_context, report_policies, report_store, report_memory


def _scene_ids(store, requested):
    from stash_ai.storage.selection import evidence_scenes
    present = evidence_scenes(store, requested)
    if requested:
        wanted = list(dict.fromkeys(map(str, requested)))
        missing = sorted(set(wanted) - set(present))
        if missing:
            raise ValueError("No saved evidence for scene(s): " + ", ".join(missing))
        return wanted
    return sorted(present)


def build_report(app, store, scenes, core_policies, scene_policies, options=ReviewOptions()):
    from stash_ai.evidence import pipeline as merge

    from stash_ai.storage.selection import selected_rows
    from stash_ai.storage.input_checks import selected as input_checks
    pending = store.pending_publication_scenes(scenes)
    refresh = {row[0] for row in selected_rows(store,
        f"SELECT DISTINCT scene FROM {store.table('refresh_requests')} WHERE scope=?", (store.scope,), scenes)}
    deferred = input_checks(store, scenes)
    accumulator = ReviewAccumulator(len(scenes), options.examples, REASONS)

    for scene in tqdm(scenes, desc="Tag review", unit="scene"):
        reason = ("unverified_inputs" if scene in deferred else
                  "pending_refresh" if scene in refresh else
                  "pending_publication" if scene in pending else None)
        message = (', '.join(f'{model}: {why}' for model, why in sorted(deferred[scene].items()))
                   if scene in deferred else None)
        if reason is None:
            try:
                hits, spans = merge.evidence(app, store, scene, core_policies, scene_policies)
                rows = {}
                _scene_signals(app, store, scene, hits, rows, options, scene_policies)
                _marker_signals(app, store, scene, spans, rows, options)
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                reason, message = "unusable_evidence", str(exc)
        if reason is not None:
            accumulator.skip(scene, reason, message)
            continue
        accumulator.record(scene, rows)

    return {
        "format": 1,
        "mode": "tag-review",
        "semantics": {
            "scope": "Replayed scene labels and marker cores under saved effective thresholds and selected exclusion/temporal policies; no live Stash inspection.",
            "known_input_failures": "Scenes with deferred missing_media, read_error or deleted inputs are excluded; saved evidence is preserved.",
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
        **accumulator.payload(),
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

    try:
        options = ReviewOptions(
            margin=.05 if args.tag_review_margin is None else args.tag_review_margin,
            long_marker_seconds=300.0 if args.tag_review_long_marker_seconds is None
            else args.tag_review_long_marker_seconds,
            examples=3 if args.tag_review_examples is None else args.tag_review_examples)
        local = report_context(app, args, both_outputs=True)
        core, tags = report_policies(app, args)
        with report_store(app, args, **report_memory(args)) as store:
            scenes = _scene_ids(store, args.scene_id)
            if not args.scene_id:
                store.preload_sources(scenes)
            report = build_report(local, store, scenes, core, tags, options)
        destination = write_json_report(report, args.tag_review_report)
        print(format_summary(report))
        print(f"Tag review report written to {destination}. No Stash requests or inference were performed.")
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print("Tag review report failed: " + str(exc))
        return 1
