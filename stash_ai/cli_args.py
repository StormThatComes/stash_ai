"""Argument parsing and cross-option validation for the Stash AI CLI."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def build_parser(app) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tag Stash scenes and timestamped markers with JoyTag, WD, DTQ and/or OppaiOracle.")
    parser.add_argument("--models", "--model", nargs="+", metavar="MODEL", help="joytag, wd, dtq, oppai, any subset, or all; commas also accepted.")
    parser.add_argument("--model-config", type=Path, default=app.MODELS_CONFIG_FILE, metavar="YAML", help="Selection, paths, thresholds and batch sizes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override configured device.")
    parser.add_argument("--offline", action="store_true", help="Require local model assets; never download.")
    parser.add_argument("--temporal-policy", type=Path, default=app.TEMPORAL_POLICY_FILE, metavar="YAML",
                        help="Automatic scene qualification, marker hysteresis and extras using Stash VTT/sprites only.")
    parser.add_argument("--no-temporal", action="store_true", default=not app.DEFAULT_TEMPORAL,
                        help="Restore original scene/marker qualification and first-frame extras.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to Stash.")
    parser.add_argument("--scene-id", action="append", metavar="ID", help="Limit to this scene; repeat for multiple scenes.")
    parser.add_argument("--output-mode", choices=("merged", "separate"), default=app.DEFAULT_OUTPUT_MODE,
                        help="merged (default): common ai: tags and interval union; separate: original model prefixes.")
    parser.add_argument("--state-db", type=Path, default=app.MERGE_STATE_FILE, metavar="SQLITE",
                        help="Durable per-model evidence for merged outputs; retain this file between runs.")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", default=None,
                        help="Reuse matching completed results and resume checkpoints (default; merged output).")
    resume.add_argument("--restart", action="store_true", help="Request fresh inference for selected scenes/models; resume later without this flag.")
    parser.add_argument("--checkpoint-batches", type=int, default=app.CHECKPOINT_BATCHES,
                        help="Persist partial inference every N batches; also saves on Ctrl+C/errors. Default: 8.")
    parser.add_argument("--prefetch-batches", type=int, choices=range(17), default=app.PREFETCH_BATCHES,
                        help="Prepare N next batches in RAM (default: 4); 0 disables. No disk image cache.")
    parser.add_argument("--ram-cache-gb", type=float, default=app.RAM_CACHE_GB, metavar="GIB",
                        help="Shared result/index/media cache in GiB (default: 24); 0 disables. Capped to 75%% of --ram-budget-gb.")
    parser.add_argument("--ram-budget-gb", type=float, default=app.RAM_BUDGET_GB, metavar="GIB",
                        help="Process RAM target in GiB (default: 32); reserves 25%% for models and active work. RSS pressure evicts caches.")
    parser.add_argument("--prepare-workers", type=int, choices=range(1, 17), default=app.PREPARE_WORKERS,
                        help="CPU image preparation workers (default: 4); one model at a time runs inference.")
    parser.add_argument("--model-report-workers", type=int, choices=range(1, 17),
                        default=min(4, os.cpu_count() or 1),
                        help="CPU worker processes for --model-report (default: up to 4); use 1 for serial execution.")
    parser.add_argument("--verbose", action="store_true", help="Include per-scene/tag diagnostics and detailed dry-run changes.")
    parser.add_argument("--profile-merge", type=Path, metavar="NEW.jsonl",
                        help="Stream per-scene wall/CPU, stage/API timings, counts and byte sizes to a new JSONL file; merged output only.")
    parser.add_argument("--force-merge", action="store_true",
                        help="Recompute even unchanged merges without rerunning models; requires --merge-only. Does not clear source state.")
    parser.add_argument("--rethreshold-scenes", action="store_true",
                        help="Apply YAML scene thresholds to retained frame evidence; requires --merge-only. "
                             "May be combined with --rethreshold-markers.")
    parser.add_argument("--rethreshold-markers", action="store_true",
                        help="Apply YAML marker thresholds to complete cached frames; requires --merge-only. "
                             "Never runs inference; rejects changed scene thresholds or insufficient retention.")
    parser.add_argument("--abandon-pending", action="store_true",
                        help="Abandon pending merge/cleanup plans, retaining applied changes and inference. "
                             "Alone: exit after abandonment; with --merge-only or cleanup: use current YAML policies. Honors --dry-run.")
    parser.add_argument("--cleanup-strategy", choices=("auto", "sql", "scenes"), default="auto",
                        help="Cleanup discovery: auto uses read-only SQL when available, else scans scenes; sql requires SQL; scenes scans scenes directly.")
    parser.add_argument("--cleanup-page-size", type=int, default=2000,
                        help="Maximum ID rows per SQL cleanup SELECT (default: 2000; no COUNT or OFFSET).")
    parser.add_argument("--cleanup-write-batch-size", type=int, default=200,
                        help="Maximum marker/scene IDs per cleanup mutation (default: 200). Tags are deleted one at a time.")
    parser.add_argument("--prune-empty-tags", action="store_true",
                        help="After successful merged sync, delete unused leaf tags in the selected prefixes.")
    parser.add_argument("--prune-prefixes", nargs="+", default=None, metavar="PREFIX",
                        help="Prefixes for empty-tag pruning (default: jt: wd: dtq: op: ai: m:). Global scope; m: requires managed group metadata.")
    parser.add_argument("--hierarchy", type=Path, default=app.JOYTAG_HIERARCHY_FILE, metavar="YAML", help="JoyTag hierarchy.")
    parser.add_argument("--wd-hierarchy", type=Path, default=app.JOYTAG_HIERARCHY_FILE.with_name("wd_hierarchy.yml"), metavar="YAML", help="WD hierarchy; defaults to bundled wd_hierarchy.yml.")
    parser.add_argument("--dtq-hierarchy", type=Path, default=app.JOYTAG_HIERARCHY_FILE.with_name("dtq_hierarchy.yml"), metavar="YAML", help="DTQ hierarchy; defaults to bundled dtq_hierarchy.yml.")
    parser.add_argument("--oppai-hierarchy", type=Path, default=app.JOYTAG_HIERARCHY_FILE.with_name("oppai_hierarchy.yml"), metavar="YAML", help="OppaiOracle hierarchy; defaults to bundled oppai_hierarchy.yml.")
    parser.add_argument("--marker-core-exclusions", type=Path, default=app.MARKER_CORE_EXCLUSIONS_FILE, metavar="YAML", help="Common/per-model excluded marker core labels.")
    parser.add_argument("--scene-tag-exclusions", "--tag-exclusions", dest="scene_tag_exclusions", type=Path,
                        default=app.SCENE_TAG_EXCLUSIONS_FILE, metavar="YAML",
                        help="Global blacklist for scene tags, marker cores/extras and hierarchy creation (legacy filename supported).")
    parser.add_argument("--no-hierarchy", action="store_true", help="Disable model/category parent synchronization; marker groups still use ai_markers.")
    parser.add_argument("--create-all-tags", action="store_true", help="With --sync-hierarchy-only, create full vocabularies.")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--markers-only", action="store_true", help="Leave scene tags unchanged; also works with --merge-only.")
    outputs.add_argument("--scene-tags-only", action="store_true", help="Leave markers unchanged; also works with --merge-only.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sync-hierarchy-only", action="store_true", help="Organize tags without inference or scene/marker edits.")
    modes.add_argument("--validate-hierarchy", action="store_true", help="Validate vocabularies/hierarchies without Stash or weights.")
    modes.add_argument("--cleanup-markers-only", "--sync-marker-core-only", dest="cleanup_markers_only", action="store_true", help="Remove excluded marker cores and global extras without inference or model files.")
    modes.add_argument("--cleanup-scene-tags-only", action="store_true", help="Apply scene blacklist without inference or model files.")
    modes.add_argument("--cleanup-only", action="store_true", help="Apply both blacklists without inference; full all-model cleanup removes globally banned tags in batches.")
    modes.add_argument("--download-models", action="store_true", help="Download selected model assets from Hugging Face without Stash or inference.")
    modes.add_argument("--merge-only", action="store_true", help="Merge saved evidence and existing source-prefixed results without models/inference.")
    modes.add_argument("--sync-marker-groups-only", action="store_true",
                       help="Repair m:<core> / m:<core>:occurance groups on existing owned ai: markers; no inference or time changes.")
    modes.add_argument("--explain-tag", metavar="TAG", help="Show saved sources, scores and intervals for a raw/ai: tag; requires --scene-id; no Stash access.")
    modes.add_argument("--prune-empty-tags-only", action="store_true",
                       help="Remove unused leaf tag definitions without models/inference. Honors --dry-run.")
    modes.add_argument("--prune-legacy-imports", action="store_true",
                       help="Remove legacy-import ledger rows whose exact source objects no longer exist in Stash; no models/inference. Honors --dry-run.")
    modes.add_argument("--db-report", action="store_true",
                       help="Report state DB health, size and aggregate evidence/cache counts without Stash or model access.")
    modes.add_argument("--db-check", action="store_true",
                       help="Deep-check state DB logical consistency and compressed state without Stash or model access.")
    modes.add_argument("--db-compact", action="store_true",
                       help="Safely VACUUM the state DB while idle; refuses pending publication/cleanup plans. Honors --dry-run.")
    modes.add_argument("--anonymous-diagnostics", type=Path, metavar="NEW.json",
                       help="Write a local content-free developer diagnostics JSON from the state DB; never uploads it.")
    modes.add_argument("--model-report", type=Path, metavar="NEW.json",
                       help="Write an offline per-model utility and leave-one-model-out ablation report from saved evidence.")
    
    return parser


def database_mode(args) -> bool:
    return bool(args.db_report or args.db_check or args.db_compact or args.anonymous_diagnostics is not None)


def mode_flags(args) -> tuple[bool, bool]:
    hierarchy_only = args.sync_hierarchy_only or args.validate_hierarchy
    cleanup_only = args.cleanup_only or args.cleanup_markers_only or args.cleanup_scene_tags_only
    return hierarchy_only, cleanup_only


def validate_args(app, parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    db_mode = database_mode(args)
    if args.model_report is not None:
        if args.output_mode != "merged":
            parser.error("--model-report requires --output-mode merged.")
        if args.models or args.device or args.offline or args.resume is not None or args.restart:
            parser.error("--model-report uses all saved model evidence; model/inference options do not apply.")
        if (args.markers_only or args.scene_tags_only or args.profile_merge or args.force_merge
                or args.rethreshold_scenes or args.rethreshold_markers or args.abandon_pending
                or args.prune_empty_tags or args.prune_prefixes or args.create_all_tags or args.no_hierarchy
                or args.dry_run):
            parser.error("--model-report cannot be combined with tagging, threshold, pruning or dry-run modifiers.")
        if args.model_report.suffix.lower() != ".json" or args.model_report.exists():
            parser.error("--model-report requires a new .json file; existing files are never overwritten.")
    if args.prune_legacy_imports:
        if args.output_mode != "merged":
            parser.error("--prune-legacy-imports requires --output-mode merged.")
        if args.scene_id:
            parser.error("--prune-legacy-imports covers the current Stash/state DB; --scene-id does not apply.")
        if args.models or args.device or args.offline or args.resume is not None or args.restart:
            parser.error("--prune-legacy-imports does not use model/inference options.")
        if (args.markers_only or args.scene_tags_only or args.profile_merge or args.force_merge
                or args.rethreshold_scenes or args.rethreshold_markers or args.abandon_pending
                or args.prune_empty_tags or args.prune_prefixes or args.create_all_tags or args.no_hierarchy):
            parser.error("--prune-legacy-imports cannot be combined with tagging, threshold, recovery or tag-pruning modifiers.")

    if db_mode:
        if args.output_mode != "merged":
            parser.error("--db-report/--db-check/--db-compact require --output-mode merged.")
        if args.scene_id:
            parser.error("Database maintenance covers the state DB; --scene-id does not apply.")
        if args.models or args.device or args.offline or args.no_temporal or args.resume is not None or args.restart:
            parser.error("Database maintenance does not use model/inference options.")
        if (args.markers_only or args.scene_tags_only or args.profile_merge or args.force_merge
                or args.rethreshold_scenes or args.rethreshold_markers or args.abandon_pending
                or args.prune_empty_tags or args.prune_prefixes or args.create_all_tags or args.no_hierarchy):
            parser.error("Database maintenance cannot be combined with tagging, threshold, cleanup or pruning modifiers.")
        if args.anonymous_diagnostics is not None and args.dry_run:
            parser.error("--anonymous-diagnostics is already read-only for the state DB; --dry-run does not apply.")

    if args.output_mode != "merged" and (args.resume or args.restart):
        parser.error("--resume/--restart require --output-mode merged; separate retains its original full-inference behavior.")
    args.resume = app.DEFAULT_RESUME if args.resume is None else args.resume
    if args.checkpoint_batches < 1:
        parser.error("--checkpoint-batches must be >= 1.")
    if not 1 <= args.cleanup_page_size <= 10000 or not 1 <= args.cleanup_write_batch_size <= 1000:
        parser.error("--cleanup-page-size must be 1..10000 and --cleanup-write-batch-size must be 1..1000.")
    if not math.isfinite(args.ram_cache_gb) or args.ram_cache_gb < 0:
        parser.error("--ram-cache-gb must be a finite number >= 0.")
    if not math.isfinite(args.ram_budget_gb) or args.ram_budget_gb <= 0:
        parser.error("--ram-budget-gb must be a finite number > 0.")
    args.ram_cache_gb = min(args.ram_cache_gb, args.ram_budget_gb * .75)

    hierarchy_only, cleanup_only = mode_flags(args)
    if args.abandon_pending:
        if (args.output_mode != "merged" or hierarchy_only or args.download_models or args.explain_tag
                or args.prune_empty_tags_only or args.sync_marker_groups_only or args.restart):
            parser.error("--abandon-pending is standalone or accompanies merged --merge-only/cleanup modes.")
        if not args.merge_only and not cleanup_only and (
                args.markers_only or args.scene_tags_only or args.prune_empty_tags or args.profile_merge):
            parser.error("Standalone --abandon-pending does not publish outputs or prune tags.")
    if args.rethreshold_scenes and (not args.merge_only or args.markers_only):
        parser.error("--rethreshold-scenes requires --merge-only and scene output.")
    if args.rethreshold_markers and (not args.merge_only or args.scene_tags_only):
        parser.error("--rethreshold-markers requires --merge-only and marker output.")
    if args.force_merge and not args.merge_only:
        parser.error("--force-merge requires --merge-only.")
    if args.profile_merge:
        if (args.output_mode != "merged" or hierarchy_only or cleanup_only or args.download_models
                or args.explain_tag or args.prune_empty_tags_only or args.sync_marker_groups_only):
            parser.error("--profile-merge applies to merged tagging or --merge-only.")
        if args.profile_merge.suffix.lower() != ".jsonl" or args.profile_merge.exists():
            parser.error("--profile-merge requires a new .jsonl file; existing files are never overwritten.")
    if (args.markers_only or args.scene_tags_only) and (
            hierarchy_only or cleanup_only or args.download_models or args.explain_tag
            or args.prune_empty_tags_only or args.sync_marker_groups_only):
        parser.error("--markers-only/--scene-tags-only apply to tagging or --merge-only.")
    if hierarchy_only and args.no_hierarchy:
        parser.error("Hierarchy-only modes cannot be combined with --no-hierarchy.")
    if (hierarchy_only or args.download_models) and args.scene_id:
        parser.error("This mode operates on global/model files; --scene-id does not apply.")
    if args.create_all_tags and not args.sync_hierarchy_only:
        parser.error("--create-all-tags requires --sync-hierarchy-only.")
    if (args.merge_only or args.explain_tag or args.sync_marker_groups_only) and args.output_mode != "merged":
        parser.error("--merge-only, --explain-tag and --sync-marker-groups-only require --output-mode merged.")
    if args.explain_tag and not args.scene_id:
        parser.error("--explain-tag requires at least one --scene-id.")
    if args.prune_empty_tags_only and args.scene_id:
        parser.error("Empty-tag definitions have global scope; --scene-id does not apply to --prune-empty-tags-only.")
    if args.prune_empty_tags and (
            hierarchy_only or args.download_models or args.explain_tag
            or args.output_mode != "merged" or args.sync_marker_groups_only):
        parser.error("--prune-empty-tags follows merged tagging, --merge-only or merged cleanup; use --prune-empty-tags-only otherwise.")
    if args.prune_prefixes and not (args.prune_empty_tags or args.prune_empty_tags_only):
        parser.error("--prune-prefixes requires an empty-tag pruning mode.")
    if args.restart and (
            hierarchy_only or cleanup_only or args.download_models or args.explain_tag
            or args.merge_only or args.prune_empty_tags_only or args.sync_marker_groups_only):
        parser.error("--restart applies to model inference.")
    if args.abandon_pending and not args.merge_only and not cleanup_only and args.scene_id:
        parser.error("Standalone abandonment covers pending plans for this Stash/state DB; --scene-id does not apply.")
    return args


def parse_args(app, argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(app)
    return validate_args(app, parser, parser.parse_args(argv))
