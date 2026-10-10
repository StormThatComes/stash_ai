"""Report, database/maintenance and pipeline option validation in CLI order."""
from __future__ import annotations

import argparse
import math
from .modes import (Command, select_command, DATABASE_COMMANDS, HIERARCHY_COMMANDS, CLEANUP_COMMANDS,
                    MODEL_OPTIONS, PUBLICATION_OPTIONS, supplied, CACHE_RULE, REVIEW_RULE, MODEL_RULE, MODEL_ASSETS_RULE)


def database_mode(args) -> bool:
    return args.command in DATABASE_COMMANDS


def mode_flags(args) -> tuple[bool, bool]:
    return args.command in HIERARCHY_COMMANDS, args.command in CLEANUP_COMMANDS


def _validate_report(parser, args, rule):
    if args.command is not rule.command:
        return
    if supplied(args, rule.inference_options):
        parser.error(rule.inference_message)
    if supplied(args, (*PUBLICATION_OPTIONS, 'dry_run')):
        parser.error(rule.modifiers_message)
    destination = getattr(args, rule.command.value)
    checked = destination.expanduser().resolve()
    setattr(args, rule.command.value, checked)
    if destination.suffix.lower() != '.json' or checked.exists():
        parser.error(rule.flag + ' requires a new .json file; existing files are never overwritten.')


def validate_reports(parser, args):
    """Reject report modifiers before applying runtime defaults."""
    if args.command is Command.RECONCILE_HISTORY:
        if not args.reconcile_run_id:
            parser.error('--reconcile-run-history requires at least one --reconcile-run-id.')
        if supplied(args, (*MODEL_OPTIONS, *PUBLICATION_OPTIONS, 'record_run')) or args.scene_id:
            parser.error('--reconcile-run-history does not use inference or publication options.')
    elif args.reconcile_run_id:
        parser.error('--reconcile-run-id requires --reconcile-run-history.')
    if args.command is Command.PRUNE_HISTORY:
        if supplied(args, (*MODEL_OPTIONS, *PUBLICATION_OPTIONS, 'record_run')) or args.scene_id:
            parser.error('--prune-run-history does not use inference/tagging/scene modifiers.')
        if any(getattr(args, name) is not None for name in ('run_limit', 'run_command', 'run_status', 'run_model')):
            parser.error('--run-* filters do not apply to history retention.')
        if args.history_keep_last is None and args.history_max_age_days is None:
            parser.error('--prune-run-history requires --history-keep-last and/or --history-max-age-days.')
    elif args.history_keep_last is not None or args.history_max_age_days is not None:
        parser.error('--history-* retention settings require --prune-run-history.')
    if args.history_keep_last is not None and args.history_keep_last < 0:
        parser.error('--history-keep-last must be >= 0.')
    if args.history_max_age_days is not None and (not math.isfinite(args.history_max_age_days) or args.history_max_age_days <= 0):
        parser.error('--history-max-age-days must be finite and positive.')
    if args.command is Command.RUN_REPORT:
        if supplied(args, (*MODEL_OPTIONS, *PUBLICATION_OPTIONS, 'dry_run', 'record_run')) or args.scene_id:
            parser.error('--run-report is an offline history inspection; model/tagging/scene modifiers do not apply.')
        if args.run_report.suffix.lower() != '.json' or args.run_report.expanduser().resolve().exists():
            parser.error('--run-report requires a new .json file; existing files are never overwritten.')
        args.run_report = args.run_report.expanduser().resolve()
    elif args.command is Command.REUSE_REPORT:
        from .modes import EXECUTION_OPTIONS
        if supplied(args, (*EXECUTION_OPTIONS, *PUBLICATION_OPTIONS, 'dry_run', 'record_run')):
            parser.error('--reuse-report reads recorded decisions; inference/tagging modifiers do not apply.')
        if args.reuse_report.suffix.lower() != '.json' or args.reuse_report.expanduser().resolve().exists():
            parser.error('--reuse-report requires a new .json file; existing files are never overwritten.')
        args.reuse_report = args.reuse_report.expanduser().resolve()
        if args.models:
            from stash_ai.models import select_models
            try:
                args.models = select_models(args.models)
            except ValueError as exc:
                parser.error(str(exc))
    elif any(getattr(args, name) is not None for name in ('run_limit', 'run_command', 'run_status', 'run_model')):
        parser.error('--run-* filters require --run-report or --reuse-report.')
    if args.run_limit is not None and not 1 <= args.run_limit <= 10000:
        parser.error('--run-limit must be in 1..10000.')
    if args.run_limit is None:
        args.run_limit = 1 if args.command is Command.REUSE_REPORT else 50
    if args.run_history_db is not None:
        import os
        history, state = args.run_history_db.expanduser().resolve(), args.state_db.expanduser().resolve()
        if history == state or (history.exists() and state.exists() and os.path.samefile(history, state)):
            parser.error('--run-history-db must be separate from --state-db.')
    _validate_report(parser, args, CACHE_RULE)
    _validate_report(parser, args, REVIEW_RULE)
    if args.tag_review_report is None and any(value is not None for value in (
            args.tag_review_margin, args.tag_review_long_marker_seconds, args.tag_review_examples)):
        parser.error("--tag-review-* settings require --tag-review-report.")
    if args.tag_review_margin is not None and (
            not math.isfinite(args.tag_review_margin) or not 0 <= args.tag_review_margin <= 1):
        parser.error("--tag-review-margin must be a finite number in [0, 1].")
    if args.tag_review_long_marker_seconds is not None and (
            not math.isfinite(args.tag_review_long_marker_seconds) or args.tag_review_long_marker_seconds <= 0):
        parser.error("--tag-review-long-marker-seconds must be finite and positive.")
    if args.tag_review_examples is not None and not 1 <= args.tag_review_examples <= 20:
        parser.error("--tag-review-examples must be in 1..20.")
    _validate_report(parser, args, MODEL_RULE)
    _validate_report(parser, args, MODEL_ASSETS_RULE)


def validate_maintenance(parser, args):
    """Validate database operations."""
    db_mode = database_mode(args)
    if db_mode:
        if args.scene_id:
            parser.error("Database maintenance covers the state DB; --scene-id does not apply.")
        if supplied(args, MODEL_OPTIONS):
            parser.error("Database maintenance does not use model/inference options.")
        if supplied(args, PUBLICATION_OPTIONS):
            parser.error("Database maintenance cannot be combined with tagging, threshold, cleanup or pruning modifiers.")
        if args.anonymous_diagnostics is not None and args.dry_run:
            parser.error("--anonymous-diagnostics is already read-only for the state DB; --dry-run does not apply.")


def validate_pipeline(app, parser, args):
    """Normalize execution defaults and validate tagging/recovery combinations."""
    if args.force_fetch_vtt and args.command is not Command.INFERENCE:
        parser.error('--force-fetch-vtt applies to normal model inference/resume runs.')
    args.resume = app.DEFAULT_RESUME if args.resume is None else args.resume
    if args.checkpoint_batches is not None and args.checkpoint_batches < 1:
        parser.error("--checkpoint-batches must be >= 1.")
    if not 1 <= args.cleanup_page_size <= 10000 or not 1 <= args.cleanup_write_batch_size <= 1000:
        parser.error("--cleanup-page-size must be 1..10000 and --cleanup-write-batch-size must be 1..1000.")
    if not math.isfinite(args.ram_cache_gb) or args.ram_cache_gb < 0:
        parser.error("--ram-cache-gb must be a finite number >= 0.")
    if not math.isfinite(args.ram_budget_gb) or args.ram_budget_gb <= 0:
        parser.error("--ram-budget-gb must be a finite number > 0.")
    args.ram_cache_gb = min(args.ram_cache_gb, args.ram_budget_gb * .75)

    hierarchy_only, cleanup_only = mode_flags(args)
    if args.dry_run_plan is not None:
        if not args.dry_run or not args.merge_only or args.prune_empty_tags:
            parser.error('--dry-run-plan requires --dry-run --merge-only without pruning.')
        args.dry_run_plan = args.dry_run_plan.expanduser().resolve()
        if args.dry_run_plan.suffix.lower() != '.jsonl' or args.dry_run_plan.exists():
            parser.error('--dry-run-plan requires a new .jsonl file; existing files are never overwritten.')
    if args.abandon_pending:
        if (hierarchy_only or args.download_models or args.explain_tag
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
        if (hierarchy_only or cleanup_only or args.download_models
                or args.explain_tag or args.prune_empty_tags_only or args.sync_marker_groups_only):
            parser.error("--profile-merge applies to merged tagging or --merge-only.")
        args.profile_merge = args.profile_merge.expanduser().resolve()
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
    if args.explain_tag and not args.scene_id:
        parser.error("--explain-tag requires at least one --scene-id.")
    if args.prune_empty_tags_only and args.scene_id:
        parser.error("Empty-tag definitions have global scope; --scene-id does not apply to --prune-empty-tags-only.")
    if args.prune_empty_tags and (
            hierarchy_only or args.download_models or args.explain_tag
            or args.sync_marker_groups_only):
        parser.error("--prune-empty-tags follows merged tagging, --merge-only or merged cleanup; use --prune-empty-tags-only otherwise.")
    if args.prune_prefixes and not (args.prune_empty_tags or args.prune_empty_tags_only):
        parser.error("--prune-prefixes requires an empty-tag pruning mode.")
    if args.restart and (
            hierarchy_only or cleanup_only or args.download_models or args.explain_tag
            or args.merge_only or args.prune_empty_tags_only or args.sync_marker_groups_only):
        parser.error("--restart applies to model inference.")
    if args.abandon_pending and not args.merge_only and not cleanup_only and args.scene_id:
        parser.error("Standalone abandonment covers pending plans for this Stash/state DB; --scene-id does not apply.")



def validate_command_capabilities(parser, args):
    """Reject user-supplied execution settings that a selected command ignores."""
    model_users = {Command.INFERENCE, Command.MERGE, Command.DOWNLOAD,
                   Command.CACHE_REPORT, Command.REUSE_REPORT,
                   Command.MODEL_ASSETS_REPORT}
    if args.models and args.command not in model_users:
        parser.error('--models is not used by this command.')
    workers = {
        'prepare_workers': {Command.INFERENCE},
        'prefetch_batches': {Command.INFERENCE},
        'result_prefetch_batches': {Command.INFERENCE},
        'checkpoint_batches': {Command.INFERENCE},
        'merge_workers': {Command.INFERENCE, Command.MERGE},
        'model_report_workers': {Command.MODEL_REPORT},
    }
    for name, commands in workers.items():
        if getattr(args, name) is not None and args.command not in commands:
            parser.error('--' + name.replace('_', '-') + ' is not used by this command.')


def validate_args(app, parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    args.command = select_command(args)
    validate_command_capabilities(parser, args)
    if args.db_init and (args.command is not Command.INFERENCE or args.markers_only
                        or args.scene_tags_only or args.restart or args.abandon_pending):
        parser.error('--db-init requires normal inference with both outputs; restart/recovery modifiers do not apply.')
    validate_reports(parser, args)
    validate_maintenance(parser, args)
    validate_pipeline(app, parser, args)
    return args
