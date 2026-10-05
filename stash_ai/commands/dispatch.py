"""Dispatch CLI phases and merged-output entry points."""
from __future__ import annotations

from stash_ai.commands.validation import database_mode
from stash_ai.commands.modes import Command


def run_database_command(app, args) -> int | None:
    """Run DB/diagnostics commands that must execute before runtime setup."""
    if not database_mode(args):
        return None
    if args.command is Command.DIAGNOSTICS:
        from stash_ai.reports.developer.report import run as run_anonymous_diagnostics
        return run_anonymous_diagnostics(args)
    from stash_ai.db import run_mode as run_database_mode
    return run_database_mode(args, app.STASH_URL)


def run_standalone_command(app, args) -> int | None:
    """Run commands that bypass model loading and the normal tagging pipeline."""
    if args.command is Command.IMPORT_LEGACY:
        from stash_ai.commands.legacy import run_only
        app.services.validate_config(scene_work=False)
        return run_only(app, args)
    if args.command is Command.ABANDON:
        from stash_ai.commands.recovery import run_only
        app.services.validate_config(scene_work=False)
        return run_only(app, args)
    if args.command is Command.POINT_MIGRATION:
        from stash_ai.commands.point_migration import run
        app.services.validate_config(scene_work=False)
        return run(app, args)
    if args.command is Command.MARKER_GROUPS:
        from stash_ai.commands.marker_groups import run_backfill
        app.services.validate_config(scene_work=False)
        return run_backfill(app, args)
    if args.command is Command.PRUNE_TAGS:
        from stash_ai.cleanup.prune import run_prune
        app.services.validate_config(scene_work=False)
        return run_prune(app, args)
    if args.command is Command.PRUNE_LEGACY:
        from stash_ai.cleanup.legacy import run as run_legacy_prune
        app.services.validate_config(scene_work=False)
        return run_legacy_prune(app, args)
    if args.command is Command.MODEL_REPORT:
        from stash_ai.reports.model.report import run as run_model_report
        return run_model_report(app, args)
    if args.command is Command.REVIEW_REPORT:
        from stash_ai.reports.review.report import run as run_tag_review
        return run_tag_review(app, args)
    if args.command is Command.CACHE_REPORT:
        from stash_ai.reports.cache import run as run_cache_report
        return run_cache_report(app, args)
    return None


def run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
    from stash_ai.runtime.diagnostics import MergeDiagnostics
    from stash_ai.runs.merged import MergedRun
    with MergeDiagnostics(args.report, args.profile_merge, dry_run=app.DRY_RUN,
                          merge_only=args.merge_only, force=args.force_merge):
        return MergedRun(app, settings, args, catalogs, hierarchy, core_policies, scene_policies).run()


def run_saved_merge_command(app, settings, args, core_policies, scene_policies) -> int | None:
    """Dispatch offline explanation or saved-evidence merge before model assets."""
    if args.command is Command.EXPLAIN:
        from stash_ai.commands.explain import explain
        return explain(app, args, core_policies, scene_policies)
    if args.command is Command.MERGE:
        from stash_ai.commands.hierarchy import combined_hierarchy
        hierarchy = combined_hierarchy(app, args, scene_policies)
        return run_merged(app, settings, args, {}, hierarchy, core_policies, scene_policies)
    return None

