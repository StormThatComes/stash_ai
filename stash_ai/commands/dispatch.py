"""Dispatch CLI phases and merged-output entry points."""
from __future__ import annotations

from stash_ai.commands.validation import database_mode


def run_database_command(app, args) -> int | None:
    """Run DB/diagnostics commands that must execute before runtime setup."""
    if not database_mode(args):
        return None
    if args.anonymous_diagnostics is not None:
        from stash_ai.reports.developer import run as run_anonymous_diagnostics
        return run_anonymous_diagnostics(args)
    from stash_ai.db import run_mode as run_database_mode
    return run_database_mode(args, app.STASH_URL)


def run_standalone_command(app, args, *, cleanup_only: bool) -> int | None:
    """Run commands that bypass model loading and the normal tagging pipeline."""
    if args.abandon_pending and not args.merge_only and not cleanup_only:
        from stash_ai.recovery import run_only
        app.validate_config(scene_work=False)
        return run_only(app, args)
    if args.sync_marker_groups_only:
        from stash_ai.marker_groups import run_backfill
        app.validate_config(scene_work=False)
        return run_backfill(app, args)
    if args.prune_empty_tags_only:
        from stash_ai.prune import run_prune
        app.validate_config(scene_work=False)
        return run_prune(app, args)
    if args.prune_legacy_imports:
        from stash_ai.legacy_prune import run as run_legacy_prune
        app.validate_config(scene_work=False)
        return run_legacy_prune(app, args)
    if args.model_report is not None:
        from stash_ai.reports.model import run as run_model_report
        return run_model_report(app, args)
    if args.tag_review_report is not None:
        from stash_ai.reports.tag_review import run as run_tag_review
        return run_tag_review(app, args)
    if args.cache_report is not None:
        from stash_ai.reports.cache import run as run_cache_report
        return run_cache_report(app, args)
    return None


def run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
    from stash_ai.diagnostics import MergeDiagnostics
    from stash_ai.merge_run import MergedRun
    with MergeDiagnostics(args.report, args.profile_merge, dry_run=app.DRY_RUN,
                          merge_only=args.merge_only, force=args.force_merge):
        return MergedRun(app, settings, args, catalogs, hierarchy, core_policies, scene_policies).run()


def run_saved_merge_command(app, settings, args, core_policies, scene_policies) -> int | None:
    """Dispatch offline explanation or saved-evidence merge before model assets."""
    if args.output_mode != "merged":
        return None
    if args.explain_tag:
        from stash_ai.commands.explain import explain
        return explain(app, args, core_policies, scene_policies)
    if args.merge_only:
        from stash_ai.commands.hierarchy import combined_hierarchy
        hierarchy = combined_hierarchy(app, args, scene_policies)
        return run_merged(app, settings, args, {}, hierarchy, core_policies, scene_policies)
    return None
