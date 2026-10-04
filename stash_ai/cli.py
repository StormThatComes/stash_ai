#!/usr/bin/env python
"""Stash AI command entry point; every invocation owns an explicit runtime."""
from __future__ import annotations

import sys

from stash_ai.runtime.context import RuntimeContext
from stash_ai.commands.modes import Command
from stash_ai.stash_api.transport import safe_error


def main(argv: list[str] | None = None, *, runtime: RuntimeContext | None = None) -> int:
    from stash_ai.commands.arguments import parse_args
    from stash_ai.commands import bootstrap
    from stash_ai.commands.dispatch import (
        run_database_command, run_standalone_command, run_saved_merge_command, run_merged)

    app = runtime if runtime is not None else RuntimeContext()
    args = parse_args(app, argv)
    result = run_database_command(app, args)
    if result is not None:
        return result

    hierarchy_only, cleanup_only = bootstrap.runtime_options(app, args)
    result = run_standalone_command(app, args)
    if result is not None:
        return result
    settings = bootstrap.selected_settings(app, args)
    if args.command is Command.DOWNLOAD:
        bootstrap.download_models(app, settings)
        return 0
    core_policies, scene_policies = bootstrap.policies(
        app, args, hierarchy_only=hierarchy_only, cleanup_only=cleanup_only)
    if cleanup_only:
        from stash_ai.cleanup.run import CleanupRun
        return CleanupRun(app, settings, args, core_policies, scene_policies).run()
    result = run_saved_merge_command(app, settings, args, core_policies, scene_policies)
    if result is not None:
        return result
    catalogs, merged_hierarchy = bootstrap.catalogs(
        app, settings, args, scene_policies)
    if args.command is Command.VALIDATE_HIERARCHY:
        print("Vocabularies/hierarchies validated. No Stash requests or model weights loaded.")
        return 0
    if args.command is Command.SYNC_HIERARCHY:
        from stash_ai.commands.hierarchy import run_hierarchy
        return run_hierarchy(app, args, catalogs, merged_hierarchy, scene_policies)
    return run_merged(app, settings, args, catalogs, merged_hierarchy, core_policies, scene_policies)

def run() -> int:
    """CLI exit handling shared by the root launcher."""
    runtime = RuntimeContext()
    try:
        return main(runtime=runtime)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {safe_error(exc, api_key=runtime.STASH_API_KEY)}", file=sys.stderr)
        return 1
