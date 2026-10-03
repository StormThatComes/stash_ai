"""Public database maintenance API and CLI dispatch.

The package preserves the former ``stash_ai.db`` entry points; implementation
modules own inspection, consistency checks and compaction independently.
"""

from __future__ import annotations

import sqlite3

from .checks import CHECK_MESSAGES, check_database, format_check_report
from .common import (
    BASE_TABLES, BLOB_COLUMNS, KNOWN_KINDS, KNOWN_MODELS, KNOWN_VERSIONS,
    STATE_TABLES, format_bytes,
)
from .compaction import MIN_FREE_HEADROOM, compact_database
from .inspection import format_report, inspect_database
from .update import update_database

__all__ = [
    "BASE_TABLES", "BLOB_COLUMNS", "CHECK_MESSAGES", "KNOWN_KINDS", "KNOWN_MODELS",
    "KNOWN_VERSIONS", "MIN_FREE_HEADROOM", "STATE_TABLES", "check_database",
    "compact_database", "format_check_report", "format_report", "inspect_database",
    "run_mode", "update_database",
]


def run_mode(args, stash_url):
    from stash_ai.state import scope_for

    try:
        if args.db_update:
            before, after = update_database(args.state_db, dry_run=args.dry_run)
            if args.dry_run:
                print(f"DRY RUN: state DB schema v{before} -> v{after}; no migrations were run.")
            elif before == after:
                print(f"State DB schema is up to date: v{after}.")
            else:
                print(f"State DB updated: schema v{before} -> v{after}.")
            return 0
        scope = scope_for(stash_url)
        if args.db_report:
            print(format_report(inspect_database(args.state_db, scope)))
            return 0
        if args.db_check:
            report = check_database(args.state_db, scope, show_progress=True)
            print(format_check_report(report))
            return 0 if report["ok"] else 1

        before, after = compact_database(args.state_db, scope, dry_run=args.dry_run)
        print(format_report(before))
        if args.dry_run:
            print(
                "DRY RUN: VACUUM was not executed; estimated immediately reclaimable pages: "
                + format_bytes(before["reclaimable_estimate_bytes"])
            )
        else:
            saved = max(0, before["file_bytes"] - after["file_bytes"])
            print(f"Compaction complete: {format_bytes(before['file_bytes'])} -> {format_bytes(after['file_bytes'])}; saved {format_bytes(saved)}.")
        return 0
    except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError) as exc:
        print("Database maintenance failed: " + str(exc))
        return 1
