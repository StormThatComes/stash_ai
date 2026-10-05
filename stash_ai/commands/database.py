"""Offline database command dispatch, before runtime/model setup."""
import sqlite3

from stash_ai.db.checks import check_database, format_check_report
from stash_ai.db.compaction import compact_database
from stash_ai.db.inspection import format_report, inspect_database
from stash_ai.db.update import update_database
from stash_ai.db.common import format_bytes


def run_mode(args, stash_url):
    from stash_ai.storage.store import scope_for

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
