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
        if args.db_repair:
            from stash_ai.db.repair import repair_database
            result = repair_database(args.state_db, dry_run=args.dry_run, show_progress=True)
            verb = 'Would repair' if args.dry_run else 'Repaired'
            print(f"{verb}: metadata={result['metadata_repaired']}, duplicates={result['duplicates_repaired']}; "
                  f"checked {result['results_checked']} result(s), {result['archives_checked']} archive(s).")
            return 0
        if args.db_update:
            details = {}
            before, after = update_database(args.state_db, dry_run=args.dry_run,
                                           show_progress=True, details=details)
            if args.dry_run:
                print(f"DRY RUN: state DB schema v{before} -> v{after}; no migrations were run.")
                if before == after and not details['noop']:
                    print('Required schema/data maintenance remains; run --db-update without --dry-run.')
            elif details['noop']:
                print(f"State DB schema is up to date: v{after}.")
            elif before == after:
                print(f"State DB maintenance completed: schema v{after}; "
                      f"required repair={details['required_repair']}, metadata={details['metadata_repaired']}.")
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
