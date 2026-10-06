"""Deep read-only validation of persisted evidence and state relationships."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3

from .common import (
    KNOWN_VERSIONS, connect_readonly, count_rows, table_names,
)
from .validation import (
    checkpoints as checkpoint_checks, common as validation_common,
    results as result_checks, schema as schema_checks, state as state_checks,
)


CHECK_MESSAGES = {
    "sqlite_integrity_check": "SQLite integrity_check reported a storage/index-level problem.",
    "missing_base_tables": "Core state tables are missing.",
    "missing_state_tables": "Current state tables are missing; run --db-update.",
    "retired_state_tables": "Retired state tables are present; inspect the schema/version before using this database.",
    "schema_upgrade_required": "Historical schema requires --db-update before a deep --db-check.",
    "invalid_table_columns": "Current state table columns do not match the schema; restore or migrate the database.",
    "missing_state_triggers": "State invalidation triggers are missing; run --db-update.",
    "missing_state_indexes": "Current state indexes are missing; run --db-update.",
    "invalid_model": "Rows contain an unknown model key.",
    "invalid_kind": "Rows contain an unknown output kind.",
    "result_missing_index": "Saved results have no matching result_index row.",
    "result_index_orphan": "result_index rows have no matching saved result.",
    "result_index_mismatch": "result_index metadata disagrees with the saved result payload.",
    "result_threshold_invalid": "Effective result thresholds must be finite probabilities between 0 and 1.",
    "result_threshold_orphan": "Effective result thresholds have no matching saved result.",
    "result_metadata_invalid": "Derived result validation metadata is malformed or uses an unsupported format.",
    "result_metadata_mismatch": "Derived result validation metadata disagrees with its evidence payload.",
    "result_metadata_orphan": "Derived result validation metadata has no matching saved result.",
    "result_payload_invalid": "Saved result payloads cannot be decoded or contain an invalid inference record, evidence rows, or values.",
    "state_payload_invalid": "Non-result application state contains an unreadable or structurally invalid compressed payload.",
    "snapshot_proof_missing": "Publication references an inference proof missing from active results and the archive.",
    "checkpoint_analysis_progress": "Checkpoint analysis disagrees with processed frames or label dimensions.",
    "checkpoint_orphan_frames": "Checkpoint frame chunks have no parent checkpoint.",
    "checkpoint_orphan_analysis": "Checkpoint analysis rows have no parent checkpoint.",
    "checkpoint_dimensions": "Checkpoint progress/top-row dimensions are impossible.",
    "checkpoint_frame_progress": "Checkpoint frame chunks have invalid progress relative to their parent checkpoint.",
    "checkpoint_scores": "Checkpoint top-score BLOBs are corrupt, have an impossible float32 shape, or contain invalid probabilities.",
    "cleanup_state_invalid": "Cleanup state has an invalid generation or mask structure.",
    "recovery_cleanup_invalid": "Recovery cleanup metadata is invalid or refers to an impossible generation.",
    "empty_snapshot_without_recovery": "Snapshot rows contain neither published nor pending state and have no archived recovery publication.",
    "recovery_publication_without_snapshot": "Archived publication recovery exists without a matching snapshot row.",
}


def _issue_list(counter):
    return [
        {"code": code, "count": count, "message": CHECK_MESSAGES[code]}
        for code, count in sorted(counter.items()) if count
    ]


def check_database(path, scope, *, show_progress=False):
    """Deep read-only consistency check across every scope in the state DB."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")

    from stash_ai.storage.schema import APPLICATION_ID, LATEST_SCHEMA_VERSION, shape_issues

    errors, warnings = Counter(), Counter()
    with connect_readonly(path) as db:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or version not in KNOWN_VERSIONS:
            raise ValueError("Not a supported stash-taggers state DB.")

        # Keep all relation/decode checks on one read snapshot even if another
        # process commits through WAL while this command is running.
        db.execute("BEGIN")
        tables = table_names(db)
        # Historical payload adapters belong exclusively to offline migration.
        # Diagnose the version/shape before running queries against current SQL.
        preflight = (Counter(schema_upgrade_required=1) if version != LATEST_SCHEMA_VERSION
                     else shape_issues(db))
        retired_tables = tables & {'legacy_imports', 'refinement_progress'}
        if version == LATEST_SCHEMA_VERSION and retired_tables:
            preflight['retired_state_tables'] = len(retired_tables)
        if preflight:
            integrity_ok = [row[0] for row in db.execute('PRAGMA integrity_check')] == ['ok']
            if not integrity_ok:
                preflight['sqlite_integrity_check'] += 1
            return {
                'path': str(path), 'sqlite_version': sqlite3.sqlite_version,
                'schema_version': version, 'integrity_check_ok': integrity_ok, 'ok': False,
                'errors': _issue_list(preflight), 'warnings': [],
                'error_count': sum(preflight.values()), 'warning_count': 0,
                **dict.fromkeys(('known_scopes', 'current_scope_results', 'results_checked',
                                 'state_payloads_checked', 'pending_publications', 'pending_cleanups',
                                 'refresh_requests', 'checkpoints'), 0),
            }
        progress = None
        if show_progress:
            from tqdm import tqdm
            progress = tqdm(
                total=validation_common.check_work_units(db, tables),
                desc="DB check",
                unit="item",
                dynamic_ncols=True,
            )
        try:
            if progress is not None:
                progress.set_postfix_str("SQLite integrity", refresh=True)
            integrity_rows = [row[0] for row in db.execute("PRAGMA integrity_check")]
            integrity_ok = integrity_rows == ["ok"]
            if not integrity_ok:
                errors["sqlite_integrity_check"] += 1
            validation_common.tick(progress)

            schema_checks.check_schema(db, tables, version, errors, warnings, progress)

            if progress is not None:
                progress.set_postfix_str("results/index", refresh=True)
            results_checked = result_checks.check_results_and_index(db, tables, errors, progress)
            result_checks.check_thresholds(db, tables, errors)
            validation_common.tick(progress)

            if progress is not None:
                progress.set_postfix_str("state payloads", refresh=True)
            state_payloads_checked = state_checks.check_state_payloads(db, tables, errors, progress)

            if progress is not None:
                progress.set_postfix_str("checkpoints", refresh=True)
            checkpoint_checks.check_checkpoints(db, tables, errors, progress)
            validation_common.tick(progress)

            if progress is not None:
                progress.set_postfix_str("cleanup/recovery", refresh=True)
            state_checks.check_cleanup_state(db, tables, errors, progress)

            if progress is not None:
                progress.set_postfix_str("relations", refresh=True)
            state_checks.check_state_relations(db, tables, warnings)
            validation_common.tick(progress)

            scopes = (
                db.execute("SELECT COUNT(DISTINCT scope) FROM results").fetchone()[0]
                if "results" in tables else 0
            )
            current_scope_results = (
                db.execute("SELECT COUNT(*) FROM results WHERE scope=?", (scope,)).fetchone()[0]
                if "results" in tables else 0
            )
            pending_publications = count_rows(db, tables, "snapshots", None, "pending IS NOT NULL")
            pending_cleanups = count_rows(db, tables, "cleanup_state", None, "pending IS NOT NULL")
            refresh_requests = count_rows(db, tables, "refresh_requests")
            checkpoints = count_rows(db, tables, "checkpoints")
            validation_common.tick(progress, phase="summary")
        finally:
            if progress is not None:
                progress.close()

    error_rows = _issue_list(errors)
    warning_rows = _issue_list(warnings)
    return {
        "path": str(path),
        "sqlite_version": sqlite3.sqlite_version,
        "schema_version": version,
        "integrity_check_ok": integrity_ok,
        "ok": not error_rows,
        "errors": error_rows,
        "warnings": warning_rows,
        "error_count": sum(item["count"] for item in error_rows),
        "warning_count": sum(item["count"] for item in warning_rows),
        "known_scopes": scopes,
        "current_scope_results": current_scope_results,
        "results_checked": results_checked,
        "state_payloads_checked": state_payloads_checked,
        "pending_publications": pending_publications,
        "pending_cleanups": pending_cleanups,
        "refresh_requests": refresh_requests,
        "checkpoints": checkpoints,
    }


def format_check_report(report):
    status = "ok" if report["ok"] else "FAILED"
    lines = [
        f"State DB consistency: {status}",
        f"State DB: {report['path']}",
        f"Schema: v{report['schema_version']} | SQLite {report['sqlite_version']}",
        (
            f"Checked: {report['results_checked']:,} result row(s), "
            f"{report['state_payloads_checked']:,} other state payload(s), "
            f"{report['known_scopes']:,} scope(s)"
        ),
        (
            "Live state: "
            f"pending publications={report['pending_publications']:,}, "
            f"pending cleanups={report['pending_cleanups']:,}, "
            f"refresh requests={report['refresh_requests']:,}, "
            f"checkpoints={report['checkpoints']:,}"
        ),
    ]
    if report["errors"]:
        lines.append("Errors:")
        lines.extend(
            f"  - {item['code']}: {item['count']:,} — {item['message']}"
            for item in report["errors"]
        )
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(
            f"  - {item['code']}: {item['count']:,} — {item['message']}"
            for item in report["warnings"]
        )
    if report["ok"] and not report["warnings"]:
        lines.append("No logical consistency problems found.")
    return "\n".join(lines)
