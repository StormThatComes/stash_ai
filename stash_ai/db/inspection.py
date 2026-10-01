"""Read-only database health, space usage and aggregate report formatting."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from .common import (
    BLOB_COLUMNS, KNOWN_VERSIONS, column_names, connect_readonly, count_rows,
    format_bytes, table_names,
)


def _payload_bytes(db, tables, table, scope):
    columns = BLOB_COLUMNS.get(table, ())
    if table not in tables or not columns:
        return 0
    available = column_names(db, table)
    columns = tuple(column for column in columns if column in available)
    if not columns:
        return 0
    expression = " + ".join(f"COALESCE(length({column}),0)" for column in columns)
    sql = f"SELECT COALESCE(SUM({expression}),0) FROM {table}"
    params = ()
    if "scope" in available:
        sql += " WHERE scope=?"
        params = (scope,)
    return int(db.execute(sql, params).fetchone()[0] or 0)


def inspect_database(path, scope):
    """Return privacy-safe aggregate health information without modifying the DB."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")

    from stash_ai.state import APPLICATION_ID

    with connect_readonly(path) as db:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or version not in KNOWN_VERSIONS:
            raise ValueError("Not a supported stash-taggers state DB.")

        tables = table_names(db)
        quick_check = db.execute("PRAGMA quick_check(1)").fetchone()[0]
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        page_count = db.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = db.execute("PRAGMA freelist_count").fetchone()[0]
        allocated = page_size * page_count
        free_pages = page_size * freelist_count

        results = []
        if "results" in tables:
            results = [
                dict(model=model, kind=kind, rows=rows, scenes=scenes, payload_bytes=int(payload or 0))
                for model, kind, rows, scenes, payload in db.execute(
                    """SELECT model,kind,COUNT(*),COUNT(DISTINCT scene),COALESCE(SUM(length(data)),0)
                       FROM results WHERE scope=? GROUP BY model,kind ORDER BY model,kind""",
                    (scope,))
            ]

        scoped_payload = {
            table: _payload_bytes(db, tables, table, scope)
            for table in BLOB_COLUMNS if table in tables
        }
        scenes = (db.execute("SELECT COUNT(DISTINCT scene) FROM results WHERE scope=?", (scope,)).fetchone()[0]
                  if "results" in tables else 0)
        scopes = (db.execute("SELECT COUNT(DISTINCT scope) FROM results").fetchone()[0]
                  if "results" in tables else 0)

        scope_pending_publications = count_rows(db, tables, "snapshots", scope, "pending IS NOT NULL")
        scope_pending_cleanups = count_rows(db, tables, "cleanup_state", scope, "pending IS NOT NULL")
        pending_publications = count_rows(db, tables, "snapshots", None, "pending IS NOT NULL")
        pending_cleanups = count_rows(db, tables, "cleanup_state", None, "pending IS NOT NULL")
        recovery_publications = count_rows(db, tables, "recovery_publications", scope)
        recovery_cleanups = count_rows(db, tables, "recovery_cleanups", scope)
        checkpoints = count_rows(db, tables, "checkpoints", scope)
        checkpoint_frames = count_rows(db, tables, "checkpoint_frames", scope)
        refresh_requests = count_rows(db, tables, "refresh_requests", scope)
        merge_checks = count_rows(db, tables, "merge_checks", scope)
        result_thresholds = count_rows(db, tables, "result_thresholds", scope)
        result_metadata = count_rows(db, tables, "result_metadata", scope)

    wal_path = Path(str(path) + "-wal")
    shm_path = Path(str(path) + "-shm")
    file_bytes = path.stat().st_size
    warnings = []
    if quick_check != "ok":
        warnings.append("SQLite quick_check failed.")
    if pending_publications:
        warnings.append(f"{pending_publications} publication plan(s) are pending.")
    if pending_cleanups:
        warnings.append(f"{pending_cleanups} cleanup plan(s) are pending.")
    if allocated and free_pages / allocated >= .20:
        warnings.append("At least 20% of allocated pages are free; compaction may reclaim meaningful space.")

    return dict(
        path=str(path),
        sqlite_version=sqlite3.sqlite_version,
        schema_version=version,
        quick_check=quick_check,
        file_bytes=file_bytes,
        wal_bytes=wal_path.stat().st_size if wal_path.exists() else 0,
        shm_bytes=shm_path.stat().st_size if shm_path.exists() else 0,
        page_size=page_size,
        page_count=page_count,
        freelist_count=freelist_count,
        allocated_bytes=allocated,
        free_page_bytes=free_pages,
        live_page_bytes=max(0, allocated - free_pages),
        reclaimable_estimate_bytes=free_pages,
        known_scopes=scopes,
        scenes=scenes,
        results=results,
        scoped_payload_bytes=scoped_payload,
        pending_publications=pending_publications,
        pending_cleanups=pending_cleanups,
        scope_pending_publications=scope_pending_publications,
        scope_pending_cleanups=scope_pending_cleanups,
        recovery_publications=recovery_publications,
        recovery_cleanups=recovery_cleanups,
        checkpoints=checkpoints,
        checkpoint_frames=checkpoint_frames,
        refresh_requests=refresh_requests,
        merge_checks=merge_checks,
        result_thresholds=result_thresholds,
        result_metadata=result_metadata,
        warnings=warnings,
    )


def format_report(report):
    lines = [
        f"State DB: {report['path']}",
        f"Health: {'ok' if report['quick_check'] == 'ok' else report['quick_check']}",
        f"Schema: v{report['schema_version']} | SQLite {report['sqlite_version']}",
        (
            f"File: {format_bytes(report['file_bytes'])} | allocated pages {format_bytes(report['allocated_bytes'])} | "
            f"free pages {format_bytes(report['free_page_bytes'])} ({report['freelist_count']:,})"
        ),
    ]
    if report["wal_bytes"] or report["shm_bytes"]:
        lines.append(f"Sidecars: WAL {format_bytes(report['wal_bytes'])} | SHM {format_bytes(report['shm_bytes'])}")
    lines.extend([
        f"Current scope: {report['scenes']:,} scene(s) with evidence; database scopes={report['known_scopes']:,}",
        "Results:",
    ])
    if report["results"]:
        for item in report["results"]:
            lines.append(
                f"  {item['model']:<8} {item['kind']:<7} scenes={item['scenes']:,} "
                f"rows={item['rows']:,} payload={format_bytes(item['payload_bytes'])}"
            )
    else:
        lines.append("  (none)")
    lines.extend([
        (
            "State (current scope): "
            f"pending publications={report['scope_pending_publications']:,}, "
            f"pending cleanups={report['scope_pending_cleanups']:,}, "
            f"recovery publications={report['recovery_publications']:,}, "
            f"recovery cleanups={report['recovery_cleanups']:,}"
        ),
        (
            "Database-wide pending: "
            f"publications={report['pending_publications']:,}, "
            f"cleanups={report['pending_cleanups']:,}"
        ),
        (
            "Work cache: "
            f"checkpoints={report['checkpoints']:,}, checkpoint chunks={report['checkpoint_frames']:,}, "
            f"refresh requests={report['refresh_requests']:,}, merge checks={report['merge_checks']:,}, "
            f"result thresholds={report['result_thresholds']:,}, result metadata={report['result_metadata']:,}"
        ),
        f"Estimated immediately reclaimable pages: {format_bytes(report['reclaimable_estimate_bytes'])}",
    ])
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend("  - " + warning for warning in report["warnings"])
    return "\n".join(lines)
