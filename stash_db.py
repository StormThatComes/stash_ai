"""Read-only state database health checks and safe SQLite compaction."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sqlite3


KNOWN_VERSIONS = frozenset(range(1, 8))
MIN_FREE_HEADROOM = 16 * 2**20

BLOB_COLUMNS = {
    "results": ("data",),
    "snapshots": ("published", "pending"),
    "legacy_imports": ("data",),
    "cleanup_state": ("data",),
    "checkpoints": ("top_scores",),
    "checkpoint_frames": ("data",),
    "checkpoint_analysis": ("data",),
    "recovery_publications": ("data",),
}


def _bytes(value):
    value = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.2f} {suffix}"
        value /= 1024
    raise AssertionError("unreachable")


def _readonly(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _count(db, tables, table, scope=None, extra=""):
    if table not in tables:
        return 0
    sql = f"SELECT COUNT(*) FROM {table}"
    params = ()
    where = []
    if scope is not None and "scope" in _columns(db, table):
        where.append("scope=?")
        params = (scope,)
    if extra:
        where.append(extra)
    if where:
        sql += " WHERE " + " AND ".join(where)
    return db.execute(sql, params).fetchone()[0]


def _payload_bytes(db, tables, table, scope):
    columns = BLOB_COLUMNS.get(table, ())
    if table not in tables or not columns:
        return 0
    available = _columns(db, table)
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

    from stash_merge import APPLICATION_ID

    with _readonly(path) as db:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or version not in KNOWN_VERSIONS:
            raise ValueError("Not a supported stash-taggers state DB.")

        tables = _tables(db)
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

        scope_pending_publications = _count(db, tables, "snapshots", scope, "pending IS NOT NULL")
        scope_pending_cleanups = _count(db, tables, "cleanup_state", scope, "pending IS NOT NULL")
        pending_publications = _count(db, tables, "snapshots", None, "pending IS NOT NULL")
        pending_cleanups = _count(db, tables, "cleanup_state", None, "pending IS NOT NULL")
        recovery_publications = _count(db, tables, "recovery_publications", scope)
        recovery_cleanups = _count(db, tables, "recovery_cleanups", scope)
        checkpoints = _count(db, tables, "checkpoints", scope)
        checkpoint_frames = _count(db, tables, "checkpoint_frames", scope)
        refresh_requests = _count(db, tables, "refresh_requests", scope)
        merge_checks = _count(db, tables, "merge_checks", scope)

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
        warnings=warnings,
    )


def _same_logical_counts(before, after):
    fields = (
        "schema_version", "known_scopes", "scenes", "pending_publications", "pending_cleanups",
        "scope_pending_publications", "scope_pending_cleanups", "recovery_publications",
        "recovery_cleanups", "checkpoints", "checkpoint_frames",
        "refresh_requests", "merge_checks", "results", "scoped_payload_bytes",
    )
    return all(before[field] == after[field] for field in fields)


def _safe_to_compact(report):
    if report["quick_check"] != "ok":
        raise RuntimeError("SQLite quick_check is not clean; compacting a suspect database is refused.")
    if report["pending_publications"]:
        raise RuntimeError("Pending publication plans exist; finish or abandon them before --db-compact.")
    if report["pending_cleanups"]:
        raise RuntimeError("A cleanup plan is pending; finish or abandon it before --db-compact.")


@contextmanager
def _writer_lock(path):
    """Use the same advisory lock file as StateStore writers."""
    lock = open(str(path) + ".lock", "a+b")
    try:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another merged-output process is using this state DB.") from exc
        yield
    finally:
        lock.close()


def compact_database(path, scope, *, dry_run=False):
    """VACUUM a healthy idle DB, refusing unfinished publication/cleanup plans."""
    path = Path(path).expanduser().resolve()
    before = inspect_database(path, scope)
    _safe_to_compact(before)
    if dry_run:
        return before, before

    with _writer_lock(path):
        # Recheck after acquiring the application writer lock to close the race
        # between inspection and VACUUM.
        before = inspect_database(path, scope)
        _safe_to_compact(before)
        # SQLite VACUUM builds a replacement database beside the original.
        # Requiring roughly one additional full DB (plus WAL/headroom) follows
        # SQLite's documented worst-case temporary-space requirement.
        required = before["file_bytes"] + before["wal_bytes"] + MIN_FREE_HEADROOM
        available = shutil.disk_usage(path.parent).free
        if available < required:
            raise RuntimeError(
                f"Insufficient free disk space for safe VACUUM: need about {_bytes(required)}, "
                f"have {_bytes(available)}."
            )
        with sqlite3.connect(path, timeout=30) as db:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("VACUUM")
            check = db.execute("PRAGMA quick_check(1)").fetchone()[0]
            if check != "ok":
                raise RuntimeError("SQLite quick_check failed after VACUUM.")

    after = inspect_database(path, scope)
    if not _same_logical_counts(before, after):
        raise RuntimeError("Database aggregates changed unexpectedly during VACUUM; restore from backup and inspect the DB.")
    return before, after


def format_report(report):
    lines = [
        f"State DB: {report['path']}",
        f"Health: {'ok' if report['quick_check'] == 'ok' else report['quick_check']}",
        f"Schema: v{report['schema_version']} | SQLite {report['sqlite_version']}",
        (
            f"File: {_bytes(report['file_bytes'])} | allocated pages {_bytes(report['allocated_bytes'])} | "
            f"free pages {_bytes(report['free_page_bytes'])} ({report['freelist_count']:,})"
        ),
    ]
    if report["wal_bytes"] or report["shm_bytes"]:
        lines.append(f"Sidecars: WAL {_bytes(report['wal_bytes'])} | SHM {_bytes(report['shm_bytes'])}")
    lines.extend([
        f"Current scope: {report['scenes']:,} scene(s) with evidence; database scopes={report['known_scopes']:,}",
        "Results:",
    ])
    if report["results"]:
        for item in report["results"]:
            lines.append(
                f"  {item['model']:<8} {item['kind']:<7} scenes={item['scenes']:,} "
                f"rows={item['rows']:,} payload={_bytes(item['payload_bytes'])}"
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
            f"refresh requests={report['refresh_requests']:,}, merge checks={report['merge_checks']:,}"
        ),
        f"Estimated immediately reclaimable pages: {_bytes(report['reclaimable_estimate_bytes'])}",
    ])
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend("  - " + warning for warning in report["warnings"])
    return "\n".join(lines)


def run_mode(args, stash_url):
    from stash_merge import scope_for

    scope = scope_for(stash_url)
    try:
        if args.db_report:
            print(format_report(inspect_database(args.state_db, scope)))
            return 0

        before, after = compact_database(args.state_db, scope, dry_run=args.dry_run)
        print(format_report(before))
        if args.dry_run:
            print(
                "DRY RUN: VACUUM was not executed; estimated immediately reclaimable pages: "
                + _bytes(before["reclaimable_estimate_bytes"])
            )
        else:
            saved = max(0, before["file_bytes"] - after["file_bytes"])
            print(f"Compaction complete: {_bytes(before['file_bytes'])} -> {_bytes(after['file_bytes'])}; saved {_bytes(saved)}.")
        return 0
    except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError) as exc:
        print("Database maintenance failed: " + str(exc))
        return 1
