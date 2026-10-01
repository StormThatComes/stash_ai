"""Guarded SQLite compaction with the application's writer-lock protocol."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sqlite3

from .common import format_bytes
from .inspection import inspect_database


MIN_FREE_HEADROOM = 16 * 2**20


def _same_logical_counts(before, after):
    fields = (
        "schema_version", "known_scopes", "scenes", "pending_publications", "pending_cleanups",
        "scope_pending_publications", "scope_pending_cleanups", "recovery_publications",
        "recovery_cleanups", "checkpoints", "checkpoint_frames",
        "refresh_requests", "merge_checks", "result_thresholds", "result_metadata", "results", "scoped_payload_bytes",
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
                f"Insufficient free disk space for safe VACUUM: need about {format_bytes(required)}, "
                f"have {format_bytes(available)}."
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
