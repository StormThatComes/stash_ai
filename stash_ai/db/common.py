"""Shared schema facts and read-only SQLite helpers for database diagnostics."""

from __future__ import annotations

import sqlite3


KNOWN_VERSIONS = frozenset(range(1, 10))
KNOWN_MODELS = frozenset(("joytag", "wd", "dtq", "oppai"))
KNOWN_KINDS = frozenset(("scene", "markers"))
STATE_TABLES = frozenset((
    "results", "result_thresholds", "result_metadata", "snapshots", "legacy_imports", "result_index", "refresh_requests",
    "checkpoints", "checkpoint_frames", "checkpoint_analysis", "cleanup_state",
    "merge_checks", "recovery_publications", "recovery_cleanups",
))
BASE_TABLES = frozenset(("results", "snapshots", "legacy_imports"))

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


def format_bytes(value):
    value = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.2f} {suffix}"
        value /= 1024
    raise AssertionError("unreachable")


def connect_readonly(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def table_names(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def column_names(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def count_rows(db, tables, table, scope=None, extra=""):
    if table not in tables:
        return 0
    sql = f"SELECT COUNT(*) FROM {table}"
    params = ()
    where = []
    if scope is not None and "scope" in column_names(db, table):
        where.append("scope=?")
        params = (scope,)
    if extra:
        where.append(extra)
    if where:
        sql += " WHERE " + " AND ".join(where)
    return db.execute(sql, params).fetchone()[0]
