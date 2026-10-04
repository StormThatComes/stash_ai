"""Shared schema facts and read-only SQLite helpers for database diagnostics."""

from __future__ import annotations

import sqlite3

from stash_ai.storage.schema import (
    LATEST_SCHEMA_VERSION, KNOWN_MODELS, KNOWN_KINDS, STATE_TABLES, BASE_TABLES, BLOB_COLUMNS,
)

KNOWN_VERSIONS = frozenset(range(1, LATEST_SCHEMA_VERSION + 1))


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

