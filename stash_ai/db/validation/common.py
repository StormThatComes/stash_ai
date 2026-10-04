"""Payload decoding and progress accounting for database validation."""
from __future__ import annotations

from ..common import column_names, count_rows


def check_work_units(db, tables):
    payload_specs = (
        ("snapshots", "published"), ("snapshots", "pending"),
        ("legacy_imports", "data"), ("cleanup_state", "data"),
        ("checkpoint_frames", "data"), ("checkpoint_analysis", "data"),
        ("recovery_publications", "data"),
    )
    total = 6  # integrity, domains, result relations, checkpoint relations, state relations, summary
    total += count_rows(db, tables, "results")
    total += count_rows(db, tables, "checkpoints")
    total += count_rows(db, tables, "cleanup_state")
    total += count_rows(db, tables, "recovery_cleanups")
    for table, column in payload_specs:
        if table in tables and column in column_names(db, table):
            total += db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL"
            ).fetchone()[0]
    return total


def tick(progress, amount=1, phase=None):
    if progress is None:
        return
    if phase is not None:
        progress.set_postfix_str(phase, refresh=False)
    progress.update(amount)


def decoded_payload(blob, expected):
    from stash_ai.storage.codec import decode
    value = decode(blob)
    if not isinstance(value, expected):
        raise ValueError("Unexpected persisted payload shape.")
    return value
