"""Validate required tables and model/output domains."""
from __future__ import annotations

from ..common import BASE_TABLES, KNOWN_KINDS, KNOWN_MODELS, STATE_TABLES
from .common import tick
from stash_ai.storage.schema import TABLES


def _invalid_enum_count(db, tables, table, column, allowed):
    if table not in tables:
        return 0
    placeholders = ",".join("?" for _ in allowed)
    return db.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} NOT IN ({placeholders})",
        tuple(sorted(allowed)),
    ).fetchone()[0]


def check_schema(db, tables, version, errors, warnings, progress=None):
    enum_specs = [(table.name, column, allowed) for table in TABLES
                  for column, enabled, allowed in (('model', table.model, KNOWN_MODELS),
                                                    ('kind', table.kind, KNOWN_KINDS))
                  if enabled]
    if progress is not None:
        progress.set_postfix_str("domains", refresh=True)
    for table, column, allowed in enum_specs:
        count = _invalid_enum_count(db, tables, table, column, allowed)
        errors["invalid_model" if column == "model" else "invalid_kind"] += count
    tick(progress)


