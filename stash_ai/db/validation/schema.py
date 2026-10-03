"""Validate required tables and model/output domains."""
from __future__ import annotations

from ..common import BASE_TABLES, KNOWN_KINDS, KNOWN_MODELS, STATE_TABLES
from .common import tick


def _invalid_enum_count(db, tables, table, column, allowed):
    if table not in tables:
        return 0
    placeholders = ",".join("?" for _ in allowed)
    return db.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} NOT IN ({placeholders})",
        tuple(sorted(allowed)),
    ).fetchone()[0]


def check_schema(db, tables, version, errors, warnings, progress=None):
    # Validation metadata is a lazy derived cache; older databases may lack it.
    missing = STATE_TABLES - tables - {'result_metadata'}
    required = BASE_TABLES | {'result_thresholds'} if version >= 8 else BASE_TABLES
    errors["missing_base_tables"] += len(required - tables)
    warnings["missing_state_tables"] += len(missing - required)

    enum_specs = (
        ("results", "model", KNOWN_MODELS),
        ("result_index", "model", KNOWN_MODELS),
        ("result_thresholds", "model", KNOWN_MODELS),
        ("result_metadata", "model", KNOWN_MODELS),
        ("refresh_requests", "model", KNOWN_MODELS),
        ("checkpoints", "model", KNOWN_MODELS),
        ("checkpoint_frames", "model", KNOWN_MODELS),
        ("checkpoint_analysis", "model", KNOWN_MODELS),
        ("results", "kind", KNOWN_KINDS),
        ("result_index", "kind", KNOWN_KINDS),
        ("result_thresholds", "kind", KNOWN_KINDS),
        ("result_metadata", "kind", KNOWN_KINDS),
        ("refresh_requests", "kind", KNOWN_KINDS),
        ("snapshots", "kind", KNOWN_KINDS),
        ("legacy_imports", "kind", KNOWN_KINDS),
        ("recovery_publications", "kind", KNOWN_KINDS),
    )
    if progress is not None:
        progress.set_postfix_str("domains", refresh=True)
    for table, column, allowed in enum_specs:
        count = _invalid_enum_count(db, tables, table, column, allowed)
        errors["invalid_model" if column == "model" else "invalid_kind"] += count
    tick(progress)
