"""Allowlisted labels and aggregate counters for anonymous diagnostics."""
from stash_ai.db.common import KNOWN_KINDS, KNOWN_MODELS


def _safe_model(value):
    return value if value in KNOWN_MODELS else "other"


def _safe_kind(value):
    return value if value in KNOWN_KINDS else "other"


def _row_count(db, table):
    return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _safe_counter(counter, allowed=None):
    result = {}
    for key, count in counter.items():
        name = str(key)
        if allowed is not None and key not in allowed:
            name = "other"
        result[name] = result.get(name, 0) + count
    return dict(sorted(result.items()))
