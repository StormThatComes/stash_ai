"""Bounded scene-local signal records shared by both evidence kinds."""
import math
from stash_ai.models import SOURCES


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _candidate(rows, model, tag):
    if model not in SOURCES:
        raise ValueError("Unknown model in review evidence.")
    return rows.setdefault((model, tag), {"reasons": set(), "evidence": []})


def _flag(row, reason, detail):
    row["reasons"].add(reason)
    # Keep scene-local evidence bounded even when a tag has hundreds of ranges.
    key = reason, detail["kind"], detail.get("peer_model")
    if not any((item["reason"], item["kind"], item.get("peer_model")) == key
               for item in row["evidence"]) and len(row["evidence"]) < 8:
        row["evidence"].append({"reason": reason, **detail})
