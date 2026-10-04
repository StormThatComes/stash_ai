"""Content-free developer diagnostics for the persistent state database.

The exported JSON intentionally contains no scene/scope identifiers, URLs, file
paths, tag names, scores, marker timestamps, inference signatures or hashes.
Only schema/runtime facts, aggregate counts and size/shape distributions leave
the database. The report is written locally and is never uploaded.
"""

from __future__ import annotations

from .database import database_report, _checkpoint_report, _grouped_counts
from .evidence import _result_evidence_report

import platform
from pathlib import Path
import sqlite3
import struct
import sys

from stash_ai.db.common import KNOWN_VERSIONS, connect_readonly, table_names
from stash_ai.reports.io import write_json_report

REPORT_FORMAT = 1


def build_report(path):
    """Inspect every scope but emit no scope value or per-scene record."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")

    from stash_ai.storage.store import APPLICATION_ID

    with connect_readonly(path) as db:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
        schema_version = db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or schema_version not in KNOWN_VERSIONS:
            raise ValueError("Not a supported stash-taggers state DB.")

        tables = table_names(db)
        database = database_report(path, db, tables, schema_version)

        evidence = _result_evidence_report(db, tables)
        checkpoints = _checkpoint_report(db, tables)
        grouped = _grouped_counts(db, tables)

    return {
        "format": REPORT_FORMAT,
        "privacy": {
            "automatic_upload": False,
            "contains_scene_ids": False,
            "contains_scope_values": False,
            "contains_urls": False,
            "contains_filesystem_paths": False,
            "contains_tag_names": False,
            "contains_scores": False,
            "contains_marker_timestamps": False,
            "contains_signatures_or_hashes": False,
            "aggregate_counts_and_sizes_describe_the_installation": True,
        },
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "sqlite": sqlite3.sqlite_version,
            "os_family": platform.system() or "unknown",
            "pointer_bits": struct.calcsize("P") * 8,
        },
        "database": database,
        "evidence": evidence,
        "checkpoints": checkpoints,
        "state_groups": grouped,
    }


def write_report(state_db, destination):
    destination = Path(destination).expanduser().resolve()
    state_db = Path(state_db).expanduser().resolve()
    if destination == state_db:
        raise ValueError("Diagnostics output must not replace the state database.")
    if destination.suffix.lower() != ".json":
        raise ValueError("Anonymous diagnostics output must use a .json filename.")

    report = build_report(state_db)
    destination = write_json_report(report, destination)
    return destination, report


def run(args):
    try:
        destination, report = write_report(args.state_db, args.anonymous_diagnostics)
        print(
            f"Anonymous developer diagnostics written to {destination}. "
            f"Report format={report['format']}; no upload was performed."
        )
        print(
            "Review the JSON before sharing it. It contains no direct content/identifiers, "
            "but aggregate counts and size distributions describe this installation."
        )
        return 0
    except (OSError, sqlite3.DatabaseError, ValueError, RuntimeError) as exc:
        print("Anonymous diagnostics failed: " + str(exc))
        return 1

