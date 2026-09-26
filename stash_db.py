"""Read-only state database health checks and safe SQLite compaction."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import zlib


KNOWN_VERSIONS = frozenset(range(1, 8))
KNOWN_MODELS = frozenset(("joytag", "wd", "dtq", "oppai"))
KNOWN_KINDS = frozenset(("scene", "markers"))
STATE_TABLES = frozenset((
    "results", "snapshots", "legacy_imports", "result_index", "refresh_requests",
    "checkpoints", "checkpoint_frames", "checkpoint_analysis", "cleanup_state",
    "merge_checks", "recovery_publications", "recovery_cleanups",
))
BASE_TABLES = frozenset(("results", "snapshots", "legacy_imports"))
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



CHECK_MESSAGES = {
    "sqlite_integrity_check": "SQLite integrity_check reported a storage/index-level problem.",
    "missing_base_tables": "Core state tables are missing.",
    "missing_state_tables": "Tables used by the current application are missing; a normal writer run may need to migrate/recreate them.",
    "invalid_model": "Rows contain an unknown model key.",
    "invalid_kind": "Rows contain an unknown output kind.",
    "result_missing_index": "Saved results have no matching result_index row.",
    "result_index_orphan": "result_index rows have no matching saved result.",
    "result_index_mismatch": "result_index metadata disagrees with the saved result payload.",
    "result_payload_invalid": "Saved result payloads cannot be decoded or violate the persisted segment shape.",
    "state_payload_invalid": "Non-result application state contains an unreadable or structurally invalid compressed payload.",
    "checkpoint_orphan_frames": "Checkpoint frame chunks have no parent checkpoint.",
    "checkpoint_orphan_analysis": "Checkpoint analysis rows have no parent checkpoint.",
    "checkpoint_dimensions": "Checkpoint progress/top-row dimensions are impossible.",
    "checkpoint_frame_progress": "Checkpoint frame chunks have invalid progress relative to their parent checkpoint.",
    "checkpoint_scores": "Checkpoint top-score BLOBs are corrupt or have an impossible float32 shape.",
    "cleanup_state_invalid": "Cleanup state has an invalid generation or mask structure.",
    "recovery_cleanup_invalid": "Recovery cleanup metadata is invalid or refers to an impossible generation.",
    "empty_snapshot_without_recovery": "Snapshot rows contain neither published nor pending state and have no archived recovery publication.",
    "legacy_import_without_result": "Legacy import history has no matching saved evidence for that scene/output kind.",
    "recovery_publication_without_snapshot": "Archived publication recovery exists without a matching snapshot row.",
}


def _issue_list(counter):
    return [
        {"code": code, "count": count, "message": CHECK_MESSAGES[code]}
        for code, count in sorted(counter.items()) if count
    ]


def _invalid_enum_count(db, tables, table, column, allowed):
    if table not in tables:
        return 0
    placeholders = ",".join("?" for _ in allowed)
    return db.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} NOT IN ({placeholders})",
        tuple(sorted(allowed)),
    ).fetchone()[0]


def _check_work_units(db, tables):
    payload_specs = (
        ("snapshots", "published"), ("snapshots", "pending"),
        ("legacy_imports", "data"), ("cleanup_state", "data"),
        ("checkpoint_frames", "data"), ("checkpoint_analysis", "data"),
        ("recovery_publications", "data"),
    )
    total = 6  # integrity, domains, result relations, checkpoint relations, state relations, summary
    total += _count(db, tables, "results")
    total += _count(db, tables, "checkpoints")
    total += _count(db, tables, "cleanup_state")
    total += _count(db, tables, "recovery_cleanups")
    for table, column in payload_specs:
        if table in tables and column in _columns(db, table):
            total += db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL"
            ).fetchone()[0]
    return total


def _tick(progress, amount=1, phase=None):
    if progress is None:
        return
    if phase is not None:
        progress.set_postfix_str(phase, refresh=False)
    progress.update(amount)


def _decoded_payload(blob, expected):
    from stash_state_codec import decode
    value = decode(blob)
    if not isinstance(value, expected):
        raise ValueError("Unexpected persisted payload shape.")
    return value


def _valid_result_segments(segments, kind):
    if not isinstance(segments, list):
        return False
    allowed_modes = {"scene"} if kind == "scene" else {"frames", "spans"}
    return all(
        isinstance(segment, dict)
        and segment.get("mode") in allowed_modes
        and isinstance(segment.get("meta"), dict)
        and isinstance(segment.get("items"), list)
        for segment in segments
    )


def _check_results_and_index(db, tables, errors, progress=None):
    if "results" not in tables:
        return 0

    indexed_signature = None
    checked = 0
    query = """
        SELECT r.scope,r.scene,r.model,r.kind,r.data,
               i.inference,i.populated,i.signature
        FROM results AS r
        LEFT JOIN result_index AS i
          ON i.scope=r.scope AND i.scene=r.scene AND i.model=r.model AND i.kind=r.kind
    """ if "result_index" in tables else """
        SELECT r.scope,r.scene,r.model,r.kind,r.data,NULL,NULL,NULL
        FROM results AS r
    """
    for _, _, _, kind, blob, inference, populated, signature in db.execute(query):
        checked += 1
        _tick(progress)
        try:
            segments = _decoded_payload(blob, list)
            if not _valid_result_segments(segments, kind):
                raise ValueError("Invalid result segment.")
            current = next(
                (segment for segment in reversed(segments)
                 if segment["meta"].get("origin") == "inference"),
                None,
            )
            expected_inference = int(current is not None)
            expected_populated = int(any(segment["items"] for segment in segments))
            if current is not None:
                if indexed_signature is None:
                    from stash_thresholds import indexed_signature as signature_for
                    indexed_signature = signature_for
                expected_signature = indexed_signature(current)
            else:
                expected_signature = None
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
            errors["result_payload_invalid"] += 1
            continue

        if inference is None and "result_index" in tables:
            errors["result_missing_index"] += 1
        elif (inference not in (0, 1) or populated not in (0, 1)
              or inference != expected_inference or populated != expected_populated
              or signature != expected_signature):
            errors["result_index_mismatch"] += 1

    if "result_index" in tables:
        errors["result_index_orphan"] += db.execute("""
            SELECT COUNT(*) FROM result_index AS i
            WHERE NOT EXISTS (
                SELECT 1 FROM results AS r
                WHERE r.scope=i.scope AND r.scene=i.scene
                  AND r.model=i.model AND r.kind=i.kind
            )
        """).fetchone()[0]
    return checked


def _check_state_payloads(db, tables, errors, progress=None):
    specs = (
        ("snapshots", "published", dict),
        ("snapshots", "pending", dict),
        ("legacy_imports", "data", list),
        ("cleanup_state", "data", dict),
        ("checkpoint_frames", "data", list),
        ("checkpoint_analysis", "data", dict),
        ("recovery_publications", "data", dict),
    )
    checked = 0
    for table, column, expected in specs:
        if table not in tables or column not in _columns(db, table):
            continue
        for blob, in db.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"):
            checked += 1
            _tick(progress)
            try:
                _decoded_payload(blob, expected)
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
                errors["state_payload_invalid"] += 1
    return checked


def _check_checkpoints(db, tables, errors, progress=None):
    if "checkpoints" in tables:
        errors["checkpoint_dimensions"] += db.execute("""
            SELECT COUNT(*) FROM checkpoints
            WHERE processed <= 0 OR total <= 0 OR processed > total
               OR top_rows < 0 OR top_rows > processed
               OR signature = ''
        """).fetchone()[0]

        for top_rows, blob in db.execute("SELECT top_rows,top_scores FROM checkpoints"):
            _tick(progress)
            try:
                raw = zlib.decompress(blob)
                valid = (
                    (top_rows == 0 and len(raw) == 0)
                    or (top_rows > 0 and len(raw) > 0 and len(raw) % (top_rows * 4) == 0)
                )
                if not valid:
                    errors["checkpoint_scores"] += 1
            except (TypeError, zlib.error):
                errors["checkpoint_scores"] += 1

    if "checkpoint_frames" in tables:
        if "checkpoints" in tables:
            errors["checkpoint_orphan_frames"] += db.execute("""
                SELECT COUNT(*) FROM checkpoint_frames AS f
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints AS c
                    WHERE c.scope=f.scope AND c.scene=f.scene AND c.model=f.model
                )
            """).fetchone()[0]
            errors["checkpoint_frame_progress"] += db.execute("""
                SELECT COUNT(*) FROM checkpoint_frames AS f
                JOIN checkpoints AS c
                  ON c.scope=f.scope AND c.scene=f.scene AND c.model=f.model
                WHERE f.processed <= 0 OR f.processed > c.processed
            """).fetchone()[0]
        else:
            errors["checkpoint_orphan_frames"] += _count(db, tables, "checkpoint_frames")

    if "checkpoint_analysis" in tables:
        if "checkpoints" in tables:
            errors["checkpoint_orphan_analysis"] += db.execute("""
                SELECT COUNT(*) FROM checkpoint_analysis AS a
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints AS c
                    WHERE c.scope=a.scope AND c.scene=a.scene AND c.model=a.model
                )
            """).fetchone()[0]
        else:
            errors["checkpoint_orphan_analysis"] += _count(db, tables, "checkpoint_analysis")


def _check_cleanup_state(db, tables, errors, progress=None):
    generations = {}
    if "cleanup_state" in tables:
        for scope, generation, blob in db.execute("SELECT scope,generation,data FROM cleanup_state"):
            _tick(progress)
            generations[scope] = generation
            invalid = type(generation) is not int or generation < 0
            try:
                masks = _decoded_payload(blob, dict)
                for values in masks.values():
                    if not isinstance(values, dict):
                        invalid = True
                        continue
                    for revision in values.values():
                        if type(revision) is not int or revision <= 0 or revision > generation:
                            invalid = True
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
                # Counted by the generic BLOB check as well; this code reports
                # the higher-level cleanup invariant, not a second payload error.
                invalid = True
            errors["cleanup_state_invalid"] += int(invalid)

    if "recovery_cleanups" in tables:
        for scope, generation, kinds in db.execute(
                "SELECT scope,generation,kinds FROM recovery_cleanups"):
            _tick(progress)
            invalid = type(generation) is not int or generation <= 0
            try:
                decoded = json.loads(kinds)
                invalid |= (
                    not isinstance(decoded, list)
                    or not decoded
                    or any(kind not in KNOWN_KINDS for kind in decoded)
                    or len(decoded) != len(set(decoded))
                )
            except (TypeError, json.JSONDecodeError):
                invalid = True
            current = generations.get(scope)
            if current is None or generation > current:
                invalid = True
            errors["recovery_cleanup_invalid"] += int(invalid)


def _check_state_relations(db, tables, warnings):
    if "snapshots" in tables:
        if "recovery_publications" in tables:
            warnings["empty_snapshot_without_recovery"] += db.execute("""
                SELECT COUNT(*) FROM snapshots AS s
                WHERE s.published IS NULL AND s.pending IS NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM recovery_publications AS r
                    WHERE r.scope=s.scope AND r.scene=s.scene AND r.kind=s.kind
                  )
            """).fetchone()[0]
        else:
            warnings["empty_snapshot_without_recovery"] += db.execute(
                "SELECT COUNT(*) FROM snapshots WHERE published IS NULL AND pending IS NULL"
            ).fetchone()[0]

    if "legacy_imports" in tables and "results" in tables:
        warnings["legacy_import_without_result"] += db.execute("""
            SELECT COUNT(*) FROM legacy_imports AS l
            WHERE NOT EXISTS (
                SELECT 1 FROM results AS r
                WHERE r.scope=l.scope AND r.scene=l.scene AND r.kind=l.kind
            )
        """).fetchone()[0]

    if "recovery_publications" in tables:
        if "snapshots" in tables:
            warnings["recovery_publication_without_snapshot"] += db.execute("""
                SELECT COUNT(*) FROM recovery_publications AS r
                WHERE NOT EXISTS (
                    SELECT 1 FROM snapshots AS s
                    WHERE s.scope=r.scope AND s.scene=r.scene AND s.kind=r.kind
                )
            """).fetchone()[0]
        else:
            warnings["recovery_publication_without_snapshot"] += _count(
                db, tables, "recovery_publications")


def check_database(path, scope, *, show_progress=False):
    """Deep read-only consistency check across every scope in the state DB."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")

    from stash_merge import APPLICATION_ID

    errors, warnings = Counter(), Counter()
    with _readonly(path) as db:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or version not in KNOWN_VERSIONS:
            raise ValueError("Not a supported stash-taggers state DB.")

        # Keep all relation/decode checks on one read snapshot even if another
        # process commits through WAL while this command is running.
        db.execute("BEGIN")
        tables = _tables(db)
        progress = None
        if show_progress:
            from tqdm import tqdm
            progress = tqdm(
                total=_check_work_units(db, tables),
                desc="DB check",
                unit="item",
                dynamic_ncols=True,
            )
        try:
            if progress is not None:
                progress.set_postfix_str("SQLite integrity", refresh=True)
            integrity_rows = [row[0] for row in db.execute("PRAGMA integrity_check")]
            integrity_ok = integrity_rows == ["ok"]
            if not integrity_ok:
                errors["sqlite_integrity_check"] += 1
            _tick(progress)

            missing = STATE_TABLES - tables
            errors["missing_base_tables"] += len(BASE_TABLES - tables)
            warnings["missing_state_tables"] += len(missing - BASE_TABLES)

            enum_specs = (
                ("results", "model", KNOWN_MODELS),
                ("result_index", "model", KNOWN_MODELS),
                ("refresh_requests", "model", KNOWN_MODELS),
                ("checkpoints", "model", KNOWN_MODELS),
                ("checkpoint_frames", "model", KNOWN_MODELS),
                ("checkpoint_analysis", "model", KNOWN_MODELS),
                ("results", "kind", KNOWN_KINDS),
                ("result_index", "kind", KNOWN_KINDS),
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
            _tick(progress)

            if progress is not None:
                progress.set_postfix_str("results/index", refresh=True)
            results_checked = _check_results_and_index(db, tables, errors, progress)
            _tick(progress)

            if progress is not None:
                progress.set_postfix_str("state payloads", refresh=True)
            state_payloads_checked = _check_state_payloads(db, tables, errors, progress)

            if progress is not None:
                progress.set_postfix_str("checkpoints", refresh=True)
            _check_checkpoints(db, tables, errors, progress)
            _tick(progress)

            if progress is not None:
                progress.set_postfix_str("cleanup/recovery", refresh=True)
            _check_cleanup_state(db, tables, errors, progress)

            if progress is not None:
                progress.set_postfix_str("relations", refresh=True)
            _check_state_relations(db, tables, warnings)
            _tick(progress)

            scopes = (
                db.execute("SELECT COUNT(DISTINCT scope) FROM results").fetchone()[0]
                if "results" in tables else 0
            )
            current_scope_results = (
                db.execute("SELECT COUNT(*) FROM results WHERE scope=?", (scope,)).fetchone()[0]
                if "results" in tables else 0
            )
            pending_publications = _count(db, tables, "snapshots", None, "pending IS NOT NULL")
            pending_cleanups = _count(db, tables, "cleanup_state", None, "pending IS NOT NULL")
            refresh_requests = _count(db, tables, "refresh_requests")
            checkpoints = _count(db, tables, "checkpoints")
            _tick(progress, phase="summary")
        finally:
            if progress is not None:
                progress.close()

    error_rows = _issue_list(errors)
    warning_rows = _issue_list(warnings)
    return {
        "path": str(path),
        "sqlite_version": sqlite3.sqlite_version,
        "schema_version": version,
        "integrity_check_ok": integrity_ok,
        "ok": not error_rows,
        "errors": error_rows,
        "warnings": warning_rows,
        "error_count": sum(item["count"] for item in error_rows),
        "warning_count": sum(item["count"] for item in warning_rows),
        "known_scopes": scopes,
        "current_scope_results": current_scope_results,
        "results_checked": results_checked,
        "state_payloads_checked": state_payloads_checked,
        "pending_publications": pending_publications,
        "pending_cleanups": pending_cleanups,
        "refresh_requests": refresh_requests,
        "checkpoints": checkpoints,
    }


def format_check_report(report):
    status = "ok" if report["ok"] else "FAILED"
    lines = [
        f"State DB consistency: {status}",
        f"State DB: {report['path']}",
        f"Schema: v{report['schema_version']} | SQLite {report['sqlite_version']}",
        (
            f"Checked: {report['results_checked']:,} result row(s), "
            f"{report['state_payloads_checked']:,} other state payload(s), "
            f"{report['known_scopes']:,} scope(s)"
        ),
        (
            "Live state: "
            f"pending publications={report['pending_publications']:,}, "
            f"pending cleanups={report['pending_cleanups']:,}, "
            f"refresh requests={report['refresh_requests']:,}, "
            f"checkpoints={report['checkpoints']:,}"
        ),
    ]
    if report["errors"]:
        lines.append("Errors:")
        lines.extend(
            f"  - {item['code']}: {item['count']:,} — {item['message']}"
            for item in report["errors"]
        )
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(
            f"  - {item['code']}: {item['count']:,} — {item['message']}"
            for item in report["warnings"]
        )
    if report["ok"] and not report["warnings"]:
        lines.append("No logical consistency problems found.")
    return "\n".join(lines)


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
        if args.db_check:
            report = check_database(args.state_db, scope, show_progress=True)
            print(format_check_report(report))
            return 0 if report["ok"] else 1

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
