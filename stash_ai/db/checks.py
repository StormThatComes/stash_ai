"""Deep read-only validation of persisted evidence and state relationships."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
import struct
import zlib

from .common import (
    BASE_TABLES, KNOWN_KINDS, KNOWN_MODELS, KNOWN_VERSIONS, STATE_TABLES,
    column_names, connect_readonly, count_rows, table_names,
)
from .payloads import valid_frames, valid_result_segments


CHECK_MESSAGES = {
    "sqlite_integrity_check": "SQLite integrity_check reported a storage/index-level problem.",
    "missing_base_tables": "Core state tables are missing.",
    "missing_state_tables": "Tables used by the current application are missing; a normal writer run may need to migrate/recreate them.",
    "invalid_model": "Rows contain an unknown model key.",
    "invalid_kind": "Rows contain an unknown output kind.",
    "result_missing_index": "Saved results have no matching result_index row.",
    "result_index_orphan": "result_index rows have no matching saved result.",
    "result_index_mismatch": "result_index metadata disagrees with the saved result payload.",
    "result_threshold_invalid": "Effective result thresholds must be finite probabilities between 0 and 1.",
    "result_threshold_orphan": "Effective result thresholds have no matching saved result.",
    "result_metadata_invalid": "Derived result validation metadata is malformed or uses an unsupported format.",
    "result_metadata_mismatch": "Derived result validation metadata disagrees with its evidence payload.",
    "result_metadata_orphan": "Derived result validation metadata has no matching saved result.",
    "result_payload_invalid": "Saved result payloads cannot be decoded or contain invalid segments, evidence rows, or values.",
    "state_payload_invalid": "Non-result application state contains an unreadable or structurally invalid compressed payload.",
    "checkpoint_orphan_frames": "Checkpoint frame chunks have no parent checkpoint.",
    "checkpoint_orphan_analysis": "Checkpoint analysis rows have no parent checkpoint.",
    "checkpoint_dimensions": "Checkpoint progress/top-row dimensions are impossible.",
    "checkpoint_frame_progress": "Checkpoint frame chunks have invalid progress relative to their parent checkpoint.",
    "checkpoint_scores": "Checkpoint top-score BLOBs are corrupt, have an impossible float32 shape, or contain invalid probabilities.",
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


def _tick(progress, amount=1, phase=None):
    if progress is None:
        return
    if phase is not None:
        progress.set_postfix_str(phase, refresh=False)
    progress.update(amount)


def _decoded_payload(blob, expected):
    from stash_ai.state_codec import decode
    value = decode(blob)
    if not isinstance(value, expected):
        raise ValueError("Unexpected persisted payload shape.")
    return value


def _check_results_and_index(db, tables, errors, progress=None):
    if "results" not in tables:
        return 0

    indexed_signature = None
    checked = 0
    index_columns = 'i.inference,i.populated,i.signature' if 'result_index' in tables else 'NULL,NULL,NULL'
    threshold_column = 't.threshold' if 'result_thresholds' in tables else 'NULL'
    metadata_columns = 'm.format,m.data' if 'result_metadata' in tables else 'NULL,NULL'
    query = f'SELECT r.kind,r.data,{index_columns},{threshold_column},{metadata_columns} FROM results AS r'
    if 'result_index' in tables:
        query += ' LEFT JOIN result_index AS i USING(scope,scene,model,kind)'
    if 'result_thresholds' in tables:
        query += ' LEFT JOIN result_thresholds AS t USING(scope,scene,model,kind)'
    if 'result_metadata' in tables:
        query += ' LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)'
    for kind, blob, inference, populated, signature, threshold, metadata_format, metadata in db.execute(query):
        checked += 1
        _tick(progress)
        try:
            segments = _decoded_payload(blob, list)
            if not valid_result_segments(segments, kind):
                raise ValueError("Invalid result segment.")
            if metadata_format is not None:
                from stash_ai.state_metadata import FORMAT, decode, encode, summarize
                try:
                    if metadata_format != FORMAT:
                        raise ValueError('Unsupported result metadata format.')
                    if encode(decode(metadata)) != encode(summarize(segments)):
                        errors['result_metadata_mismatch'] += 1
                except (TypeError, ValueError):
                    errors['result_metadata_invalid'] += 1
            if threshold is not None:
                from stash_ai.state_thresholds import overlay
                from stash_ai.thresholds import _score
                if _score(threshold):
                    overlay(segments, kind, threshold)
            current = next(
                (segment for segment in reversed(segments)
                 if segment["meta"].get("origin") == "inference"),
                None,
            )
            expected_inference = int(current is not None)
            expected_populated = int(any(segment["items"] for segment in segments))
            if current is not None:
                if indexed_signature is None:
                    from stash_ai.thresholds import indexed_signature as signature_for
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
    if 'result_metadata' in tables:
        errors['result_metadata_orphan'] += db.execute('''
            SELECT COUNT(*) FROM result_metadata AS m WHERE NOT EXISTS (
                SELECT 1 FROM results AS r WHERE r.scope=m.scope AND r.scene=m.scene
                    AND r.model=m.model AND r.kind=m.kind)
        ''').fetchone()[0]
    return checked


def _check_thresholds(db, tables, errors):
    if 'result_thresholds' not in tables:
        return
    errors['result_threshold_invalid'] += db.execute('''
        SELECT COUNT(*) FROM result_thresholds
        WHERE threshold IS NULL OR typeof(threshold) NOT IN ('integer','real')
            OR threshold < 0 OR threshold > 1
    ''').fetchone()[0]
    if 'results' in tables:
        errors['result_threshold_orphan'] += db.execute('''
            SELECT COUNT(*) FROM result_thresholds AS t WHERE NOT EXISTS (
                SELECT 1 FROM results AS r WHERE r.scope=t.scope AND r.scene=t.scene
                    AND r.model=t.model AND r.kind=t.kind)
        ''').fetchone()[0]
    else:
        errors['result_threshold_orphan'] += count_rows(db, tables, 'result_thresholds')


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
        if table not in tables or column not in column_names(db, table):
            continue
        for blob, in db.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"):
            checked += 1
            _tick(progress)
            try:
                payload = _decoded_payload(blob, expected)
                if table == "checkpoint_frames" and not valid_frames(payload, checkpoint=True):
                    raise ValueError("Invalid checkpoint frame evidence.")
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
                errors["state_payload_invalid"] += 1
    return checked


def _check_checkpoints(db, tables, errors, progress=None):
    if "checkpoints" in tables:
        errors["checkpoint_dimensions"] += db.execute("""
            SELECT COUNT(*) FROM checkpoints
            WHERE processed <= 0 OR total <= 0 OR processed > total
               OR top_rows < 0 OR top_rows > processed
               OR typeof(processed) != 'integer' OR typeof(total) != 'integer'
               OR typeof(top_rows) != 'integer'
               OR signature = ''
        """).fetchone()[0]

        for top_rows, blob in db.execute("SELECT top_rows,top_scores FROM checkpoints"):
            _tick(progress)
            try:
                raw = zlib.decompress(blob)
                valid = type(top_rows) is int and (
                    (top_rows == 0 and len(raw) == 0)
                    or (top_rows > 0 and len(raw) > 0 and len(raw) % (top_rows * 4) == 0)
                )
                # The shape alone accepts NaN/Inf and out-of-range scores. Match
                # inference's probability contract without importing NumPy.
                valid = valid and all(0 <= score <= 1 for score, in struct.iter_unpack("<f", raw))
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
            errors["checkpoint_orphan_frames"] += count_rows(db, tables, "checkpoint_frames")

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
            errors["checkpoint_orphan_analysis"] += count_rows(db, tables, "checkpoint_analysis")


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
            warnings["recovery_publication_without_snapshot"] += count_rows(
                db, tables, "recovery_publications")


def check_database(path, scope, *, show_progress=False):
    """Deep read-only consistency check across every scope in the state DB."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")

    from stash_ai.state import APPLICATION_ID

    errors, warnings = Counter(), Counter()
    with connect_readonly(path) as db:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if application_id != APPLICATION_ID or version not in KNOWN_VERSIONS:
            raise ValueError("Not a supported stash-taggers state DB.")

        # Keep all relation/decode checks on one read snapshot even if another
        # process commits through WAL while this command is running.
        db.execute("BEGIN")
        tables = table_names(db)
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
            _tick(progress)

            if progress is not None:
                progress.set_postfix_str("results/index", refresh=True)
            results_checked = _check_results_and_index(db, tables, errors, progress)
            _check_thresholds(db, tables, errors)
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
            pending_publications = count_rows(db, tables, "snapshots", None, "pending IS NOT NULL")
            pending_cleanups = count_rows(db, tables, "cleanup_state", None, "pending IS NOT NULL")
            refresh_requests = count_rows(db, tables, "refresh_requests")
            checkpoints = count_rows(db, tables, "checkpoints")
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
