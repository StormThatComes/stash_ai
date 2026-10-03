"""Content-free developer diagnostics for the persistent state database.

The exported JSON intentionally contains no scene/scope identifiers, URLs, file
paths, tag names, scores, marker timestamps, inference signatures or hashes.
Only schema/runtime facts, aggregate counts and size/shape distributions leave
the database. The report is written locally and is never uploaded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import platform
from pathlib import Path
import sqlite3
import struct
import sys
import zlib

from stash_ai.db.common import (
    BLOB_COLUMNS, KNOWN_KINDS, KNOWN_MODELS, KNOWN_VERSIONS,
    STATE_TABLES as KNOWN_TABLES, column_names, connect_readonly, table_names,
)

REPORT_FORMAT = 1


class Distribution:
    """Bounded-memory integer distribution with deterministic approximate percentiles."""

    # Small cardinalities stay exact. Larger values use a fixed logarithmic
    # histogram, preventing a diagnostics report from retaining one dictionary
    # entry for every distinct BLOB size in a very large database.
    EXACT_MAX = 255
    LOG_BUCKETS = 32
    MAX_EXPONENT = 62  # SQLite INTEGER / practical Python container sizes are <= 2**63-1.
    MAX_BUCKETS = (EXACT_MAX + 1) + (MAX_EXPONENT - EXACT_MAX.bit_length() + 1) * LOG_BUCKETS

    def __init__(self):
        # key -> [count, minimum observed value, maximum observed value].
        # The key space is bounded by MAX_BUCKETS regardless of sample count.
        self.buckets = {}
        self.count = 0
        self.total = 0
        self.minimum = None
        self.maximum = None

    @classmethod
    def _bucket(cls, value):
        if value <= cls.EXACT_MAX:
            return (0, value)
        exponent = min(value.bit_length() - 1, cls.MAX_EXPONENT)
        base = 1 << exponent
        width = max(1, base // cls.LOG_BUCKETS)
        slot = min(cls.LOG_BUCKETS - 1, max(0, (value - base) // width))
        return (1, exponent, slot)

    def add(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return
        key = self._bucket(value)
        bucket = self.buckets.get(key)
        if bucket is None:
            self.buckets[key] = [1, value, value]
        else:
            bucket[0] += 1
            bucket[1] = min(bucket[1], value)
            bucket[2] = max(bucket[2], value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def _percentile(self, numerator):
        if not self.count:
            return 0
        # Nearest-rank over a deterministic bounded histogram. Exact buckets
        # return the original value; larger buckets return the midpoint of the
        # observed range within that bucket.
        rank = max(1, (self.count * numerator + 99) // 100)
        seen = 0
        for key in sorted(self.buckets):
            occurrences, low, high = self.buckets[key]
            seen += occurrences
            if seen >= rank:
                return low if low == high else (low + high) // 2
        return self.maximum or 0

    def summary(self, *, total=False):
        result = {
            "count": self.count,
            "min": self.minimum or 0,
            "p50": self._percentile(50),
            "p95": self._percentile(95),
            "p99": self._percentile(99),
            "max": self.maximum or 0,
            "mean": round(self.total / self.count, 2) if self.count else 0,
        }
        if total:
            result["total"] = self.total
        return result


def _safe_model(value):
    return value if value in KNOWN_MODELS else "other"


def _safe_kind(value):
    return value if value in KNOWN_KINDS else "other"


def _row_count(db, table):
    return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _blob_table_report(db, tables):
    report = {}
    for table, configured in BLOB_COLUMNS.items():
        if table not in tables:
            continue
        available = column_names(db, table)
        columns = [column for column in configured if column in available]
        item = {"rows": _row_count(db, table), "blob_columns": {}}
        for column in columns:
            sizes = Distribution()
            nonnull = 0
            for length, in db.execute(f"SELECT length({column}) FROM {table}"):
                if length is not None:
                    nonnull += 1
                    sizes.add(int(length))
            item["blob_columns"][column] = {
                "nonnull": nonnull,
                "bytes": sizes.summary(total=True),
            }
        report[table] = item
    return report


def _decode_result_blob(blob):
    raw = zlib.decompress(blob)
    return raw, json.loads(raw)


def _group_bucket(groups, model, kind):
    key = (_safe_model(model), _safe_kind(kind))
    if key not in groups:
        groups[key] = {
            "rows": 0,
            "decode_errors": 0,
            "compressed_bytes": Distribution(),
            "json_bytes": Distribution(),
            "compression_pct": Distribution(),
            "segments_per_row": Distribution(),
            "items_per_segment": Distribution(),
            "frame_detections": Distribution(),
            "scene_candidates": Distribution(),
            "scene_observation_frames": Distribution(),
            "scene_observations": Distribution(),
            "span_items": Distribution(),
            "segment_modes": Counter(),
            "origins": Counter(),
            "scene_frame_format": Counter(),
            "analysis_format": Counter(),
        }
    return groups[key]


def _safe_counter(counter, allowed=None):
    result = {}
    for key, count in counter.items():
        name = str(key)
        if allowed is not None and key not in allowed:
            name = "other"
        result[name] = result.get(name, 0) + count
    return dict(sorted(result.items()))


def _result_evidence_report(db, tables):
    if "results" not in tables:
        return []

    groups = {}
    for model, kind, blob in db.execute("SELECT model,kind,data FROM results"):
        bucket = _group_bucket(groups, model, kind)
        bucket["rows"] += 1
        bucket["compressed_bytes"].add(len(blob))
        try:
            raw, segments = _decode_result_blob(blob)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
            bucket["decode_errors"] += 1
            continue

        bucket["json_bytes"].add(len(raw))
        if raw:
            bucket["compression_pct"].add(round(len(blob) * 100 / len(raw)))
        if not isinstance(segments, list):
            bucket["decode_errors"] += 1
            continue

        bucket["segments_per_row"].add(len(segments))
        for segment in segments:
            if not isinstance(segment, dict):
                bucket["decode_errors"] += 1
                continue
            mode = segment.get("mode")
            safe_mode = mode if mode in ("scene", "frames", "spans") else "other"
            bucket["segment_modes"][safe_mode] += 1

            meta = segment.get("meta")
            if isinstance(meta, dict):
                origin = meta.get("origin")
                bucket["origins"][origin if origin in ("inference", "legacy") else "other"] += 1
                scene_format = meta.get("scene_frames_format")
                if isinstance(scene_format, int):
                    bucket["scene_frame_format"]["1" if scene_format == 1 else "other"] += 1
                analysis_format = meta.get("analysis_format")
                if isinstance(analysis_format, int):
                    bucket["analysis_format"][str(analysis_format) if analysis_format in (1, 2, 3) else "other"] += 1

            items = segment.get("items")
            if not isinstance(items, list):
                bucket["decode_errors"] += 1
                continue
            bucket["items_per_segment"].add(len(items))

            if safe_mode == "frames":
                for frame in items:
                    tags = frame.get("tags") if isinstance(frame, dict) else None
                    bucket["frame_detections"].add(len(tags) if isinstance(tags, list) else 0)
            elif safe_mode == "scene":
                bucket["scene_candidates"].add(len(items))
                frames = segment.get("frames")
                if isinstance(frames, list):
                    bucket["scene_observation_frames"].add(len(frames))
                    for frame in frames:
                        tags = frame.get("tags") if isinstance(frame, dict) else None
                        bucket["scene_observations"].add(len(tags) if isinstance(tags, list) else 0)
            elif safe_mode == "spans":
                bucket["span_items"].add(len(items))

    result = []
    for (model, kind), bucket in sorted(groups.items()):
        result.append({
            "model": model,
            "kind": kind,
            "rows": bucket["rows"],
            "decode_errors": bucket["decode_errors"],
            "compressed_bytes": bucket["compressed_bytes"].summary(total=True),
            "json_bytes": bucket["json_bytes"].summary(total=True),
            "compressed_to_json_pct": bucket["compression_pct"].summary(),
            "segments_per_row": bucket["segments_per_row"].summary(),
            "items_per_segment": bucket["items_per_segment"].summary(),
            "frame_detections_per_frame": bucket["frame_detections"].summary(),
            "scene_candidates_per_segment": bucket["scene_candidates"].summary(),
            "scene_observation_frames_per_segment": bucket["scene_observation_frames"].summary(),
            "scene_observations_per_frame": bucket["scene_observations"].summary(),
            "span_items_per_segment": bucket["span_items"].summary(),
            "segment_modes": _safe_counter(bucket["segment_modes"], {"scene", "frames", "spans", "other"}),
            "origins": _safe_counter(bucket["origins"], {"inference", "legacy", "other"}),
            "scene_frame_formats": _safe_counter(bucket["scene_frame_format"]),
            "analysis_formats": _safe_counter(bucket["analysis_format"]),
        })
    return result


def _checkpoint_report(db, tables):
    groups = defaultdict(lambda: {
        "rows": 0,
        "processed": Distribution(),
        "total_frames": Distribution(),
        "progress_permille": Distribution(),
        "top_rows": Distribution(),
        "top_score_blob_bytes": Distribution(),
    })
    if "checkpoints" in tables:
        for model, processed, total, top_rows, size in db.execute(
                "SELECT model,processed,total,top_rows,length(top_scores) FROM checkpoints"):
            bucket = groups[_safe_model(model)]
            bucket["rows"] += 1
            bucket["processed"].add(int(processed))
            bucket["total_frames"].add(int(total))
            bucket["top_rows"].add(int(top_rows))
            bucket["top_score_blob_bytes"].add(int(size or 0))
            if total:
                bucket["progress_permille"].add(round(processed * 1000 / total))

    chunks = defaultdict(Distribution)
    if "checkpoint_frames" in tables:
        for model, size in db.execute("SELECT model,length(data) FROM checkpoint_frames"):
            chunks[_safe_model(model)].add(int(size or 0))

    analysis = defaultdict(Distribution)
    if "checkpoint_analysis" in tables:
        for model, size in db.execute("SELECT model,length(data) FROM checkpoint_analysis"):
            analysis[_safe_model(model)].add(int(size or 0))

    models = sorted(set(groups) | set(chunks) | set(analysis))
    return [{
        "model": model,
        "rows": groups[model]["rows"],
        "processed_frames": groups[model]["processed"].summary(),
        "total_frames": groups[model]["total_frames"].summary(),
        "progress_permille": groups[model]["progress_permille"].summary(),
        "top_rows": groups[model]["top_rows"].summary(),
        "top_score_blob_bytes": groups[model]["top_score_blob_bytes"].summary(total=True),
        "frame_chunk_blob_bytes": chunks[model].summary(total=True),
        "analysis_blob_bytes": analysis[model].summary(total=True),
    } for model in models]


def _grouped_counts(db, tables):
    result = {}

    if "result_index" in tables:
        rows = Counter()
        inference = Counter()
        populated = Counter()
        for model, kind, inf, pop, count in db.execute(
                """SELECT model,kind,inference,populated,COUNT(*)
                   FROM result_index GROUP BY model,kind,inference,populated"""):
            key = (_safe_model(model), _safe_kind(kind))
            rows[key] += count
            inference[key] += count * int(bool(inf))
            populated[key] += count * int(bool(pop))
        result["result_index"] = [
            {"model": model, "kind": kind, "rows": rows[model, kind],
             "inference_rows": inference[model, kind], "populated_rows": populated[model, kind]}
            for model, kind in sorted(rows)
        ]

    if "refresh_requests" in tables:
        counts = Counter()
        for model, kind, count in db.execute(
                "SELECT model,kind,COUNT(*) FROM refresh_requests GROUP BY model,kind"):
            counts[_safe_model(model), _safe_kind(kind)] += count
        result["refresh_requests"] = [
            {"model": model, "kind": kind, "rows": count}
            for (model, kind), count in sorted(counts.items())
        ]

    for table in ("snapshots", "legacy_imports", "recovery_publications"):
        if table not in tables:
            continue
        counts = Counter()
        for kind, count in db.execute(f"SELECT kind,COUNT(*) FROM {table} GROUP BY kind"):
            counts[_safe_kind(kind)] += count
        result[table] = [{"kind": kind, "rows": count} for kind, count in sorted(counts.items())]

    return result


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
        quick_check = db.execute("PRAGMA quick_check(1)").fetchone()[0]
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        page_count = db.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = db.execute("PRAGMA freelist_count").fetchone()[0]

        scopes = (db.execute("SELECT COUNT(DISTINCT scope) FROM results").fetchone()[0]
                  if "results" in tables else 0)
        scenes = (db.execute(
            "SELECT COUNT(*) FROM (SELECT scope,scene FROM results GROUP BY scope,scene)"
        ).fetchone()[0] if "results" in tables else 0)

        known_present = sorted(tables & KNOWN_TABLES)
        table_rows = {table: _row_count(db, table) for table in known_present}
        index_count = db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'"
        ).fetchone()[0]

        database = {
            "schema_version": schema_version,
            "quick_check_ok": quick_check == "ok",
            "file_bytes": path.stat().st_size,
            "wal_bytes": Path(str(path) + "-wal").stat().st_size if Path(str(path) + "-wal").exists() else 0,
            "shm_bytes": Path(str(path) + "-shm").stat().st_size if Path(str(path) + "-shm").exists() else 0,
            "page_size": page_size,
            "page_count": page_count,
            "freelist_count": freelist_count,
            "free_page_bytes": page_size * freelist_count,
            "scope_count": scopes,
            "scene_count": scenes,
            "known_tables_present": known_present,
            "unknown_table_count": len(tables - KNOWN_TABLES),
            "non_auto_index_count": index_count,
            "table_rows": table_rows,
            "blob_tables": _blob_table_report(db, tables),
        }

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
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
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
