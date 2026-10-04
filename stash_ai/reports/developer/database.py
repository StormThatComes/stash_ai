"""Aggregate database, checkpoint and state table statistics."""
from collections import Counter, defaultdict
from pathlib import Path

from stash_ai.db.common import BLOB_COLUMNS, STATE_TABLES as KNOWN_TABLES, column_names
from .distribution import Distribution
from .common import _row_count, _safe_model, _safe_kind


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


def database_report(path, db, tables, schema_version):
    """Describe file/page usage without exposing database content."""
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

    return database
