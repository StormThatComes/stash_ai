"""Offline space reporting and VACUUM, without recognition migrations/recovery."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

from stash_ai.db.common import format_bytes
from stash_ai.runtime.cleanup import closing
from stash_ai.storage.writer_lock import writer_lock
from . import database as schema

MIN_FREE_HEADROOM = 16 * 2**20


def _path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Face database does not exist: {path}')
    if path.stat().st_nlink != 1:
        raise ValueError('Recognition SQLite must not be a hard-linked file.')
    return path


def _header(db):
    version = db.execute('PRAGMA user_version').fetchone()[0]
    if (db.execute('PRAGMA application_id').fetchone()[0] != schema.APPLICATION_ID
            or version not in range(1, schema.SCHEMA_VERSION + 1)):
        raise ValueError('Not a supported facial recognition database; tagger/foreign SQLite is refused.')
    schema._reject_executable_schema(db)
    return version


@contextmanager
def _connect(path, *, readonly=True):
    # mode=rw, unlike connect(path), cannot silently create a missing database.
    mode = 'ro' if readonly else 'rw'
    with closing(sqlite3.connect(path.as_uri() + '?mode=' + mode, uri=True, timeout=5),
                 name='close recognition maintenance SQLite') as db:
        db.execute('PRAGMA trusted_schema=OFF')
        if readonly:
            db.execute('PRAGMA query_only=ON')
        yield db


def _quoted(name):
    return '"' + name.replace('"', '""') + '"'


def _page_usage(db):
    """Include overflow pages and every index; unavailable builds remain explicit."""
    try:
        return {name: dict(bytes=size, payload_bytes=payload, unused_bytes=unused)
                for name, size, payload, unused in db.execute(
                    "SELECT name,SUM(pgsize),SUM(payload),SUM(unused) FROM dbstat('main') GROUP BY name")}
    except sqlite3.OperationalError as error:
        if 'no such table: dbstat' not in str(error).lower() and 'no such module: dbstat' not in str(error).lower():
            raise
        return None


def _files(path):
    def size(suffix):
        try:
            return Path(str(path) + suffix).stat().st_size
        except FileNotFoundError:
            return 0
    result = {name: size(suffix) for name, suffix in (
        ('file_bytes', ''), ('wal_bytes', '-wal'), ('shm_bytes', '-shm'), ('journal_bytes', '-journal'))}
    result['total_disk_bytes'] = sum(result.values())
    return result


def _inspect(db, path, notify):
    """Caller owns a read transaction: all SQL numbers share one snapshot."""
    notify('Validate recognition schema', 0, 1)
    version = _header(db)
    expected, tables = ((schema.SCHEMA, schema.TABLES) if version == schema.SCHEMA_VERSION else
                        (getattr(schema, f'SCHEMA_V{version}'), getattr(schema, f'TABLES_V{version}')))
    actual = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # Reuse the read-only validator, never the constructor that migrates and
    # changes running history entries to interrupted.
    reader = SimpleNamespace(connection=db)
    schema.FaceDatabase._validate(reader, expected, tables, actual)
    schema.FaceDatabase.installation.fget(reader)
    notify('Validate recognition schema', 1, 1)
    notify('SQLite quick_check', 0, 1)
    health = db.execute('PRAGMA quick_check(1)').fetchone()[0]
    notify('SQLite quick_check', 1, 1)
    page_size = db.execute('PRAGMA page_size').fetchone()[0]
    page_count = db.execute('PRAGMA page_count').fetchone()[0]
    free_count = db.execute('PRAGMA freelist_count').fetchone()[0]
    notify('Measure table and index pages', 0, 1)
    usage = _page_usage(db)
    notify('Measure table and index pages', 1, 1)
    indices = {}
    for name, table in db.execute("SELECT name,tbl_name FROM sqlite_master WHERE type='index'"):
        indices.setdefault(table, []).append(name)
    rows = []
    for index, table in enumerate(sorted(tables)):
        notify('Count table rows', index, len(tables))
        quoted = _quoted(table)
        row = dict(name=table, rows=db.execute(f'SELECT COUNT(*) FROM {quoted}').fetchone()[0],
                   table_bytes=None, index_bytes=None, total_bytes=None, unused_bytes=None,
                   logical_value_bytes=None)
        if usage is not None:
            owned = [table, *indices.get(table, [])]
            row.update(table_bytes=usage[table]['bytes'],
                       index_bytes=sum(usage[name]['bytes'] for name in owned[1:]),
                       total_bytes=sum(usage[name]['bytes'] for name in owned),
                       unused_bytes=sum(usage[name]['unused_bytes'] for name in owned))
        else:
            # Portable fallback: actual BLOB / UTF-8 value lengths, not claimed
            # physical sizes. Numeric values are measured as their text form.
            columns = [r[1] for r in db.execute(f'PRAGMA table_info({quoted})')]
            expression = '+'.join(f'COALESCE(length(CAST({_quoted(c)} AS BLOB)),0)' for c in columns)
            row['logical_value_bytes'] = db.execute(f'SELECT COALESCE(SUM({expression}),0) FROM {quoted}').fetchone()[0]
        rows.append(row)
    notify('Count table rows', len(tables), len(tables))
    rows.sort(key=lambda row: (-(row['total_bytes'] if usage is not None else row['logical_value_bytes']), row['name']))
    states = {table: dict(db.execute(f'SELECT status,COUNT(*) FROM {_quoted(table)} GROUP BY status'))
              for table in ('runs', 'scans', 'publications', 'portraits', 'identity_merges') if table in tables}
    allocated, free = page_size * page_count, page_size * free_count
    accounted = sum(row['total_bytes'] for row in rows) if usage is not None else None
    warnings = []
    if health != 'ok':
        warnings.append('SQLite quick_check failed; VACUUM is refused.')
    if usage is None:
        warnings.append('SQLite dbstat is unavailable: physical per-table sizes are unknown. '
                        'Logical value bytes count BLOB/UTF-8 values (numbers as text), excluding indexes and overhead.')
    return dict(path=str(path), sqlite_version=sqlite3.sqlite_version, schema_version=version, quick_check=health,
                journal_mode=db.execute('PRAGMA journal_mode').fetchone()[0],
                page_size=page_size, page_count=page_count, freelist_count=free_count,
                allocated_bytes=allocated, free_page_bytes=free, reclaimable_estimate_bytes=free,
                live_page_bytes=allocated-free, dbstat_available=usage is not None, tables=rows, states=states,
                tables_total_bytes=accounted,
                non_table_bytes=allocated-free-accounted if accounted is not None else None,
                in_page_unused_bytes=sum(row['unused_bytes'] for row in rows) if usage is not None else None,
                warnings=warnings, **_files(path))


def inspect_database(path, *, notify=lambda *args: None):
    path = _path(path)
    with _connect(path) as db:
        db.execute('BEGIN')
        return _inspect(db, path, notify)


def _checkpoint(db):
    if db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal':
        return db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0
    return True


def compact_database(path, *, dry_run=False, notify=lambda *args: None):
    path = _path(path)
    # Reject unrelated files before creating the writer sidecar.
    with _connect(path) as check:
        _header(check)
    if dry_run:
        before = inspect_database(path, notify=notify)
        return before, before
    try:
        with writer_lock(path):
            # Recheck the pathname after ownership is acquired, then inspect
            # the same connection used for VACUUM. No stale pre-lock snapshot.
            _path(path)
            with _connect(path, readonly=False) as db:
                db.execute('BEGIN')
                before = _inspect(db, path, notify)
                db.rollback()
                if before['quick_check'] != 'ok':
                    raise RuntimeError('SQLite quick_check is not clean; VACUUM refused.')
                # SQLite documents up to twice the original DB size in free
                # temporary space. WAL can contain pages beyond the main file.
                required = 2 * max(before['allocated_bytes'], before['file_bytes']) + before['wal_bytes'] + MIN_FREE_HEADROOM
                available = shutil.disk_usage(path.parent).free
                if available < required:
                    raise RuntimeError(f'Insufficient free disk space for VACUUM: need about {format_bytes(required)}, '
                                       f'have {format_bytes(available)}.')
                notify('Checkpoint WAL before VACUUM', 0, 1)
                if not _checkpoint(db):
                    raise RuntimeError('WAL checkpoint is busy; close active database readers before VACUUM.')
                notify('Checkpoint WAL before VACUUM', 1, 1)
                notify('VACUUM', 0, 1)
                db.execute('VACUUM')
                notify('VACUUM', 1, 1)
                checkpointed = _checkpoint(db)
                db.execute('BEGIN')
                after = _inspect(db, path, notify)
                if (after['quick_check'] != 'ok' or before['schema_version'] != after['schema_version'] or
                        before['states'] != after['states'] or
                        {r['name']: r['rows'] for r in before['tables']} != {r['name']: r['rows'] for r in after['tables']}):
                    raise RuntimeError('Post-VACUUM validation failed: database health or logical counts changed.')
                if not checkpointed:
                    after['warnings'].append('VACUUM completed, but a reader prevented final WAL truncation.')
            after.update(_files(path))
            return before, after
    except RuntimeError as error:
        if str(error) == 'Another merged-output process is using this state DB.':
            raise RuntimeError('Another recognition writer owns this SQLite database; VACUUM refused.') from error
        raise


def format_report(report):
    lines = [f"Recognition DB: {report['path']}",
             f"Health: {report['quick_check']} | schema v{report['schema_version']} | SQLite {report['sqlite_version']}",
             f"Main file: {format_bytes(report['file_bytes'])} | WAL: {format_bytes(report['wal_bytes'])} | "
             f"SHM: {format_bytes(report['shm_bytes'])} | journal: {format_bytes(report['journal_bytes'])}",
             f"Total on disk: {format_bytes(report['total_disk_bytes'])} | logical allocated pages: {format_bytes(report['allocated_bytes'])}",
             'Tables (all scopes and model namespaces):']
    if report['dbstat_available']:
        lines.append(f"  {'Table':<26} {'Rows':>12} {'Data':>13} {'Indexes':>13} {'Total':>13} {'Unused inside':>13}")
        for row in report['tables']:
            lines.append(f"  {row['name']:<26} {row['rows']:>12,} " + ' '.join(
                f'{format_bytes(row[key]):>13}' for key in ('table_bytes', 'index_bytes', 'total_bytes', 'unused_bytes')))
        lines.append(f"  Table/index total: {format_bytes(report['tables_total_bytes'])}; "
                     f"schema/other pages: {format_bytes(report['non_table_bytes'])}")
        lines.append(f"Unused inside occupied pages: {format_bytes(report['in_page_unused_bytes'])} "
                     '(not added to estimated savings; not all is reclaimable).')
    else:
        for row in report['tables']:
            lines.append(f"  {row['name']:<26} rows={row['rows']:,} logical values={format_bytes(row['logical_value_bytes'])} | physical size unavailable")
    lines.append(f"Estimated VACUUM savings from free pages: {format_bytes(report['reclaimable_estimate_bytes'])} "
                 f"({report['freelist_count']:,} pages; conservative estimate, actual savings can differ).")
    for table, counts in report['states'].items():
        lines.append(f"{table}: " + (', '.join(f'{status}={count:,}' for status, count in sorted(counts.items())) or '(empty)'))
    lines.extend('Warning: ' + warning for warning in report['warnings'])
    return '\n'.join(lines)


def execute(args, config):
    from stash_ai.runtime.console import present, activity
    from stash_ai.runtime.report import RunReport
    report = RunReport(args.progress_style == 'verbose')
    vacuum = args.command in ('db-vacuum', 'db-compact')
    with present(report, 'off' if args.json else args.progress_style, status_position=0):
        def notify(phase, completed, total):
            report.console.update(stage=phase, work=f'{completed}/{total}')
        with activity(report, 'Recognition DB VACUUM' if vacuum else 'Recognition DB status'):
            if vacuum:
                before, after = compact_database(config.database, dry_run=args.dry_run, notify=notify)
                result = dict(dry_run=args.dry_run, before=before, after=after,
                              saved_disk_bytes=before['total_disk_bytes']-after['total_disk_bytes'],
                              saved_allocated_bytes=before['allocated_bytes']-after['allocated_bytes'])
            else:
                result = inspect_database(config.database, notify=notify)
        report.console.finish_rendering()
        if args.json:
            report.write(json.dumps(result, indent=2, allow_nan=False))
        elif not vacuum:
            report.write(format_report(result))
        else:
            report.write(format_report(before))
            if args.dry_run:
                report.write('DRY RUN: VACUUM was not executed.')
            else:
                report.write(f"VACUUM complete: total on disk {format_bytes(before['total_disk_bytes'])} -> "
                             f"{format_bytes(after['total_disk_bytes'])}; saved {format_bytes(result['saved_disk_bytes'])}. "
                             f"Health: {after['quick_check']}.")
                for warning in after['warnings']:
                    report.write('Warning: ' + warning)
    return 0 if (after if vacuum else result)['quick_check'] == 'ok' else 1
