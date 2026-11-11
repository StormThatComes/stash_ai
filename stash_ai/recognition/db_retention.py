"""Explicit offline pruning and opt-in post-success retention; never VACUUM."""
from .profiling import pipeline_stage, pipeline_timing
import json
import sqlite3
from types import SimpleNamespace

from stash_ai.db.common import format_bytes
from stash_ai.runtime.console import current_report, phase
from stash_ai.storage.writer_lock import writer_lock
from . import database as schema
from . import retention_plan as plan
from .db_maintenance import _path, _connect, _header
from .progress import stage, tracking, work

FACE_CROPS = 'recognition-pruned-face-crops:'


def _validate(db):
    if _header(db) != schema.SCHEMA_VERSION:
        raise ValueError('Run recognition with this version to upgrade the database before db-prune.')
    reader = SimpleNamespace(connection=db)
    schema.FaceDatabase._validate(reader, schema.SCHEMA, schema.TABLES,
        {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")})
    schema.FaceDatabase.installation.fget(reader)
    from .dependency_schema import validate_triggers
    validate_triggers(db, required=True)
    if db.execute('PRAGMA quick_check(1)').fetchone()[0] != 'ok':
        raise ValueError('SQLite quick_check failed; retention refused.')


@pipeline_stage('retention_delete')
def _apply(db):
    if not any(db.execute(sql).fetchone() for sql in (
        'SELECT 1 FROM retention_actions LIMIT 1',
        "SELECT 1 FROM retention_scans WHERE action='delete' LIMIT 1",
        'SELECT 1 FROM retention_contracts LIMIT 1')):
        return
    statements = []
    def add(label, sql, parameters=()):
        statements.append((label, sql, parameters))

    # Preserve enough information to restore archived face crops without model
    # inference if the operator selects that exact scan policy again.
    add('Remember archived face crops', "INSERT OR IGNORE INTO metadata SELECT ?||o.scan,'true' FROM observations o "
               "JOIN retention_actions a ON a.id=o.id WHERE a.table_name='observations' AND a.action='clear_crop'",
               (FACE_CROPS,))
    add('Collect obsolete thumbnail references', "INSERT OR IGNORE INTO retention_thumbnails SELECT json_extract(o.quality,'$.evidence_thumbnail_id') "
               "FROM observations o JOIN retention_actions a ON a.id=o.id WHERE a.table_name='observations' "
               "AND a.action='delete' AND json_extract(o.quality,'$.evidence_thumbnail_id') IS NOT NULL")
    for table, address, metadata in (('person_evidence', 'observation', 'metadata'),
                                    ('body_observations', 'id', 'quality'), ('observations', 'id', None)):
        match = f'a.id={table}.{address}' + (f' AND a.contract={table}.contract' if table == 'person_evidence' else '')
        update = 'crop=NULL'
        if metadata:
            # Existing person backfill can rehydrate these pixels on reuse.
            update += f",{metadata}=json_set({metadata},'$.crop_status','retention_pruned','$.crop_attempt_limit',0)"
        add(f'Clear obsolete crops: {table}', f'UPDATE {table} SET {update} WHERE {address} IN (SELECT id FROM retention_actions '
                   f"WHERE table_name=? AND action='clear_crop') AND EXISTS (SELECT 1 FROM retention_actions a "
                   f"WHERE a.table_name=? AND a.action='clear_crop' AND {match})", (table,table))
    # Delete references before their parents. Pins include SQL and JSON roots.
    add('Delete obsolete candidates', "DELETE FROM candidates WHERE scan IN (SELECT id FROM retention_scans WHERE action='delete')")
    add('Delete obsolete body cues', "DELETE FROM body_cues WHERE scan IN (SELECT id FROM retention_scans WHERE action='delete')")
    add('Delete obsolete body contracts', "DELETE FROM body_cues WHERE (scan,contract) IN ("
        "SELECT c.scan,c.contract FROM retention_contracts c WHERE c.kind='body' "
        "AND NOT EXISTS (SELECT 1 FROM body_observations b WHERE b.scan=c.scan AND b.contract=c.contract "
        "AND NOT EXISTS (SELECT 1 FROM retention_actions a WHERE a.table_name='body_observations' "
        "AND a.id=b.id AND a.action='delete')))")
    for table, address in (('person_evidence', 'observation'), ('body_observations', 'id'), ('observations', 'id')):
        match = f'a.id={table}.{address}' + (f' AND a.contract={table}.contract' if table == 'person_evidence' else '')
        add(f'Delete obsolete evidence: {table}', f'DELETE FROM {table} WHERE {address} IN (SELECT id FROM retention_actions '
                   f"WHERE table_name=? AND action='delete') AND EXISTS (SELECT 1 FROM retention_actions a "
                   f"WHERE a.table_name=? AND a.action='delete' AND {match})", (table,table))
    add('Delete obsolete scans', "DELETE FROM scans WHERE id IN (SELECT id FROM retention_scans WHERE action='delete')")
    # Only collect thumbnails made unreferenced by this plan, including sharing
    # with other scenes/model namespaces. Never touch matching RGB signatures.
    add('Delete unreferenced thumbnails', 'DELETE FROM evidence_thumbnails WHERE id IN (SELECT id FROM retention_thumbnails) '
               "AND NOT EXISTS (SELECT 1 FROM observations o WHERE CASE WHEN json_valid(o.quality) THEN json_extract(o.quality,'$.evidence_thumbnail_id') END=+evidence_thumbnails.id)")
    for table in ('evidence_versions', 'evidence_fingerprints'):
        kinds="'scan','raw-observations','raw-person','raw-body','scene-face-vectors','hybrid-observations','hybrid-person','vtt-scan-decisions'"
        add(f'Delete obsolete bookkeeping: {table}', f"DELETE FROM {table} WHERE kind IN ({kinds}) "
                   "AND key IN (SELECT id FROM retention_scans WHERE action='delete')")
    from .rejection_summary import PREFIX
    add('Delete obsolete rejection summaries', "DELETE FROM metadata WHERE key IN "
        "(SELECT ?||id FROM retention_scans WHERE action='delete')", (PREFIX,))
    add('Delete obsolete cue shots', "DELETE FROM metadata WHERE key IN "
        "(SELECT 'cue-view-shots:'||id FROM retention_scans WHERE action='delete' "
        "UNION ALL SELECT 'future-crowd:'||id FROM retention_scans WHERE action='delete')")
    add('Delete obsolete crop markers', "DELETE FROM metadata WHERE key IN (SELECT ?||id FROM retention_scans WHERE action='delete' "
        "UNION ALL SELECT 'verified-scan-geometry:'||id FROM retention_scans WHERE action='delete')",(FACE_CROPS,))
    with tracking(statements, 'Prune obsolete recognition evidence', unit='statement') as steps:
        for label, sql, parameters in steps:
            with stage(label):
                db.execute(sql, parameters)
    from .cold_payloads import collect
    collect(db)
    from .decision_contexts import collect
    collect(db)
    with stage('Validate retained identity references'):
        if db.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('Retention would break identity references; changes rolled back.')


def on_connection(db, config, *, apply=False, scope=None, scenes=None, notify=lambda *args: None):
    """Use an existing owner. Caller owns the transaction and commit/rollback."""
    if not db.in_transaction:
        raise ValueError('Retention requires an explicit transaction.')
    notify('Plan recognition retention', 0, 1)
    try:
        with pipeline_timing('retention_plan'):
            plan.prepare(db, config, scope=scope, scenes=scenes,writable=apply)
        with stage('Summarize reclaimable evidence'):
            result = plan.summarize(db)
        result.update(applied=apply, scope=scope, scenes=scenes,
                      settings=dict(config.data['storage']['retention']))
        notify('Plan recognition retention', 1, 1)
        if apply:
            notify('Prune obsolete recognition evidence', 0, 1)
            _apply(db)
            notify('Prune obsolete recognition evidence', 1, 1)
        return result
    finally:
        with stage('Discard temporary retention plan'):
            plan.discard(db)


def prune_database(config, *, apply=False, scope=None, scenes=None, notify=lambda *args: None):
    path = _path(config.database)
    # Refuse foreign databases before creating a lock sidecar.
    with _connect(path) as check:
        _header(check)
    def execute():
        with _connect(_path(path), readonly=not apply) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            # main is still mode=ro in previews; temp planning tables are allowed.
            db.execute('PRAGMA query_only=OFF')
            db.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
            try:
                notify('Validate recognition database', 0, 1)
                _validate(db)
                # Known application-owned JSON triggers/indexes are validated
                # above; older SQLite versions require trusted_schema for writes.
                if apply:
                    db.execute('PRAGMA trusted_schema=ON')
                notify('Validate recognition database', 1, 1)
                result = on_connection(db, config, apply=apply, scope=scope, scenes=scenes, notify=notify)
                if apply:
                    db.commit()
                else:
                    db.rollback()
                return result
            except BaseException:
                db.rollback()
                raise
    if not apply:
        return execute()
    try:
        with writer_lock(path):
            return execute()
    except RuntimeError as error:
        if str(error) == 'Another merged-output process is using this state DB.':
            raise RuntimeError('Another recognition writer owns this SQLite database; retention refused.') from error
        raise


@pipeline_stage('auto_prune')
def after_run(database, config, run, *, api_key=''):
    from .diagnostics import message
    """Only a successfully finalized run may prune its own selected scenes."""
    if not config.data['storage']['retention']['auto_prune']:
        return None
    db = database.connection
    record = db.execute('SELECT scope,selection,status FROM runs WHERE id=?', (run,)).fetchone()
    if record is None or record['status'] != 'success':
        return None
    if not db.execute("SELECT 1 FROM progress WHERE run=? AND phase='rebuild'", (run,)).fetchone():
        return None  # portrait refreshes never authorize retention
    scenes = json.loads(record['selection'])
    if not scenes:
        return None
    if db.in_transaction:
        raise ValueError('Finish the recognition transaction before automatic retention.')
    with phase(current_report(), 'Recognition auto prune'), work(
            'Auto prune recognition', 3, unit='step', position=0) as progress:
        console = getattr(current_report(), 'console', None)
        if console is not None:
            with console.lock:
                console.completed = None
        def notify(label, completed, total):
            # Planning and deletion finish before commit. Never report 3/3
            # until the atomic result and retention history are durable.
            offset = 0 if label == 'Plan recognition retention' else 1
            # Phase transitions must be visible before the next blocking SQL
            # call, even when planning finished inside the UI throttle window.
            progress.set_completed(offset + completed, force=True)
        with stage('Begin automatic retention transaction'):
            db.execute('BEGIN IMMEDIATE')
        try:
            result = on_connection(db, config, apply=True, scope=record['scope'], scenes=scenes, notify=notify)
            with stage('Commit automatic retention'):
                result['outcome']='success'
                db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,?)', (run, 'retention', schema.encoded(result)))
                db.commit()
        except BaseException as error:
            with stage('Roll back automatic retention'):
                db.rollback()
            outcome='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            detail=dict(outcome=outcome,applied=False,scope=record['scope'],scenes=scenes,
                        error_type=type(error).__name__,error=message(error,api_key)[:1000])
            with db:
                db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,?)',(run,'retention',schema.encoded(detail)))
            raise
        progress.advance()
        return result


def format_report(result):
    lines = ['Recognition retention: ' + ('applied' if result['applied'] else 'PREVIEW (no changes)'),
             'Scans: ' + ', '.join(f'{k}={v}' for k, v in sorted(result['scans'].items())),
             'Retained scan reasons: ' + ', '.join(f'{k}={v}' for k, v in sorted(result['protected_reasons'].items())),
             f"Obsolete person contracts considered: {result['obsolete_person_contracts']}"]
    for row in result['actions']:
        lines.append(f"  {row['table']}: {row['action']} rows={row['rows']:,}, payload={format_bytes(row['logical_bytes'])}")
    lines.append(f"Reclaimable evidence payload: {format_bytes(result['reclaimable_payload_bytes'])} "
                 '(logical bytes; excludes indexes, bookkeeping and page fragmentation).')
    lines.append('Identity/audit references may exceed the retention count. Run db-vacuum separately to shrink the file.')
    return '\n'.join(lines)


def execute(args, config):
    from stash_ai.runtime.console import present, activity
    from stash_ai.runtime.report import RunReport
    report = RunReport(args.progress_style == 'verbose')
    with present(report, 'off' if args.json else args.progress_style, status_position=2):
        def notify(phase, completed, total):
            report.console.update(stage=phase, work=f'{completed}/{total}')
        with activity(report, 'Recognition retention'):
            result = prune_database(config, apply=args.apply, scope=args.scope, scenes=args.scene_id, notify=notify)
        report.console.finish_rendering()
        report.write(json.dumps(result, indent=2) if args.json else format_report(result))
    return 0
