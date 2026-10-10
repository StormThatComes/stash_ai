"""Conservative reconciliation of explicitly selected, provably orphaned runs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

from stash_ai.history.store import HistoryStore, history_path
from stash_ai.storage.store import scope_for


def _process_start(pid):
    """Linux /proc start tick disambiguates PID reuse; unavailable elsewhere."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19]  # field 22 (starttime) after pid/comm
    except (OSError, IndexError, ValueError):
        return None


def local_owner():
    pid = os.getpid()
    return dict(host=socket.gethostname(), pid=pid, start=_process_start(pid))


def owner_is_dead(owner):
    """None means unknown: do not mutate rows without a proved owner."""
    if not isinstance(owner, dict) or owner.get('host') != socket.gethostname():
        return None
    pid = owner.get('pid')
    if type(pid) is not int or pid <= 0:
        return None
    try:
        os.kill(pid, 0)  # existence check ONLY; never send a terminating signal
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return None
    expected = owner.get('start')
    current = _process_start(pid)
    if expected is not None and current is not None and expected != current:
        return True  # PID reused; original owner cannot still be running
    return False if expected is None or current is not None else None


def reconcile(store, scope, identifiers, *, dry_run=True):
    unique = tuple(dict.fromkeys(str(item) for item in identifiers))
    if not unique:
        raise ValueError('Select exact run IDs before reconciliation.')
    db = store.connection
    if db.in_transaction or store.pending:
        raise RuntimeError('Run-history reconciliation requires an idle connection.')
    db.execute('BEGIN' if dry_run else 'BEGIN IMMEDIATE')
    try:
        rows = []
        for identifier in unique:
            record = db.execute('SELECT scope,status,finished_at,configuration FROM runs WHERE id=?',
                                (identifier,)).fetchone()
            if record is None or record[0] != scope:
                rows.append(dict(id=identifier, action='not_found'))
                continue
            if record[1] != 'running' or record[2] is not None:
                rows.append(dict(id=identifier, action='already_finished'))
                continue
            try:
                configuration = json.loads(record[3])
            except (TypeError, ValueError):
                configuration = {}
            dead = owner_is_dead(configuration.get('process_owner'))
            action = 'orphaned' if dead else 'active' if dead is False else 'unverified'
            rows.append(dict(id=identifier, action=action))
            if dead and not dry_run:
                now = datetime.now(timezone.utc).isoformat()
                db.execute('''UPDATE runs SET status='interrupted',finished_at=?,
                    last_progress_at=COALESCE(last_progress_at,?),
                    progress_seq=progress_seq+1
                    WHERE id=? AND scope=? AND status='running' AND finished_at IS NULL''',
                    (now, now, identifier, scope))
        if dry_run:
            db.rollback()
        else:
            db.commit()
        return dict(dry_run=dry_run, rows=rows,
                    reconciled=sum(row['action'] == 'orphaned' for row in rows))
    except BaseException:
        db.rollback()
        raise


def run(app, args):
    dry_run = bool(args.dry_run or app.DRY_RUN)
    with HistoryStore(history_path(args), readonly=dry_run) as store:
        report = reconcile(store, scope_for(app.STASH_URL), args.reconcile_run_id,
                           dry_run=dry_run)
    for row in report['rows']:
        print(f"{row['id']}: {row['action']}")
    print(f"{'Would reconcile' if dry_run else 'Reconciled'} "
          f"{report['reconciled']} verified orphaned run(s).")
    return 0
