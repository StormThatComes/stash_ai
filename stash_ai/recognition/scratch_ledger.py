"""Serialize scratch capacity admission across spawned trials in one invocation."""
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from .resource_errors import ResourceLimit


def reserve(directory, amount, limit):
    root = os.environ.get('RECOGNITION_WORK_ROOT')
    if not root:
        return
    with closing(sqlite3.connect(Path(root) / 'scratch-leases.sqlite', timeout=30)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS leases(directory TEXT PRIMARY KEY, bytes INTEGER NOT NULL)')
        owner = str(directory)
        if not amount:
            db.execute('DELETE FROM leases WHERE directory=?', (owner,))
            return
        others = db.execute('SELECT COALESCE(SUM(bytes),0) FROM leases WHERE directory<>?', (owner,)).fetchone()[0]
        if others + amount > limit:
            raise ResourceLimit(f'Shared recognition scratch budget at {root} cannot admit {others + amount} bytes.',
                                budget='execution.rebuild.scratch_bytes', limit=limit, requested=others+amount)
        db.execute('INSERT OR REPLACE INTO leases VALUES (?,?)', (owner, amount))
