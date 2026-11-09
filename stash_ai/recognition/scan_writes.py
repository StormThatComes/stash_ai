"""One outstanding complete-cue transaction per scan; only the owner reports progress."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3

from .database import encoded


def write(db, batch, collector):
    from .pixel_storage import compact
    from .cold_payloads import available, put
    scan, cursor, pending, supplemental, shots, state = batch
    with db:
        cold = available(db)
        def records():
            for item in pending:
                row = list(item)
                row[7] = compact(db, row[7])
                row[11] = put(db, row[11], supported=cold)
                yield row
        db.executemany('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', records())
        if supplemental:
            collector.write_records(db, supplemental)
        db.execute('UPDATE scans SET cursor=? WHERE id=?', (cursor, scan))
        if shots:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('cue-view-shots:'+scan, encoded(shots)))
        if state and state[0]+1 == cursor:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('partial-view-shot:'+scan, encoded(state)))
    return cursor


def write_file(path, batch, collector):
    with closing(sqlite3.connect(path, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        return write(db, batch, collector)


class Writer:
    def __init__(self, database, collector, progress, asynchronous):
        self.db = database.connection
        self.path = getattr(database, 'path', None)
        self.collector = collector
        self.progress = progress
        self.executor = None
        self.future = None
        self.failed = False
        if asynchronous and self.path and not self.db.in_transaction and getattr(database, 'lock', None):
            self.executor = ThreadPoolExecutor(1, thread_name_prefix='recognition-checkpoint')

    def drain(self):
        if self.future is not None:
            future, self.future = self.future, None
            try:
                cursor = future.result()
            except BaseException:
                self.failed = True
                raise
            self.progress(cursor)

    def submit(self, batch):
        self.drain()
        if self.executor is None:
            self.progress(write(self.db, batch, self.collector))
        else:
            self.future = self.executor.submit(write_file, self.path, batch, self.collector)

    def close(self):
        try:
            self.drain()
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
