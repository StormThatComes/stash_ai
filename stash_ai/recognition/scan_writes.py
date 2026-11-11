"""One outstanding complete-cue transaction per scan; only the owner reports progress."""
from .profiling import pipeline_bind, pipeline_stage, pipeline_timing
from concurrent.futures import ThreadPoolExecutor
import sqlite3

from .database import encoded


@pipeline_stage('scan_sqlite_write')
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


class Writer:
    def __init__(self, database, collector, progress, asynchronous):
        self.db = database.connection
        self.path = getattr(database, 'path', None)
        self.collector = collector
        self.progress = progress
        self.executor = None
        self.worker_db = None
        self.future = None
        self.failed = False
        if asynchronous and self.path and not self.db.in_transaction and getattr(database, 'lock', None):
            self.executor = ThreadPoolExecutor(1, thread_name_prefix='recognition-checkpoint')

    def write_batch(self, batch):
        # Created, reused and closed on the single writer thread.
        if self.worker_db is None:
            self.worker_db = sqlite3.connect(self.path, timeout=30)
            self.worker_db.row_factory = sqlite3.Row
            self.worker_db.execute('PRAGMA foreign_keys=ON')
        return write(self.worker_db, batch, self.collector)

    def close_connection(self):
        if self.worker_db is not None:
            self.worker_db.close()
            self.worker_db = None

    def drain(self):
        if self.future is not None:
            future, self.future = self.future, None
            try:
                from stash_ai.runtime.console import activity, current_report
                with activity(current_report(), 'save VTT checkpoint', wait='SQLite write'):
                    with pipeline_timing('scan_writer_wait'):
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
            self.future = self.executor.submit(pipeline_bind(self.write_batch), batch)

    def close(self):
        try:
            self.drain()
        finally:
            if self.executor:
                try:
                    self.executor.submit(self.close_connection).result()
                finally:
                    self.executor.shutdown(wait=True)
