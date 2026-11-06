"""Disposable SQLite replay with independent disk and resident page budgets."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile

from .memory_size import Reservation
from .resource_errors import ResourceLimit


class Connection(sqlite3.Connection):
    def __init__(self,*args,**kwargs):
        import weakref
        super().__init__(*args,**kwargs)
        self.cursors=weakref.WeakSet()
    def _call(self,method,*args,**kwargs):
        try:
            cursor=method(*args,**kwargs);self.cursors.add(cursor);return cursor
        except sqlite3.OperationalError as error:
            if getattr(error,'sqlite_errorcode',None)==sqlite3.SQLITE_FULL:
                raise ResourceLimit('Replay disk image exceeds execution.replay_snapshot_bytes.') from error
            raise
    def execute(self,*args,**kwargs):return self._call(super().execute,*args,**kwargs)
    def executemany(self,*args,**kwargs):return self._call(super().executemany,*args,**kwargs)
    def executescript(self,*args,**kwargs):return self._call(super().executescript,*args,**kwargs)
    def close(self):
        # Tracebacks may retain partially iterated cursors. Finalize them before
        # closing so a failed worker cannot keep the live vector source locked.
        for cursor in list(self.cursors):cursor.close()
        self.cursors.clear();super().close()


@contextmanager
def replay(config):
    from .run_cache import current
    cache=current()
    limit=min(config.data['execution']['rebuild_memory_bytes'],cache.working if cache and cache.working is not None else config.data['execution']['rebuild_memory_bytes'])
    pages=min(2*1024**2,max(32768,limit//16))
    reservation=Reservation(limit,label='replay-page-cache')
    reservation.set(pages*2)
    try:
        with tempfile.TemporaryDirectory(prefix='recognition-replay-') as directory:
            uri=(Path(directory)/'replay.sqlite3').as_uri()
            db=sqlite3.connect(uri,uri=True,factory=Connection)
            try:
                db.row_factory=sqlite3.Row
                db.execute('PRAGMA cache_size=-'+str(max(32,pages//1024)))
                db.execute('PRAGMA temp_store=FILE')
                db.execute('PRAGMA mmap_size=0')
                size=db.execute('PRAGMA page_size').fetchone()[0]
                db.execute('PRAGMA max_page_count='+str(config.data['execution']['replay_snapshot_bytes']//size))
                yield db,uri
            finally:
                from stash_ai.runtime.cleanup import cleanup
                cleanup([('close disk replay',db.close)])
    finally:reservation.set(0)
