"""Disposable RAM replay and optional immutable next-scene validation lookahead."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import sqlite3
import uuid

from stash_ai.runtime.cancellation import Cancellation, use
from stash_ai.runtime.cleanup import closing, cleanup
from .profiling import count

_active = ContextVar('recognition_replay_lookahead', default=None)
MISSING = object()


@contextmanager
def memory_replay(config):
    uri = 'file:recognition-replay-' + uuid.uuid4().hex + '?mode=memory&cache=shared'
    with closing(sqlite3.connect(uri, uri=True), name='close RAM replay') as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA temp_store=MEMORY')
        page_size = db.execute('PRAGMA page_size').fetchone()[0]
        pages = max(1, config.data['execution']['rebuild_memory_bytes']//2//page_size)
        db.execute(f'PRAGMA max_page_count={pages}')
        try:
            yield db, uri
        except sqlite3.OperationalError as error:
            if getattr(error,'sqlite_errorcode',None) == sqlite3.SQLITE_FULL:
                raise ValueError('RAM replay exceeds execution.rebuild_memory_bytes; increase its capacity.') from error
            raise


def report_pages(db):
    count('replay_image_bytes', db.execute('PRAGMA page_count').fetchone()[0] *
          db.execute('PRAGMA page_size').fetchone()[0])


class Lookahead:
    """One immutable next scene, <=256 rows and <=8 MiB, on an isolated reader.

    Disabled by default: RAM-resident replay often has no I/O to hide. The
    coordinator never consumes prepared membership or decisions from a worker.
    """
    def __init__(self, uri, config):
        self.uri = uri
        self.enabled = config.data['execution'].get('rebuild_prefetch_scenes',False)
        self.limit = min(8*1024*1024,config.data['execution']['rebuild_memory_bytes']//64)
        self.executor = self.pending = self.scan = None
        self.prepared = {}
        self.cancel = Cancellation()

    def __enter__(self):
        self.context = _active.set(self)
        return self

    def _prepare(self, scan):
        from .clustering import vector
        with use(self.cancel), closing(sqlite3.connect(self.uri,uri=True),name='close replay lookahead reader') as db:
            db.row_factory=sqlite3.Row
            db.execute('PRAGMA query_only=ON')
            n,size=db.execute('SELECT COUNT(*),COALESCE(SUM(LENGTH(embedding)),0) FROM observations '
                             'WHERE scan=? AND rejected IS NULL',(scan,)).fetchone()
            if n>256 or size*2+n*1024>self.limit:
                return {}
            values={}
            for row in db.execute('SELECT id,dimensions,embedding FROM observations WHERE scan=? '
                                  'AND rejected IS NULL ORDER BY cue,geometry,embedding,id',(scan,)):
                self.cancel.check()
                try:
                    value=vector(row)
                except ValueError as error:
                    value=error  # Expose malformed evidence only at its original use.
                values[row['id']]=(row['dimensions'],row['embedding'],value)
            return values

    def submit(self, scan):
        if not self.enabled or scan is None:
            return
        if self.pending is not None:
            raise RuntimeError('Replay lookahead has an unconsumed scene.')
        if self.executor is None:
            self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='recognition-replay-input')
        self.scan=scan
        self.pending=self.executor.submit(self._prepare,scan)

    def take(self, scan):
        self.prepared.clear()
        if self.pending is None:
            return
        if self.scan!=scan:
            raise RuntimeError('Replay lookahead order changed.')
        self.prepared=self.pending.result()
        self.pending=self.scan=None

    def __exit__(self,*error):
        _active.reset(self.context)
        cleanup([('cancel replay lookahead',self.cancel.cancel),
                 ('close replay lookahead worker',lambda: self.executor.shutdown(wait=True,cancel_futures=True)
                  if self.executor is not None else None),
                 ('release replay lookahead vectors',self.prepared.clear)])


def prepared_vector(row):
    active=_active.get()
    found=active.prepared.get(row['id']) if active is not None else None
    if found is None or found[:2]!=(row['dimensions'],row['embedding']):
        return MISSING
    count('prefetched_replay_vectors')
    count('validated_vectors')
    if isinstance(found[2],ValueError):
        raise found[2]
    return found[2]


def release_vectors():
    active=_active.get()
    if active is not None:
        active.prepared.clear()
