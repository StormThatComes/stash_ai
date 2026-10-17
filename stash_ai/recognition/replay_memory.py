"""Disposable RAM replay and explicitly enabled, fail-fast next-scene lookahead."""
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
    """One immutable next scene on an isolated reader, without fixed scene caps.

    Once enabled, each selected scene must be prepared successfully or the
    rebuild fails. No extra per-scene RAM fraction or observation cap applies;
    each scene is prepared in full and memory allocation failures propagate.
    The coordinator never consumes prepared membership or decisions from a worker.
    """
    def __init__(self, uri, config):
        self.uri = uri
        self.enabled = config.data['execution'].get('rebuild_prefetch_scenes',False)
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
            values={}
            # This is a keyed cache, so matching-order sort keys are unnecessary.
            # ORDER BY cue,id uses observations_scan, avoiding an unbounded
            # temporary SQLite sort for large scenes.
            for row in db.execute('SELECT id,dimensions,embedding FROM observations WHERE scan=? '
                                  'AND rejected IS NULL ORDER BY cue,id',(scan,)):
                self.cancel.check()
                blob=row['embedding']
                try:
                    value=vector(row)
                except ValueError as error:
                    value=error  # Expose malformed evidence only at its original use.
                values[row['id']]=(row['dimensions'],blob,value)
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
        if not self.enabled:
            return
        if self.pending is None:
            raise RuntimeError(f'Recognition rebuild prefetch was not scheduled for scan {scan}.')
        if self.scan!=scan:
            raise RuntimeError('Replay lookahead order changed.')
        try:
            self.prepared=self.pending.result()
        except Exception as error:
            # Enabled lookahead is an explicit execution contract. Report
            # SQLite lock/IO errors, worker failures and MemoryError rather
            # than silently replacing this mode with serial vector decoding.
            raise RuntimeError(f'Recognition rebuild prefetch failed for scan {scan}: {error}') from error
        finally:
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
    if found is None:
        return MISSING  # Reference from another scene or already released.
    if found[:2]!=(row['dimensions'],row['embedding']):
        raise RuntimeError('Recognition rebuild prefetched evidence changed during matching.')
    count('prefetched_replay_vectors')
    count('validated_vectors')
    if isinstance(found[2],ValueError):
        raise found[2]
    return found[2]


def release_vectors():
    active=_active.get()
    if active is not None:
        active.prepared.clear()
