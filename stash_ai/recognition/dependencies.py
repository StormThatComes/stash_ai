"""Transactional dependency stamps and disposable scalar digest caches."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import re
import sqlite3
import uuid

from .profiling import count


_writable = ContextVar('recognition_dependency_cache_writes', default=True)


@contextmanager
def read_only():
    """Validation may reuse cache entries; only a successful apply creates them."""
    token = _writable.set(False)
    try:
        yield
    finally:
        _writable.reset(token)


def supported(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_versions'").fetchone() is not None


def version(db, kind='global', key='*'):
    row=db.execute('SELECT version FROM evidence_versions WHERE kind=? AND key=?',(kind,key)).fetchone()
    if row is None:return 0
    if type(row[0]) is not int or row[0]<1:
        raise ValueError('Invalid recognition dependency version.')
    return row[0]


@contextmanager
def snapshot(db):
    name='evidence_snapshot_'+uuid.uuid4().hex
    db.execute('SAVEPOINT '+name)
    try:
        yield
    except BaseException:
        db.execute('ROLLBACK TO '+name)
        raise
    finally:
        db.execute('RELEASE '+name)


def cached(db, kind, key, policy, compute, *, dependency='global', dependency_key='*', checked=False):
    if not checked:
        if not supported(db):return compute()
        from .dependency_schema import validate_triggers
        validate_triggers(db,required=True)
    with nullcontext() if checked else snapshot(db):
        generation=version(db,dependency,dependency_key)
        row=db.execute('SELECT version,digest FROM evidence_fingerprints WHERE kind=? AND key=? AND policy=?',
                       (kind,key,policy)).fetchone()
        if row and row[0]==generation and isinstance(row[1],str) and re.fullmatch('[0-9a-f]{64}',row[1]):
            count('dependency_digest_hits')
            return row[1]
        count('dependency_digest_misses')
        value=compute()
        store(db,kind,key,policy,generation,value)
        return value


def store(db, kind, key, policy, generation, value):
    if not _writable.get() or db.execute('PRAGMA query_only').fetchone()[0]:
        return
    try:
        db.execute('INSERT OR REPLACE INTO evidence_fingerprints VALUES (?,?,?,?,?)',
                   (kind,key,policy,generation,value))
        db.execute('DELETE FROM evidence_fingerprints WHERE rowid IN (SELECT rowid FROM evidence_fingerprints '
                   'WHERE kind=? AND key=? ORDER BY rowid DESC LIMIT -1 OFFSET 8)',(kind,key))
        if kind=='historical-context':
            db.execute('DELETE FROM evidence_fingerprints WHERE rowid IN (SELECT rowid FROM evidence_fingerprints '
                       'WHERE kind=? ORDER BY rowid DESC LIMIT -1 OFFSET 128)',(kind,))
    except sqlite3.OperationalError as error:
        if 'readonly' not in str(error).lower():raise


def certify(db, scope, namespace, state):
    from .config import digest
    if supported(db):
        store(db,'completed',digest([scope,namespace]),state['fingerprint'],version(db),digest(state))


def certified(db, scope, namespace, state):
    from .config import digest
    if not supported(db):return False
    from .dependency_schema import validate_triggers
    validate_triggers(db,required=True)
    row=db.execute('SELECT version,digest FROM evidence_fingerprints WHERE kind=? AND key=? AND policy=?',
                   ('completed',digest([scope,namespace]),state['fingerprint'])).fetchone()
    return bool(row and tuple(row)==(version(db),digest(state)))
