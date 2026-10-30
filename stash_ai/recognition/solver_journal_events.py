"""Checksummed append-only transitions with occasional atomic JSON snapshots."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path

_journals = ContextVar('recognition_solver_journals', default=None)
ZERO = '0'*64


def apply(document, event):
    kind = event['kind']
    if kind == 'schedule':
        document.setdefault('schedule', []).append(event['entry'])
    elif kind in ('task', 'raw'):
        document.setdefault('tasks' if kind == 'task' else 'raw_results', {})[event['name']] = event['entry']
    elif kind == 'record':
        document['records'].append(event['entry'])
        if event.get('task') is not None:
            document.setdefault('tasks', {})[event['name']] = event['task']
        if event.get('winner') is not None:
            document['winner'] = event['winner']
        document.get('raw_results', {}).pop(event['name'], None)
    else:
        raise ValueError('Invalid solver journal event.')


def log_path(path):
    path = Path(path)
    return path.with_name(path.name+'.trials')/'events.jsonl'


def read(path):
    document = json.loads(Path(path).read_bytes())
    sequence, previous, valid_bytes = 0, ZERO, 0
    if document.get('journal_format') != 1:
        return document, sequence, previous, valid_bytes
    applied = document.get('journal_applied', 0)
    with log_path(path).open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                break  # A process may die halfway through the final append.
            try:
                row = json.loads(raw)
                content = json.dumps(row['event'], sort_keys=True, separators=(',', ':'), allow_nan=False)
                digest = hashlib.sha256((previous+content).encode()).hexdigest()
                if row['sequence'] != sequence+1 or row['sha256'] != digest:
                    raise ValueError('Corrupt solver journal event chain.')
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError('Corrupt solver journal event.') from error
            sequence += 1; previous = digest; valid_bytes += len(raw)
            if sequence > applied:
                apply(document, row['event'])
    if applied > sequence:
        raise ValueError('Solver journal is missing committed events.')
    document['journal_applied'] = sequence
    return document, sequence, previous, valid_bytes


def load_document(path):
    """Read current durable state, including events since the JSON snapshot."""
    return read(path)[0]


class Events:
    def __init__(self, journal, *, resume=False, state=None):
        import weakref
        self.journal = weakref.proxy(journal)
        self.path = log_path(journal.path)
        if state is None:
            from .merge_solver_checkpoint import AtomicFile
            AtomicFile(self.path).write('')
            self.sequence, self.previous, self.size = 0, ZERO, 0
        else:
            self.sequence, self.previous, self.size = state
            if self.path.stat().st_size != self.size:
                with self.path.open('r+b') as stream:
                    stream.truncate(self.size); stream.flush(); os.fsync(stream.fileno())
        journal.document.setdefault('journal_format', 1)
        journal.document.setdefault('journal_applied', 0)

    def append(self, event):
        self.journal.file.check()
        content = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256((self.previous+content).encode()).hexdigest()
        raw = (json.dumps(dict(sequence=self.sequence+1, sha256=digest, event=event),
                          separators=(',', ':'), allow_nan=False)+'\n').encode()
        with self.path.open('ab') as stream:
            if os.fstat(stream.fileno()).st_size != self.size:
                raise FileExistsError('Solver journal changed outside this solver.')
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.sequence += 1; self.previous = digest; self.size += len(raw)


@contextmanager
def journal_scope():
    journals = []
    token = _journals.set(journals)
    try:
        yield
    finally:
        try:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('close solver journal', journal.close) for journal in journals])
        finally:
            _journals.reset(token)
