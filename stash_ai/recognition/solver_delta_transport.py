"""Transport dense prediction deltas through the invocation-owned solver directory."""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain, islice
import json
import os
from pathlib import Path
import tempfile

from .resource_errors import ResourceLimit

_target = ContextVar('solver_delta_target', default=None)
INLINE_ROWS = 4096


@contextmanager
def worker_target(directory, limit):
    token = _target.set((Path(directory), limit))
    try:
        yield
    finally:
        _target.reset(token)


@dataclass(frozen=True)
class Delta:
    filename: str
    checksum: str
    rows: int
    journal: bool = False

    @classmethod
    def from_archive(cls, trial, metadata):
        delta = cls(**metadata, journal=True)
        if (len(delta.checksum) != 64 or any(c not in '0123456789abcdef' for c in delta.checksum)
                or delta.filename != f'{trial}-delta-{delta.checksum}.jsonl'
                or Path(delta.filename).name != delta.filename or type(delta.rows) is not int or delta.rows < 0):
            raise ValueError('Invalid checkpoint prediction delta.')
        return delta

    def archive(self, directory, destination, trial):
        """Publish a verified sidecar before publishing a recoverable raw result."""
        from .disk_budget import check
        from tempfile import NamedTemporaryFile
        filename = f'{trial}-delta-{self.checksum}.jsonl'
        if Path(filename).name != filename or Path(self.filename).name != self.filename:
            raise ValueError('Invalid solver delta filename.')
        checksum = sha256()
        source = directory / self.filename
        check(source.stat().st_size, destination)
        with NamedTemporaryFile(dir=destination, delete=False) as target:
            temporary = Path(target.name)
        try:
            with source.open('rb') as reader, temporary.open('wb') as writer:
                while chunk := reader.read(1024**2):
                    checksum.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if checksum.hexdigest() != self.checksum:
                raise ValueError('Corrupt solver prediction delta.')
            os.replace(temporary, destination / filename)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(filename=filename, checksum=self.checksum, rows=self.rows)

    def read(self, directory, *, indexed=False):
        if Path(self.filename).name != self.filename:
            raise ValueError('Invalid solver delta filename.')
        checksum = sha256()
        def rows():
            with (directory / self.filename).open('rb') as stream:
                for line in stream:
                    checksum.update(line)
                    yield json.loads(line)
        changes = DiskChanges(rows()) if indexed and self.rows>INLINE_ROWS else dict(rows())
        if len(changes) != self.rows or checksum.hexdigest() != self.checksum:
            if isinstance(changes,DiskChanges):
                changes.close()
            raise ValueError('Incomplete or corrupt solver prediction delta.')
        return changes


class DiskChanges(Mapping):
    def __init__(self, rows):
        from .scratch_space import connect
        self.db = connect()
        try:
            self.db.execute('PRAGMA cache_size=-512')
            self.db.execute('CREATE TABLE changes(id TEXT PRIMARY KEY, person TEXT) WITHOUT ROWID')
            self.db.executemany('INSERT INTO changes VALUES (?,?)',rows)
            self.db.execute('CREATE INDEX changes_owner ON changes(person,id)')
            self.count = self.db.execute('SELECT COUNT(*) FROM changes').fetchone()[0]
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.count

    def __iter__(self):
        return (row[0] for row in self.db.execute('SELECT id FROM changes'))

    def items(self):
        return self.db.execute('SELECT id,person FROM changes')

    def __getitem__(self, key):
        row = self.db.execute('SELECT person FROM changes WHERE id=?',(key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def for_owners(self, owners):
        for owner in owners:
            yield from self.db.execute('SELECT id,person FROM changes WHERE person IS ?', (owner,))

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def __del__(self):
        if getattr(self,'db',None) is not None:
            self.close()


def collect(rows):
    target = _target.get()
    if target is None:
        return dict(rows)
    from .disk_budget import reserve_file
    from contextlib import ExitStack
    rows = iter(rows)
    first = list(islice(rows, INLINE_ROWS + 1))
    if len(first) <= INLINE_ROWS:
        return dict(first)
    directory, limit = target
    descriptor, name = tempfile.mkstemp(prefix='delta-', dir=directory)
    path = Path(name)
    checksum = sha256()
    written = count = 0
    try:
        with os.fdopen(descriptor, 'wb') as stream, ExitStack() as leases:
            grant = 0
            entries = chain(first, rows)
            while group := list(islice(entries, 4096)):
                data = b''.join((json.dumps(tuple(row), separators=(',', ':'))+'\n').encode() for row in group)
                needed = written + len(data)
                if needed > limit:
                    raise ResourceLimit('Solver delta exceeds its configured scratch budget.')
                if needed > grant:
                    stream.flush()
                    leases.close()
                    grant = leases.enter_context(reserve_file(path,min(limit,needed+4*1024**2)))
                    if grant < needed:
                        raise ResourceLimit('Insufficient space to transport solver prediction delta.')
                stream.write(data)
                stream.flush()
                checksum.update(data)
                written = needed
                count += len(group)
        return Delta(path.name, checksum.hexdigest(), count)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
