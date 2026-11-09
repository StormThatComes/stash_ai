"""Transport dense prediction deltas through the invocation-owned solver directory."""
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

    def read(self, directory):
        if Path(self.filename).name != self.filename:
            raise ValueError('Invalid solver delta filename.')
        checksum = sha256()
        changes = {}
        with (directory / self.filename).open('rb') as stream:
            for line in stream:
                checksum.update(line)
                key, person = json.loads(line)
                changes[key] = person
        if len(changes) != self.rows or checksum.hexdigest() != self.checksum:
            raise ValueError('Incomplete or corrupt solver prediction delta.')
        return changes


def collect(rows):
    target = _target.get()
    if target is None:
        return dict(rows)
    from .disk_budget import reserve_file
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
        with os.fdopen(descriptor, 'wb') as stream:
            entries = chain(first, rows)
            while group := list(islice(entries, 4096)):
                data = b''.join((json.dumps(tuple(row), separators=(',', ':'))+'\n').encode() for row in group)
                needed = written + len(data)
                if needed > limit:
                    raise ResourceLimit('Solver delta exceeds its configured scratch budget.')
                with reserve_file(path, needed) as grant:
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
