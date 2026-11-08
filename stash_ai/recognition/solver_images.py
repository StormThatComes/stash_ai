"""Private trial images derived from one immutable solver seed."""
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3
import tempfile

_active = ContextVar('solver_images', default=None)


@dataclass(frozen=True)
class Images:
    directory: Path
    seed: Path
    limit: int
    writer: bool

    def restore(self, name, connection):
        from .replay_input_cache import raw
        dest = raw(connection)
        path = self.directory / name
        if not path.is_file():
            return False
        size = path.stat().st_size
        if size > self.limit or size > (getattr(dest, 'page_limit', None) or self.limit):
            return False
        if getattr(dest, 'budget', None) is not None:
            if hasattr(dest, 'account_pages'):
                dest.budget.reserve(dest.owner, size)
            else:
                dest.budget.set(dest, 2 * size)
                dest.capacity = dest.estimate = size
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as source:
            source.backup(dest)
        if hasattr(dest, 'account_pages'):
            dest.account_pages()
        return True

    def publish(self, name, connection):
        from .replay_input_cache import raw
        from .disk_budget import reserve_file
        from .profiling import count
        if not self.writer or (self.directory / name).exists():
            return
        source = raw(connection)
        size = source.execute('PRAGMA main.page_count').fetchone()[0] * source.execute('PRAGMA main.page_size').fetchone()[0]
        if size + sum(p.stat().st_size for p in self.directory.iterdir()) > self.limit:
            return
        temporary = self.directory / (name + '.partial')
        try:
            temporary.touch(exist_ok=False)
            with reserve_file(temporary, size) as grant:
                if size > grant:
                    return
                with closing(sqlite3.connect(temporary)) as target:
                    source.backup(target)
                os.replace(temporary, self.directory / name)
        except (OSError, sqlite3.Error):
            count('solver_image_write_failures')
        finally:
            temporary.unlink(missing_ok=True)


def current():
    return _active.get()


@contextmanager
def session(seed, limit):
    path = seed.execute('PRAGMA database_list').fetchone()[2] if seed is not None else ''
    if not path:
        yield
        return
    path = Path(path).resolve()
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='solver-inputs-') as directory:
        token = _active.set(Images(Path(directory), path, limit, True))
        try:
            yield
        finally:
            _active.reset(token)


def worker(images):
    from dataclasses import replace
    _active.set(replace(images, writer=False) if images is not None else None)
