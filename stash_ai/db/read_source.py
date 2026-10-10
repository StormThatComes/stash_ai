"""One guardian connection validates the entire report's source interval."""
from contextlib import closing, contextmanager
from pathlib import Path

from .common import connect_readonly


@contextmanager
def consistent_source(path):
    path = Path(path).expanduser().resolve()
    identity = path.stat()
    with closing(connect_readonly(path)) as guardian:
        version = guardian.execute('PRAGMA data_version').fetchone()[0]
        yield
        current = path.stat()
        if ((identity.st_dev, identity.st_ino) != (current.st_dev, current.st_ino)
                or guardian.execute('PRAGMA data_version').fetchone()[0] != version):
            raise RuntimeError('The source state DB changed during offline report; repeat the report.')
