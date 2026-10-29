"""One durable boundary for face, hybrid, body and publication decisions."""
from contextlib import contextmanager
import uuid


class Connection:
    """Nested stage contexts are savepoints, never independent commits."""
    def __init__(self, raw):
        self.raw = raw
        self.contexts = []

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __enter__(self):
        name = 'rebuild_' + uuid.uuid4().hex
        self.raw.execute('SAVEPOINT ' + name)
        self.contexts.append(name)
        return self

    def __exit__(self, kind, error, traceback):
        name = self.contexts.pop()
        if kind is not None:
            self.raw.execute('ROLLBACK TO ' + name)
        self.raw.execute('RELEASE ' + name)

    def commit(self):
        # A stage may finish, but the complete rebuild owns durability.
        pass

    def rollback(self):
        raise RuntimeError('Rebuild stages must roll back their savepoint context.')

    def executescript(self, sql):
        raise RuntimeError('Implicit-commit scripts are forbidden during an identity rebuild.')


@contextmanager
def atomic(database):
    raw = database.connection
    borrowed = raw.in_transaction
    if not borrowed:
        raw.execute('BEGIN IMMEDIATE')
    wrapper = Connection(raw)
    database.connection = wrapper
    try:
        with wrapper:
            yield
        if not borrowed:
            raw.commit()
    except BaseException:
        if not borrowed:
            raw.rollback()
        raise
    finally:
        database.connection = raw
