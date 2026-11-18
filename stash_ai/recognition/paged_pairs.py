"""Exact pair deduplication with a bounded SQLite page cache in run scratch."""
from contextlib import closing
from .scratch_space import connect
from .memory_size import Reservation


class Pairs:
    def __init__(self, limit):
        self.reservation = Reservation(limit, label='body-pair-pages')
        self.db = None
        self.count = 0

    def __enter__(self):
        self.reservation.set(512 * 1024)
        try:
            self.db = connect()
            self.db.execute('PRAGMA cache_size=-256')
            self.db.execute('CREATE TABLE pairs(a TEXT,b TEXT,PRIMARY KEY(a,b)) WITHOUT ROWID')
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def add(self, pair):
        changed = self.db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?)', tuple(sorted(pair))).rowcount
        self.count += bool(changed)
        return bool(changed)

    def __iter__(self):
        yield from self.db.execute('SELECT a,b FROM pairs ORDER BY a,b')

    def __exit__(self,*args):
        try:
            if self.db is not None:self.db.close()
        finally:self.reservation.set(0)
