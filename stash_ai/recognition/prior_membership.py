"""Indexed frozen membership in the replay-owned SQLite store."""
from hashlib import sha256


class PriorMembership:
    def __init__(self, db, table, ordered_ids):
        self.db, self.table = db, table
        digest = sha256()
        self.size = 0
        def rows():
            for identifier in ordered_ids:
                value = identifier.encode()
                digest.update(len(value).to_bytes(4, 'big'))
                digest.update(value)
                self.size += 1
                yield (identifier,)
        db.execute(f'CREATE TEMP TABLE {table}(id TEXT PRIMARY KEY) WITHOUT ROWID')
        db.executemany(f'INSERT INTO {table} VALUES (?)', rows())
        self.digest = digest.hexdigest()

    def __contains__(self, key):
        return self.db.execute(f'SELECT 1 FROM {self.table} WHERE id=?',(key,)).fetchone() is not None

    def __bool__(self):
        return bool(self.size)
