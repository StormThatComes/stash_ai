"""Indexed pair evidence with the lifetime of a single consolidation plan."""
import json
from .database import encoded
from .scratch_space import connect


class PairProofs:
    def __init__(self):
        self.db = connect()
        self.db.execute('PRAGMA cache_size=-256')
        self.db.execute('CREATE TABLE proofs(a TEXT,b TEXT,proof TEXT,PRIMARY KEY(a,b)) WITHOUT ROWID')

    def __setitem__(self, pair, value):
        self.db.execute('INSERT INTO proofs VALUES(?,?,?)', (*pair,encoded(value)))

    def get(self, pair, default=None):
        row = self.db.execute('SELECT proof FROM proofs WHERE a=? AND b=?',pair).fetchone()
        return json.loads(row[0]) if row else default

    def items(self):
        for a,b,proof in self.db.execute('SELECT a,b,proof FROM proofs ORDER BY a,b'):
            yield (a,b),json.loads(proof)

    def __getitem__(self, pair):
        value = self.get(pair)
        if value is None:
            raise KeyError(pair)
        return value

    def close(self):
        self.db.close()
