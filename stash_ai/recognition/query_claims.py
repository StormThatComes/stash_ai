"""Disk-owned query decisions; vector lifetime is bounded by one cue/family."""
from contextlib import AbstractContextManager
import pickle
import sqlite3

from .database import encoded
from .memory_size import Reservation, size
from .resource_errors import ResourceLimit


class Claims(AbstractContextManager):
    def __init__(self, config):
        from .scratch_space import connect
        self.db = connect()
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA cache_size=-512')
        self.db.executescript('''
            CREATE TABLE claims(ordinal INTEGER PRIMARY KEY, observation TEXT UNIQUE,
                family TEXT, prior TEXT, owner TEXT, payload BLOB, active INTEGER DEFAULT 1);
            CREATE INDEX families ON claims(active,family,ordinal);
            CREATE TABLE cues(scan TEXT,cue INTEGER,PRIMARY KEY(scan,cue));
            CREATE TABLE conflicts(id TEXT PRIMARY KEY,person TEXT,scan TEXT,cue INTEGER,
                geometry TEXT,dimensions INTEGER,embedding BLOB);
            CREATE INDEX conflict_order ON conflicts(person,scan,cue,id);
            CREATE TABLE support(family TEXT PRIMARY KEY,total INTEGER,minimum INTEGER);
        ''')
        self.limit = config.data['sapiensid']['execution']['plan_bytes']
        self.used = 0
        self.reservation = Reservation(config.data['execution']['rebuild_memory_bytes'], label='hybrid-query-working')
        self.reservation.set(1024**2)

    def add(self, row, claim, family):
        payload = pickle.dumps(claim, protocol=5)
        self.used += len(payload) + len(row['embedding']) + len(row['geometry']) + 1024
        if self.used > self.limit:
            raise ResourceLimit('SapiensID query plan_bytes budget exceeded.')
        self.db.execute('INSERT INTO claims(observation,family,prior,owner,payload) VALUES (?,?,?,?,?)',
            (row['id'],encoded(family),claim[4],claim[2],payload))
        self.conflict(dict(row, person=claim[2]))
        self.db.execute('INSERT OR IGNORE INTO cues VALUES (?,?)', (row['scan'],row['cue']))

    def conflict(self, row):
        self.db.execute('INSERT OR IGNORE INTO conflicts VALUES (?,?,?,?,?,?,?)',
            tuple(row[k] for k in ('id','person','scan','cue','geometry','dimensions','embedding')))

    def rows(self, *, active=False):
        for row in self.db.execute('SELECT payload FROM claims '+('WHERE active=1 ' if active else '')+'ORDER BY ordinal'):
            yield pickle.loads(row[0])

    def complete_births(self, counts):
        return {p for p,n,owners in self.db.execute('SELECT prior,COUNT(*),COUNT(DISTINCT owner) '
            'FROM claims WHERE active=1 AND prior IS NOT NULL GROUP BY prior') if n==counts.get(p) and owners==1}

    def resolve(self, counts, minimum):
        from itertools import groupby
        from .independent_evidence import Votes
        complete=self.complete_births(counts)
        self.db.execute('UPDATE claims SET active=0 WHERE prior IN (SELECT value FROM json_each(?)) '
            'AND prior NOT IN (SELECT value FROM json_each(?))',(encoded(list(counts)),encoded(sorted(complete))))
        while True:
            self.db.execute('DELETE FROM support')
            rows=self.db.execute('SELECT family,payload FROM claims WHERE active=1 ORDER BY family,ordinal')
            for family, records in groupby(rows,lambda r:r['family']):
                votes=Votes();charge=1024**2;first=None
                for row in records:
                    claim=pickle.loads(row['payload']);first=claim if first is None else first
                    # Includes graph edges, keys, matching maps and decode scratch.
                    charge += size(claim[5])*8+4096
                    self.reservation.set(charge)
                    votes.add(claim[5])
                total=len(votes)
                self.db.execute('INSERT INTO support VALUES (?,?,?)',(family,total,minimum(first)))
                self.reservation.set(1024**2)
            self.db.execute('UPDATE claims SET active=0 WHERE family IN (SELECT family FROM support WHERE total<minimum)')
            complete=self.complete_births(counts)
            removed=self.db.execute('UPDATE claims SET active=0 WHERE active=1 AND prior IN '
                '(SELECT value FROM json_each(?)) AND prior NOT IN (SELECT value FROM json_each(?))',
                (encoded(list(counts)),encoded(sorted(complete)))).rowcount
            if not removed:return complete

    def accepted(self, observation):
        return bool(self.db.execute('SELECT active FROM claims WHERE observation=?',(observation,)).fetchone()[0])

    def support(self, family):
        row=self.db.execute('SELECT total FROM support WHERE family=?',(encoded(family),)).fetchone()
        return row[0] if row else 0

    def __exit__(self,*error):
        try:self.db.close()
        finally:self.reservation.set(0)
