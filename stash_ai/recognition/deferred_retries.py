"""Replay-local negative decisions indexed by their frozen dependencies.

A new strict rival or a changed co-presence membership invalidates the generation.
No accepted decision is reused and the durable source database is never written.
"""
import hashlib
from .database import encoded
from .clustering import reference_context
from .profiling import count


class RetryQueue:
    def __init__(self,db,groups,policy):
        self.db=db
        digest=hashlib.sha256(encoded([policy,reference_context(groups)]).encode())
        self.generation=digest.hexdigest()
        self.cues={}
        db.execute('CREATE TEMP TABLE IF NOT EXISTS deferred_retry_cache '
                   '(candidate TEXT PRIMARY KEY,input TEXT NOT NULL,generation TEXT NOT NULL)')

    def input(self,row):
        return hashlib.sha256(encoded([row['id'],row['geometry'],row['dimensions'],
            bytes(row['embedding']).hex()]).encode()).hexdigest()

    def dependencies(self, row):
        # Every strict bank remains a potential rival. Membership elsewhere in
        # the library cannot affect this query's physical co-presence gate.
        if 'scan' not in row.keys() or 'cue' not in row.keys():
            return self.generation
        key = row['scan'], row['cue']
        if key not in self.cues:
            peers = self.db.execute('SELECT m.observation,m.person FROM members m '
                'JOIN observations o ON o.id=m.observation WHERE o.scan=? AND o.cue=? '
                'ORDER BY m.observation', key)
            self.cues[key] = hashlib.sha256(encoded([self.generation, [tuple(r) for r in peers]]).encode()).hexdigest()
        return self.cues[key]

    def pending(self,rows):
        for row in rows:
            prior=self.db.execute('SELECT input,generation FROM deferred_retry_cache WHERE candidate=?',
                                  (row['candidate'],)).fetchone()
            if prior is not None and tuple(prior)==(self.input(row),self.dependencies(row)):
                count('deferred_retries_reused')
                continue
            count('deferred_retries_queued')
            yield row

    def remember(self,row):
        self.db.execute('INSERT OR REPLACE INTO deferred_retry_cache VALUES (?,?,?)',
                        (row['candidate'],self.input(row),self.dependencies(row)))
