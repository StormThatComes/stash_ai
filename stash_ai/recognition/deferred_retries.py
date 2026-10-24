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
        for row in db.execute('SELECT observation,person FROM members ORDER BY observation'):
            digest.update(encoded(tuple(row)).encode())
        self.generation=digest.hexdigest()
        db.execute('CREATE TEMP TABLE IF NOT EXISTS deferred_retry_cache '
                   '(candidate TEXT PRIMARY KEY,input TEXT NOT NULL,generation TEXT NOT NULL)')

    def input(self,row):
        return hashlib.sha256(encoded([row['id'],row['geometry'],row['dimensions'],
            bytes(row['embedding']).hex()]).encode()).hexdigest()

    def pending(self,rows):
        for row in rows:
            prior=self.db.execute('SELECT input,generation FROM deferred_retry_cache WHERE candidate=?',
                                  (row['candidate'],)).fetchone()
            if prior is not None and tuple(prior)==(self.input(row),self.generation):
                count('deferred_retries_reused')
                continue
            count('deferred_retries_queued')
            yield row

    def remember(self,row):
        self.db.execute('INSERT OR REPLACE INTO deferred_retry_cache VALUES (?,?,?)',
                        (row['candidate'],self.input(row),self.generation))
