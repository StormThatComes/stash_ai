"""Coordinator-only bounded transactions for replayable VTT cache metadata."""
from contextvars import ContextVar
from functools import wraps

from stash_ai.runtime.cleanup import cleanup

_active=ContextVar('recognition_portrait_writes',default=None)


class Writes:
    def __init__(self,db):
        self.db=db
        self.items=self.bytes=0

    def execute(self,sql,parameters):
        result=self.db.execute(sql,parameters)
        self.items+=1
        self.bytes+=sum(len(v.encode()) if isinstance(v,str) else len(v) if isinstance(v,bytes) else 64
                        for v in parameters)
        if self.items>=128 or self.bytes>=4*1024*1024:
            self.flush()
        return result

    def flush(self):
        if self.items:
            self.db.commit()
            self.items=self.bytes=0


def execute(db,sql,parameters):
    active=_active.get()
    if active is not None and active.db is db:
        return active.execute(sql,parameters)
    with db:
        return db.execute(sql,parameters)


def flush(db):
    active=_active.get()
    if active is not None and active.db is db:
        active.flush()


def batched(function):
    @wraps(function)
    def call(database,*args,**kwargs):
        current=_active.get()
        if current is not None and current.db is database.connection:
            return function(database,*args,**kwargs)
        writer=Writes(database.connection)
        token=_active.set(writer)
        try:
            result=function(database,*args,**kwargs)
            writer.flush()
            return result
        except BaseException:
            cleanup([('roll back VTT cache metadata',database.connection.rollback)])
            raise
        finally:
            _active.reset(token)
    return call
