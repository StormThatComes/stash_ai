"""Repeatable disk-staged scene plans; only one scene is decoded at a time."""
import json
import sqlite3
from .database import encoded
from .resource_errors import ResourceLimit


class Plans:
    def __init__(self,limit,disk_limit):
        self.limit,self.disk_limit,self.used=limit,disk_limit,0
        from .scratch_space import connect
        self.db=connect()
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA cache_size=-512')
        self.db.execute('CREATE TABLE plans(position INTEGER PRIMARY KEY,ids TEXT,payload TEXT)')
    def append(self,plan):
        scan,rows,decisions=plan
        ids = encoded([r['id'] for r in rows])
        value = encoded([dict(scan), decisions])
        charge = len(ids.encode()) + len(value.encode())
        if charge+len(rows)*128>self.limit:
            raise ResourceLimit('One person scene plan exceeds sapiensid.execution.plan_bytes.')
        if self.used+charge>self.disk_limit:
            raise ResourceLimit('Staged body plans exceed execution.replay_snapshot_bytes.')
        self.db.execute('INSERT INTO plans(ids,payload) VALUES (?,?)',(ids,value));self.used+=charge
    def identifiers(self):
        for ids, in self.db.execute('SELECT ids FROM plans ORDER BY position'):
            yield from json.loads(ids)

    def __iter__(self):
        for ids, value in self.db.execute('SELECT ids,payload FROM plans ORDER BY position'):
            scan, decisions = json.loads(value)
            yield scan, [dict(id=oid) for oid in json.loads(ids)], decisions
    def __bool__(self):return self.used>0
    def __enter__(self):return self
    def __exit__(self,*error):self.db.close()
