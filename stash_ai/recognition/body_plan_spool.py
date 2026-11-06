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
        self.db.execute('CREATE TABLE plans(position INTEGER PRIMARY KEY,payload TEXT)')
    def append(self,plan):
        scan,rows,decisions=plan
        value=encoded([dict(scan),[dict(id=r['id']) for r in rows],decisions])
        charge=len(value.encode())
        if charge+len(rows)*128>self.limit:
            raise ResourceLimit('One person scene plan exceeds sapiensid.execution.plan_bytes.')
        if self.used+charge>self.disk_limit:
            raise ResourceLimit('Staged body plans exceed execution.replay_snapshot_bytes.')
        self.db.execute('INSERT INTO plans(payload) VALUES (?)',(value,));self.used+=charge
    def __iter__(self):
        for (value,) in self.db.execute('SELECT payload FROM plans ORDER BY position'):yield json.loads(value)
    def __bool__(self):return self.used>0
    def __enter__(self):return self
    def __exit__(self,*error):self.db.close()
