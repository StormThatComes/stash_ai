"""Optional cue membership index with bounded RAM and a scratch SQLite spill."""
from .database import encoded
from .resource_errors import ResourceLimit


class CueIndex:
    def __init__(self,budget):self.budget=budget;self.used=0;self.owners={};self.db=None
    def add(self,owner,scan,cue):
        if self.db is None:
            self.used+=256+len(scan)*4
            if self.used<=self.budget:
                self.owners.setdefault(owner,set()).add((scan,cue));return
            from .scratch_space import connect
            self.db=connect()
            self.db.execute('PRAGMA cache_size=-256')
            self.db.execute('CREATE TABLE cues(owner TEXT,scan TEXT,cue INTEGER,PRIMARY KEY(owner,scan,cue)) WITHOUT ROWID')
            self.db.executemany('INSERT INTO cues VALUES (?,?,?)',
                ((p,s,c) for p,values in self.owners.items() for s,c in values))
            self.owners.clear()
        self.db.execute('INSERT OR IGNORE INTO cues VALUES (?,?,?)',(owner,scan,cue))
    def overlap(self,pairs):
        if self.db is None:
            return [(a,b) for a,b in pairs if not self.owners.get(a,set()).isdisjoint(self.owners.get(b,set()))]
        return [tuple(r) for r in self.db.execute("WITH pairs(a,b) AS (SELECT json_extract(value,'$[0]'),"
            "json_extract(value,'$[1]') FROM json_each(?)) SELECT a,b FROM pairs WHERE EXISTS("
            'SELECT 1 FROM cues x JOIN cues y ON y.owner=pairs.b AND y.scan=x.scan AND y.cue=x.cue '
            'WHERE x.owner=pairs.a)',(encoded(pairs),))]
    def close(self):
        if self.db is not None:self.db.close();self.db=None
        self.owners.clear()


def build(db,owned,budget):
    if db is None or len(owned)<64:return None
    index=CueIndex(budget)
    rows=db.execute('SELECT m.person,o.scan,o.cue FROM members m JOIN observations o ON m.observation=o.id '
                    'WHERE o.rejected IS NULL GROUP BY m.person,o.scan,o.cue')
    try:
        for owner,scan,cue in rows:
            if owner in owned:index.add(owner,scan,cue)
        if index.db is not None:index.db.commit()
        return index
    except ResourceLimit:
        index.close();return None
    except BaseException:
        index.close();raise
    finally:rows.close()
