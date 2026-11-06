"""Bounded fresh remote reads for already reconciled scene relationships.

Snapshots authorize skipping work only. Every mutation uses the existing fresh
ownership checks and journal; a write invalidates the remainder of the batch.
"""
import json
from .database import encoded
from .stash import checked_id
from .publication_batch import _rows,_read,_owned


class Probe:
    def __init__(self,database,client,scope,namespace):
        self.database,self.client,self.scope,self.namespace=database,client,scope,namespace
        self.clear()

    def clear(self):self.scenes={};self.profiles={}

    def prepare(self,items,rebuilt):
        self.clear()
        try:
            people=sorted({g['person'] for item in items for g in rebuilt[item['scan']] if g['status']=='accepted'})
            rows=_rows(self.database,self.scope,people)
            ids=sorted({r['performer'] for r in rows.values() if r['performer']})
            # The client bounds GraphQL aliases too; bound fallback adapters here.
            for start in range(0,len(ids),32):self.profiles.update(_read(self.client,ids[start:start+32]))
            scenes=[item['scene'] for item in items]
            method=getattr(self.client,'scenes',None)
            self.scenes=method(scenes) if callable(method) else {s:self.client.scene(s) for s in scenes}
        except Exception:
            # The ordinary per-scene path retains diagnostics and partial progress.
            self.clear()

    def unchanged(self,scene,people):
        live=self.scenes.get(scene)
        if not isinstance(live,dict):return None
        try:
            live=checked_id(live,scene)
            linked={checked_id(p)['id'] for p in live['performers']}
            rows=_rows(self.database,self.scope,people)
            if any(not r['performer'] or r['performer'] not in self.profiles for r in rows.values()):return None
            targets={p:_owned(r,self.profiles.get(r['performer'])) for p,r in rows.items()}
            if not set(targets.values())<=linked:return None
            db=self.database.connection
            journals=list(db.execute('SELECT * FROM publications WHERE scope=? AND scene=?',(self.scope,scene)))
            seen=set()
            for row in journals:
                if row['status'] in ('pending','unknown','failed'):return None
                if row['kind']!='link':continue
                payload=json.loads(row['payload'])
                if row['person'] in targets and row['target']==targets[row['person']] and payload.get('active',True):
                    seen.add(row['person'])
                elif payload.get('active',True) and payload.get('managed',True):return None
            if seen!=set(people):return None
            if db.execute('SELECT 1 FROM people WHERE scope=? AND namespace<>? AND performer IN '
                '(SELECT value FROM json_each(?)) LIMIT 1',(self.scope,self.namespace,encoded(sorted(linked)))).fetchone():return None
            return targets
        except Exception:return None
