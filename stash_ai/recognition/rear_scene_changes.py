"""Transactional scene dirtiness for direct incremental rear-gallery queries."""

TABLES = ('rear_dirty_scenes','rear_indexed_scenes')
table_statements = (
    'CREATE TABLE rear_dirty_scenes(scope TEXT,namespace TEXT,scene TEXT,PRIMARY KEY(scope,namespace,scene))',
    'CREATE TABLE rear_indexed_scenes(scope TEXT,namespace TEXT,scene TEXT,scan TEXT,identity TEXT,'
    'PRIMARY KEY(scope,namespace,scene),UNIQUE(scope,namespace,scan))',
)
TRIGGERS = {}
for event in ('INSERT','UPDATE'):
    name = 'recognition_rear_clock_'+event.lower()
    TRIGGERS[name] = (f"CREATE TRIGGER {name} AFTER {event} ON evidence_versions WHEN new.kind='scene' BEGIN "
                      "INSERT INTO rear_dirty_scenes VALUES(json_extract(new.key,'$[0]'),"
                      "json_extract(new.key,'$[1]'),json_extract(new.key,'$[2]')) ON CONFLICT DO NOTHING; END")
# Old addresses on destructive operations need a posting even when the last scan
# for that scene has disappeared before the ordinary version trigger runs.
for event in ('DELETE','UPDATE','INSERT'):
    name = 'recognition_rear_scan_'+event.lower()
    sides = ('old',) if event=='DELETE' else ('new',) if event=='INSERT' else ('old','new')
    mark = ''.join(f'INSERT INTO rear_dirty_scenes VALUES({s}.scope,{s}.namespace,{s}.scene) ON CONFLICT DO NOTHING;' for s in sides)
    TRIGGERS[name] = f'CREATE TRIGGER {name} AFTER {event} ON scans BEGIN '+mark+' END'
# SQLite REPLACE can bypass DELETE triggers on the old row.
TRIGGERS['recognition_rear_scan_replace'] = ('CREATE TRIGGER recognition_rear_scan_replace BEFORE INSERT ON scans BEGIN '
    'INSERT INTO rear_dirty_scenes SELECT scope,namespace,scene FROM scans WHERE id=new.id ON CONFLICT DO NOTHING; END')
SCHEMA = ';\n'.join((*table_statements,*TRIGGERS.values()))+';\n'


class Directory:
    def __init__(self,db,scope,namespace,index):
        self.db,self.address,self.index = db,(scope,namespace),index

    def select(self,selected,nominated,heads_version):
        from .scan_heads import predicate
        from .database import encoded
        db = self.db
        dirty = [r[0] for r in db.execute('SELECT scene FROM rear_dirty_scenes WHERE scope=? AND namespace=?',self.address)]
        # Consume before applying membership deltas. Changes made during replay
        # are retained for the next pass. A rollback restores the worklist.
        db.execute('DELETE FROM rear_dirty_scenes WHERE scope=? AND namespace=?',self.address)
        previous = self.index.previous
        direct = (nominated is not None and previous.get('directory_heads')==heads_version and len(dirty)<=512)
        self.index.state['directory_heads'] = heads_version
        if not direct:
            # A new policy, a different remembered head, or a broad invalidation.
            for old, in db.execute('SELECT d.scan FROM rear_indexed_scenes d WHERE d.scope=? AND d.namespace=? '
                    'AND NOT EXISTS (SELECT 1 FROM scans s WHERE s.id=d.scan AND s.scope=d.scope '
                    "AND s.namespace=d.namespace AND s.scene=d.scene AND s.status='complete' AND "+predicate()+')',self.address).fetchall():
                self.remove(old)
            return None
        wanted = set(selected)|set(nominated)
        args = (encoded(dirty),*self.address)
        query = ('SELECT s.scene,s.id FROM json_each(?) q CROSS JOIN scans s ON s.scene=q.value '
                 "WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND "+predicate())
        current = dict(db.execute(query,args))
        for scene,old in db.execute('SELECT d.scene,d.scan FROM json_each(?) q CROSS JOIN rear_indexed_scenes d ON d.scene=q.value '
                                    'WHERE d.scope=? AND d.namespace=?',args).fetchall():
            if current.get(scene)!=old:
                self.remove(old)
        return wanted|set(current.values())

    def remove(self,scan):
        self.index.post(scan,set())
        self.db.execute('DELETE FROM metadata WHERE key=?',(self.index.prefix+'post:'+scan,))
        self.db.execute('DELETE FROM rear_indexed_scenes WHERE scope=? AND namespace=? AND scan=?',(*self.address,scan))

    def remember(self,scene,scan,identity):
        from .database import encoded
        value = encoded(identity)
        self.db.execute('INSERT INTO rear_indexed_scenes VALUES (?,?,?,?,?) ON CONFLICT(scope,namespace,scene) '
                        'DO UPDATE SET scan=excluded.scan,identity=excluded.identity WHERE scan<>excluded.scan '
                        'OR identity<>excluded.identity',(*self.address,scene,scan,value))
