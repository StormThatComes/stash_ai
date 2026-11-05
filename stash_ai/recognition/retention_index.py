"""Transactionally maintained reverse lookups for automatic retention."""
from .progress import work

SOURCES={
    'merge':('identity_merges','source','evidence'),
    'body':('body_members','observation','proof'),
    'portrait':('portraits','id','metadata'),
    'publication':('publications','id','payload'),
    'precondition':('publications','id','precondition'),
    'candidate':('candidates','id','payload'),
}
MARKER='retention-index-v1'
TABLES=('retention_dirty','retention_links','retention_verified')
DDL="""
CREATE TABLE retention_dirty(kind TEXT NOT NULL,doc TEXT NOT NULL,PRIMARY KEY(kind,doc));
CREATE INDEX retention_dirty_kind ON retention_dirty(kind);
CREATE TABLE retention_links(kind TEXT NOT NULL,doc TEXT NOT NULL,value BLOB NOT NULL,
 PRIMARY KEY(kind,doc,value));
CREATE INDEX retention_links_target ON retention_links(kind,value);
CREATE TABLE retention_verified(run TEXT NOT NULL,scan TEXT NOT NULL,PRIMARY KEY(run,scan));
CREATE INDEX retention_verified_scan ON retention_verified(scan);
"""


def _triggers():
    triggers={}
    def install(table,kind,key,columns,phase=None):
        for event in ('INSERT','UPDATE','DELETE'):
            sides={'INSERT':('new',),'UPDATE':('old','new'),'DELETE':('old',)}[event]
            operation='UPDATE OF '+','.join(columns) if event=='UPDATE' else event
            body=''
            for side in sides:
                condition=f" AND {side}.phase='rebuild'" if phase else ''
                body+=(f"INSERT INTO retention_dirty SELECT '{kind}',{side}.{key} WHERE {side}.{key} IS NOT NULL"
                       +condition+" ON CONFLICT DO NOTHING;")
            name=f'retention_track_{kind}_{table}_{event.lower()}'
            triggers[name]=(f'CREATE TRIGGER {name} AFTER {operation} ON {table} WHEN '
                f"EXISTS(SELECT 1 FROM metadata WHERE key='{MARKER}') BEGIN {body} END")
    for kind,(table,key,column) in SOURCES.items():
        install(table,kind,key,(key,column,'scan') if kind=='candidate' else (key,column))
    install('runs','success','id',('id','status'))
    install('progress','success','run',('run','phase','data'),phase=True)
    # Repair a missing or edited projection from its original source before
    # authorizing deletion. Repair writes coalesce in the same dirty entry.
    for table,key,kind in (('retention_links','doc',None),('retention_verified','run','success')):
        for event in ('INSERT','UPDATE','DELETE'):
            sides={'INSERT':('new',),'UPDATE':('old','new'),'DELETE':('old',)}[event]
            name=f'retention_repair_{table}_{event.lower()}'
            body=''
            for side in sides:
                label=side+'.kind' if kind is None else "'success'"
                body+=f'INSERT INTO retention_dirty VALUES ({label},{side}.{key}) ON CONFLICT DO NOTHING;'
            triggers[name]=(f'CREATE TRIGGER {name} AFTER {event} ON {table} WHEN '
                f"EXISTS(SELECT 1 FROM metadata WHERE key='{MARKER}') BEGIN {body} END")
    return triggers


TRIGGERS=_triggers()
SCHEMA=DDL+';\n'.join(TRIGGERS.values())+';\n'


def refresh(db,writable):
    from .retention_plan import REFERENCE_BATCH_ROWS
    from .database import encoded
    ready=db.execute('SELECT value FROM metadata WHERE key=?',(MARKER,)).fetchone()
    if ready and ready[0]!='1':raise ValueError('Invalid retention index version.')
    if not writable:
        return bool(ready and not db.execute('SELECT 1 FROM retention_dirty LIMIT 1').fetchone())
    if not ready:
        db.execute('DELETE FROM retention_links')
        db.execute('DELETE FROM retention_verified')
        db.execute('DELETE FROM retention_dirty')
        for kind,(table,key,_) in SOURCES.items():
            db.execute(f'INSERT INTO retention_dirty SELECT ?,{key} FROM {table} ORDER BY rowid',(kind,))
        db.execute("INSERT INTO retention_dirty SELECT 'success',run FROM progress WHERE phase='rebuild'")
    for kind,(table,key,column) in {**SOURCES,'success':('progress','run','data')}.items():
        with work(f'Inspect references: {table}.{column}',unit='record',
                  total=lambda:db.execute('SELECT COUNT(*) FROM retention_dirty WHERE kind=?',(kind,)).fetchone()[0]) as progress:
            while True:
                rows=db.execute('SELECT doc FROM retention_dirty WHERE kind=? ORDER BY rowid LIMIT ?',
                                (kind,REFERENCE_BATCH_ROWS)).fetchall()
                if not rows:break
                ids=encoded([r[0] for r in rows])
                selected='(SELECT value FROM json_each(?))'
                if kind=='success':
                    db.execute('DELETE FROM retention_verified WHERE run IN '+selected,(ids,))
                    db.execute("INSERT OR IGNORE INTO retention_verified SELECT p.run,json_extract(j.value,'$.scan') "
                        "FROM progress p JOIN runs r ON r.id=p.run CROSS JOIN json_each(p.data,'$.selection') j "
                        "WHERE p.phase='rebuild' AND r.status='success' AND p.run IN "+selected+
                        " AND json_extract(j.value,'$.scan') IS NOT NULL",(ids,))
                else:
                    if db.execute(f'SELECT 1 FROM {table} WHERE {key} IN '+selected+
                                  f' AND NOT json_valid({column}) LIMIT 1',(ids,)).fetchone():
                        raise ValueError(f'Retention cannot inspect malformed {table}.{column}; no data was pruned.')
                    db.execute('DELETE FROM retention_links WHERE kind=? AND doc IN '+selected,(kind,ids))
                    db.execute(f'INSERT OR IGNORE INTO retention_links SELECT ?,t.{key},substr(CAST(j.value AS BLOB),1,128) '
                        f'FROM {table} t CROSS JOIN json_tree(t.{column}) j WHERE t.{key} IN '+selected+
                        " AND j.type='text'",(kind,ids))
                db.execute('DELETE FROM retention_dirty WHERE kind=? AND doc IN '+selected,(kind,ids))
                progress.advance(len(rows),force=True)
    if db.execute('SELECT 1 FROM retention_dirty LIMIT 1').fetchone():
        raise ValueError('Unknown retention index dependency.')
    if not ready:db.execute('INSERT INTO metadata VALUES (?,?)',(MARKER,'1'))
    return True


def pins(db,table,column,condition,image):
    kind=next(k for k,v in SOURCES.items() if v[0]==table and v[2]==column)
    key=SOURCES[kind][1]
    # Only long addresses need to check the original document after a prefix hit.
    # Join the live source as well: an explicit rowid REPLACE can retire a
    # different primary key without firing its DELETE trigger.
    db.execute('INSERT INTO retention_pins SELECT e.id,? FROM retention_targets e '
        'CROSS JOIN retention_links l CROSS JOIN '+table+' t '
        f'WHERE l.kind=? AND l.value=substr(CAST(e.id AS BLOB),1,128) AND t.{key}=l.doc AND ('+condition+') '
        f"AND (length(CAST(e.id AS BLOB))<128 OR EXISTS(SELECT 1 FROM json_tree(t.{column}) j WHERE j.type='text' AND j.value=e.id)) "
        'ON CONFLICT(id) DO UPDATE SET image=MAX(image,excluded.image)',(image,kind))
