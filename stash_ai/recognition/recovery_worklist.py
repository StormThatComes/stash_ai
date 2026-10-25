"""Private replay cue worklist; a changed frozen bank invalidates the filter."""
from contextlib import contextmanager
from .clustering import reference_context


def condition(db, alias='o'):
    if not db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='recovery_filter'").fetchone():return '1=1'
    return ('((SELECT active FROM recovery_filter)=0 OR EXISTS (SELECT 1 FROM recovery_active_cues w '
            f'WHERE w.scan={alias}.scan AND w.cue={alias}.cue))')


def bank(db, groups):
    if not db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='recovery_filter'").fetchone():return
    key=reference_context(groups)
    row=db.execute('SELECT bank FROM recovery_filter').fetchone()
    if row[0]!=key:db.execute('UPDATE recovery_filter SET bank=?,active=0',(key,))


def advance(db):
    db.execute('DELETE FROM recovery_active_cues')
    db.execute('INSERT INTO recovery_active_cues SELECT * FROM recovery_dirty_cues')
    db.execute('DELETE FROM recovery_dirty_cues')
    db.execute('UPDATE recovery_filter SET active=1')


@contextmanager
def scope(db):
    columns={r[1] for r in db.execute('PRAGMA table_info(candidates)')}
    enabled={'scan','payload'}<=columns and db.execute("SELECT 1 FROM sqlite_master WHERE name='observations'").fetchone()
    if not enabled:
        yield False;return
    db.execute('CREATE TEMP TABLE recovery_dirty_cues(scan TEXT,cue INTEGER,PRIMARY KEY(scan,cue))')
    db.execute('CREATE TEMP TABLE recovery_active_cues(scan TEXT,cue INTEGER,PRIMARY KEY(scan,cue))')
    db.execute('CREATE TEMP TABLE recovery_filter(active INTEGER,bank TEXT)')
    db.execute('INSERT INTO recovery_filter VALUES (0,NULL)')
    triggers=[]
    try:
        for table in ('candidates','members'):
            for event in ('INSERT','UPDATE','DELETE'):
                sides=('new',) if event=='INSERT' else ('old',) if event=='DELETE' else ('old','new')
                body=[]
                for side in sides:
                    address=side+'.observation' if table=='members' else f"json_extract({side}.payload,'$.observation')"
                    body.append('INSERT OR IGNORE INTO recovery_dirty_cues SELECT scan,cue FROM observations WHERE id='+address+';')
                name='recovery_watch_'+table+'_'+event.lower();triggers.append(name)
                db.execute(f'CREATE TEMP TRIGGER {name} AFTER {event} ON {table} BEGIN '+''.join(body)+' END')
        yield True
    finally:
        for name in triggers:db.execute('DROP TRIGGER IF EXISTS '+name)
        for name in ('recovery_filter','recovery_active_cues','recovery_dirty_cues'):db.execute('DROP TABLE '+name)
