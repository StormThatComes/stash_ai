"""Reuse only scene-independent evidence before global competition."""
from . import derived_cache
from .clustering import matching_policy
from .config import digest, plain
from .database import encoded
from .rebuild_cache import resource_limits


def face_keys(db, selected, config):
    if derived_cache.current() is None or not config.data['clustering']['anchor_view_matching']:
        return {}
    settings=[matching_policy(config),resource_limits(config),plain(config.data['execution'])]
    return {item['scan']:derived_cache.fingerprint(db,[
        ('SELECT id,cue,seconds,geometry,quality,rejected,dimensions,embedding FROM observations WHERE scan=? ORDER BY id',(item['scan'],))
    ],[settings,item]) for item in selected}


def prepare(db, selected, config, keys):
    from .local_cohorts import prepare as build
    if not keys:
        build(db,selected,config)
        return
    store=derived_cache.current()
    missing=[]
    db.execute('CREATE TEMP TABLE IF NOT EXISTS local_core_order(observation TEXT PRIMARY KEY,position INTEGER)')
    for item in selected:
        order=store.get('face-order',keys[item['scan']])
        if order is None:
            missing.append(item)
        else:
            db.executemany('INSERT INTO local_core_order VALUES (?,?)',order)
    if missing:
        build(db,missing,config)
    for item in missing:
        rows=db.execute('SELECT l.observation,l.position FROM local_core_order l JOIN observations o '
                        'ON o.id=l.observation WHERE o.scan=? ORDER BY o.id',(item['scan'],))
        store.put('face-order',keys[item['scan']],[list(r) for r in rows])


def match(database, scan, config, base, build):
    if base is None:
        return build()
    db=database.connection
    prior=db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='prior_attachment_queries'").fetchone()
    hints=[r[0] for r in db.execute("SELECT q.id FROM prior_attachment_queries q JOIN observations o "
        "ON o.id=q.id WHERE o.scan=? ORDER BY q.id",(scan,))] if prior else []
    key=digest([base,hints,database.installation])
    store=derived_cache.current()
    entry=store.get('strict-scene',key)
    if entry is None:
        build()
        candidates=[list(r) for r in db.execute('SELECT * FROM candidates WHERE scan=? ORDER BY rowid',(scan,))]
        members=[list(r) for r in db.execute('SELECT m.* FROM members m JOIN observations o ON o.id=m.observation '
                                              'WHERE o.scan=? ORDER BY m.rowid',(scan,))]
        people=[list(r) for r in db.execute('SELECT p.* FROM people p WHERE id IN '
            '(SELECT m.person FROM members m JOIN observations o ON o.id=m.observation WHERE o.scan=?) ORDER BY created,id',(scan,))]
        store.put('strict-scene',key,[people,candidates,members])
        return
    people,candidates,members=entry
    for person in people:
        person[3]=database.identity_clock()
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',person)
    db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)',candidates)
    db.executemany('INSERT INTO members VALUES (?,?)',members)
    from .clustering import _groups
    from .reference_usage import ReferenceUsage
    scope=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(scan,)).fetchone()
    policy=config.data['clustering']
    _groups(db,*scope,policy,usage=ReferenceUsage(policy['max_reference_bytes']),stage='retained')


def body_key(db, scan, config):
    from .body_observations import contract
    from .body_identity import policy
    from .identity_merges import canonical
    from .scan_heads import predicate
    from .reference_roles import strict_sql
    expected=contract(config)
    queries=[
        ('SELECT scope,namespace,scene,input_sha,status,total FROM scans WHERE id=?',(scan,)),
        ('SELECT COUNT(*) FROM body_cues WHERE scan=? AND contract=?',(scan,expected)),
        ('SELECT b.id,b.cue,b.seconds,b.geometry,b.quality,b.rejected,b.embedding,b.face,'
         'o.dimensions,o.embedding,o.quality,o.rejected,m.person,'+strict_sql()+
         ' FROM body_observations b LEFT JOIN observations o ON o.id=b.face '
         'LEFT JOIN members m ON m.observation=o.id WHERE b.scan=? AND b.contract=? ORDER BY b.id',(scan,expected)),
        ('SELECT b.geometry,b.cue,b.seconds,i.person,p.created FROM body_origins i JOIN body_observations b '
         'ON i.observation=b.id JOIN scans s ON s.id=b.scan JOIN scans t ON t.id=? LEFT JOIN people p ON p.id=i.person '
         'WHERE s.scope=t.scope AND s.namespace=t.namespace AND s.scene=t.scene AND s.input_sha=t.input_sha '
         'ORDER BY b.id,i.person',(scan,))]
    roots=set()
    for row in db.execute('SELECT DISTINCT m.person FROM body_observations b JOIN members m ON m.observation=b.face '
                          'WHERE b.scan=? AND b.contract=? ORDER BY m.person',(scan,expected)):
        roots.add((row[0],canonical(db,row[0])))
    queries.append(('SELECT m.person,o.id,o.dimensions,o.embedding FROM observations o JOIN members m '
        'ON m.observation=o.id JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) '
        'AND o.rejected IS NULL AND s.status=\'complete\' AND '+predicate()+' ORDER BY m.person,o.id',
        (encoded(sorted({root for _,root in roots})),)))
    return derived_cache.fingerprint(db,queries,[scan,policy(config),plain(config.data['sapiensid']['execution']),sorted(roots)])

