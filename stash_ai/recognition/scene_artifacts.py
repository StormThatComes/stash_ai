"""Reuse only scene-independent evidence before global competition."""
from . import derived_cache
from .clustering import matching_policy
from .config import digest, plain
from .database import encoded
from .rebuild_cache import resource_limits


def face_keys(db, selected, config, *, source=None):
    if derived_cache.current() is None or not config.data['clustering']['anchor_view_matching']:
        return {}
    from .dependencies import cached
    from .dependency_schema import validate_triggers
    # Replay compacts quality and canonicalizes duplicate pixels across the
    # selection. Sign those exact replay values separately from durable vectors.
    audited=source is not None and source.execute('PRAGMA user_version').fetchone()[0]>=8
    if audited:validate_triggers(source,required=True)
    grouping=config.data['clustering']
    # Only inputs read by local_cohorts, pose_policy and view witnesses.
    local={k:v for k,v in grouping.items() if k.startswith(('min_anchor_','min_auxiliary_')) or k in (
        'anchor_view_matching','max_anchor_pose_deviation','view_cohesion_threshold','view_match_threshold',
        'max_view_witnesses','adaptive_view_witnesses','max_view_prototypes_per_person','view_min_support','max_reference_bytes','max_people_per_lookup')}
    settings=['local-order-v2',plain(local),config.data['recognition']['min_identity_support'],
              config.data['recognition']['match_threshold']]
    keys={}
    for item in selected:
        scan=item['scan']
        vectors='SELECT id,dimensions,embedding FROM observations WHERE scan=? ORDER BY id'
        def vector_digest():
            return derived_cache.fingerprint(source if audited else db,[(vectors,(scan,))],scan)
        signed=(cached(source,'scene-face-vectors',scan,'v1',vector_digest,
                       dependency='raw-observations',dependency_key=scan,checked=True) if audited else vector_digest())
        keys[scan]=derived_cache.fingerprint(db,[
            ('SELECT id,cue,seconds,geometry,quality,rejected FROM observations WHERE scan=? ORDER BY id',(scan,))
        ],[settings,item,signed])
    return keys


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


def lookup(database,scan,config,base):
    if base is None:
        return None,None
    db=database.connection
    prior=db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='prior_attachment_queries'").fetchone()
    hints=[r[0] for r in db.execute("SELECT q.id FROM prior_attachment_queries q JOIN observations o "
        "ON o.id=q.id WHERE o.scan=? ORDER BY q.id",(scan,))] if prior else []
    key=digest([base,hints,database.installation,matching_policy(config),resource_limits(config),
                config.data['execution']['rebuild_memory_bytes']])
    store=derived_cache.current()
    entry=store.get('strict-scene',key)
    return key,entry


def match(database, scan, config, base, build, *, prepared=None):
    if base is None:
        build()
        return False
    db=database.connection
    store=derived_cache.current()
    key,entry=prepared if prepared is not None else lookup(database,scan,config,base)
    if entry is None:
        build()
        candidates=[list(r) for r in db.execute('SELECT * FROM candidates WHERE scan=? ORDER BY rowid',(scan,))]
        members=[list(r) for r in db.execute('SELECT m.* FROM members m JOIN observations o ON o.id=m.observation '
                                              'WHERE o.scan=? ORDER BY m.rowid',(scan,))]
        people=[list(r) for r in db.execute('SELECT p.* FROM people p WHERE id IN '
            '(SELECT m.person FROM members m JOIN observations o ON o.id=m.observation WHERE o.scan=?) ORDER BY created,id',(scan,))]
        store.put('strict-scene',key,[people,candidates,members])
        return False
    people,candidates,members=entry
    for person in people:
        person[3]=database.identity_clock()
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',person)
    db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)',candidates)
    db.executemany('INSERT INTO members VALUES (?,?)',members)
    # Membership triggers record the exact delta. The next consumer admits the
    # resulting bank once; no full snapshot is needed for each cached scene.
    return True


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
    from .dependencies import cached,version
    from .dependency_schema import validate_triggers
    audited=db.execute('PRAGMA user_version').fetchone()[0]>=10
    if audited:validate_triggers(db,required=True)
    # Membership, old/new head selection, REPLACE and owner age can span scans.
    # Use the audited global clock for joined inputs. Keep owner digests separate
    # so multiple scene plans reuse a shared veto bank.
    from .local_dependencies import available as local_available,scene_key
    local_clocks=local_available(db)
    def signed(kind,key,records,dependency,dependency_key):
        read=lambda:derived_cache.fingerprint(db,records,key)
        return cached(db,kind,key,'v3-'+str(version(db,'heads')),read,checked=True,dependency=dependency if local_clocks else 'global',
                      dependency_key=dependency_key if local_clocks else '*') if audited else read()
    banks=[]
    for owner in sorted({root for _,root in roots}):
        banks.append(signed('body-scene-owner',owner,[(
            'SELECT m.person,o.id,o.dimensions,o.embedding FROM observations o JOIN members m '
            'ON m.observation=o.id JOIN scans s ON s.id=o.scan WHERE m.person=? '
            "AND o.rejected IS NULL AND s.status='complete' AND "+predicate()+' ORDER BY o.id',(owner,))],'owner',owner))
    local=signed('body-scene-input',digest([scan,expected]),queries,'scene',scene_key(db,scan))
    return digest([scan,policy(config),plain(config.data['sapiensid']['execution']),sorted(roots),local,banks])
