"""Durable publication debt; enqueue with identity changes, acknowledge after HTTP.

Only metadata is stored. Raw evidence, profile ownership and manual links remain
under their existing contracts. The normal run never widens its selection.
"""
import hashlib
import json
import uuid

from .clustering import matching_policy,vector
from .config import digest
from .database import encoded
from .reference_roles import observation_sql
from .rebuild_cache import _hash_row
from .scan_heads import predicate as current_scan

PREFIX='identity-publication:'


def key(scope,namespace,scene):
    return PREFIX+digest([scope,namespace,scene])


def current(db,scope,namespace,scene):
    row=db.execute('SELECT value FROM metadata WHERE key=?',(key(scope,namespace,scene),)).fetchone()
    if row is None:return None
    value=json.loads(row[0])
    if (value.get('format')!=1 or [value.get(k) for k in ('scope','namespace','scene')]!=[scope,namespace,scene]
        or not isinstance(value.get('revision'),str)):
        raise ValueError('Invalid recognition publication queue entry.')
    return value


def latest(db,scope,namespace,scene):
    return db.execute('SELECT s.* FROM scans s WHERE s.scope=? AND s.namespace=? AND s.scene=? AND '+current_scan(),
                      (scope,namespace,scene)).fetchone()


def enqueue(db,scope,namespace,scene,config,reason):
    previous=current(db,scope,namespace,scene)
    scan=latest(db,scope,namespace,scene)
    document=dict(format=1,scope=scope,namespace=namespace,scene=scene,
        scan=scan['id'] if scan else None,policy=matching_policy(config),revision=uuid.uuid4().hex,
        reasons=sorted(set((previous or {}).get('reasons',()))|{reason}))
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key(scope,namespace,scene),encoded(document)))


def acknowledge(db,scope,namespace,scene,revision):
    if revision is None:return
    queued=current(db,scope,namespace,scene)
    if queued is not None and queued['revision']==revision:
        db.execute('DELETE FROM metadata WHERE key=?',(key(scope,namespace,scene),))


def snapshot(db,scope,namespace,scene):
    scan=latest(db,scope,namespace,scene)
    if scan is None or scan['status']!='complete':
        raise ValueError('Latest scene scan is incomplete or missing; run recognition first.')
    h=hashlib.sha256();_hash_row(h,scan);people=set()
    rows=db.execute('SELECT o.id,o.scan,o.cue,o.geometry,o.quality,o.rejected,o.dimensions,o.embedding,'
        'm.person,p.scope AS person_scope,p.namespace AS person_namespace,p.token,'
        'c.person AS decision_person,c.status,c.payload FROM observations o '
        'LEFT JOIN members m ON m.observation=o.id LEFT JOIN people p ON p.id=m.person '
        'LEFT JOIN candidates c ON c.rowid=(SELECT n.rowid FROM candidates n WHERE '+observation_sql('n')+
        '=o.id ORDER BY n.rowid DESC LIMIT 1) WHERE o.scan=? ORDER BY o.id',(scan['id'],))
    for row in rows:
        _hash_row(h,row)
        if row['rejected'] is not None:
            if row['person'] is not None:raise ValueError('Rejected face still has a membership; run recognition first.')
            continue
        if row['person'] is not None:
            if row['person_scope']!=scope or row['person_namespace']!=namespace:
                raise ValueError('Scene membership crosses recognition scope or namespace.')
            if row['payload'] is not None and (row['status']!='accepted' or row['decision_person']!=row['person']):
                raise ValueError('Scene membership contradicts its current decision; run recognition first.')
            vector(row);people.add(row['person'])
        elif row['payload'] is None or row['status']!='withheld':
            raise ValueError('Latest scene scan has no complete identity decisions; run recognition first.')
    return dict(scene=scene,scan=scan['id']),sorted(people),h.hexdigest()


def plan(db,scope,namespace,config,scenes=()):
    chosen=set(scenes);items=[]
    for row in db.execute('SELECT key,value FROM metadata WHERE key LIKE ? ORDER BY key',(PREFIX+'%',)):
        value=json.loads(row['value'])
        if value.get('scope')!=scope or value.get('namespace')!=namespace:continue
        scene=value.get('scene')
        if chosen and scene not in chosen:continue
        entry=dict(scene=scene,revision=value.get('revision'),reasons=value.get('reasons',[]),status='blocked',origin='queue')
        try:
            queued=current(db,scope,namespace,scene)
            if row['key']!=key(scope,namespace,scene):raise ValueError('Publication queue key is inconsistent.')
            if queued['policy']!=matching_policy(config):
                raise ValueError('Publication policy changed; run recognition for this scene first.')
            item,people,signature=snapshot(db,scope,namespace,scene)
            if item['scan']!=queued['scan']:raise ValueError('Scene scan changed; run recognition first.')
            entry.update(status='ready',scan=item['scan'],people=people,signature=signature)
        except (ValueError,TypeError,KeyError) as error:
            entry['reason']=str(error)
        items.append(entry)
    queued={e['scene'] for e in items}
    for scene,reasons in discover(db,scope,namespace).items():
        if scene in queued or chosen and scene not in chosen:continue
        entry=dict(scene=scene,revision=None,reasons=sorted(reasons),status='blocked',origin='journal_inventory')
        try:
            item,people,signature=snapshot(db,scope,namespace,scene)
            entry.update(status='ready',scan=item['scan'],people=people,signature=signature)
        except (ValueError,TypeError,KeyError) as error:entry['reason']=str(error)
        items.append(entry)
    return sorted(items,key=lambda e:int(e['scene']))


def discover(db,scope,namespace):
    """Read-only migration of old debt: absent or mismatched publication journals."""
    desired={}
    for row in db.execute('SELECT DISTINCT s.scene,m.person,p.performer FROM members m JOIN people p ON p.id=m.person '
        'JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? '
        "AND s.status='complete' AND o.rejected IS NULL AND "+current_scan(),(scope,namespace)):
        desired.setdefault(row['scene'],{})[row['person']]=row['performer']
    complete=set();debt={}
    for row in db.execute("SELECT w.* FROM publications w JOIN people p ON p.id=w.person WHERE w.scope=? "
        "AND p.namespace=? AND w.kind IN ('link','unlink') AND w.scene<>''",(scope,namespace)):
        payload=json.loads(row['payload']);scene=row['scene']
        if row['kind']=='unlink':
            if row['status'] in ('pending','unknown'):debt.setdefault(scene,set()).add('unresolved_unlink')
            continue
        if not payload.get('active',True):continue
        if row['status'] in ('confirmed','observed'):
            if row['person'] in desired.get(scene,{}) and desired[scene][row['person']]==row['target']:
                complete.add((scene,row['person']))
            elif payload.get('managed',True):debt.setdefault(scene,set()).add('managed_link_mismatch')
        elif row['status'] in ('pending','unknown'):
            debt.setdefault(scene,set()).add('unresolved_link')
    for scene,people in desired.items():
        if any((scene,p) not in complete for p in people):debt.setdefault(scene,set()).add('missing_publication')
    return debt


def adopt(db,scope,namespace,config,entry):
    """Only --apply turns an inspected legacy journal mismatch into a request."""
    if entry['origin']!='journal_inventory':return
    from .publication import PendingRecognition
    if current(db,scope,namespace,entry['scene']) is not None:
        raise PendingRecognition('Recognition publication request changed; refresh synchronization plan.')
    item,people,signature=snapshot(db,scope,namespace,entry['scene'])
    if signature!=entry['signature'] or people!=entry['people'] or item['scan']!=entry['scan']:
        raise PendingRecognition('Recognition scene assignments changed; refresh synchronization plan.')
    enqueue(db,scope,namespace,entry['scene'],config,'retained_publication_inventory')
    entry['revision']=current(db,scope,namespace,entry['scene'])['revision'];entry['origin']='queue'


def validate(db,scope,namespace,entry):
    from .publication import PendingRecognition
    queued=current(db,scope,namespace,entry['scene'])
    if queued is None or queued['revision']!=entry['revision']:
        raise PendingRecognition('Recognition publication request changed; refresh synchronization plan.')
    item,people,signature=snapshot(db,scope,namespace,entry['scene'])
    if signature!=entry['signature'] or people!=entry['people'] or item['scan']!=entry['scan']:
        raise PendingRecognition('Recognition scene assignments changed; refresh synchronization plan.')


def inventory(db,scope,namespace):
    active='''FROM members m JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan
        WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL
        AND '''+current_scan()
    count=db.execute('SELECT COUNT(DISTINCT m.person) '+active,(scope,namespace)).fetchone()[0]
    unused=db.execute('SELECT p.id,p.performer FROM people p WHERE p.scope=? AND p.namespace=? AND p.performer IS NOT NULL '
        'AND NOT EXISTS (SELECT 1 '+active+' AND m.person=p.id) ORDER BY p.created,p.id',
        (scope,namespace,scope,namespace)).fetchall()
    return dict(identities_with_latest_memberships=count,
        retained_profiles_without_latest_memberships=[dict(r) for r in unused])
