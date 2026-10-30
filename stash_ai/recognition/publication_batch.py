"""Batch network work while retaining one durable journal per relationship."""
import json
from .database import encoded
from .identity_merges import canonical
from .publication import (ensure_person,publish_person,operation_id,PendingRecognition,
                          LINK_INPUT_CONTRACT,legacy_rejected_link,before_dispatch)
from .stash import checked_id,checked_token,owns


def _rows(database,scope,people):
    rows={}
    for person in people:
        row=database.connection.execute('SELECT * FROM people WHERE id=? AND scope=?',(person,scope)).fetchone()
        if row is None or canonical(database.connection,person)!=person:
            raise PendingRecognition('Missing or merged identity cannot be published.')
        checked_token(database,person,row['token']);rows[person]=row
    return rows


def _read(client,identifiers):
    method=getattr(client,'performers',None)
    return method(identifiers) if callable(method) else {i:client.performer(i) for i in identifiers}


def _owned(row,live):
    if isinstance(live,BaseException):raise live
    if not owns(live,row['token']):
        raise PendingRecognition('Stored performer ownership changed; no duplicate profile was created.')
    return checked_id(live,row['performer'])['id']


def ensure_people(database,client,report,scope,scene,people,*,guard=None,progress=None):
    if guard:guard()
    rows=_rows(database,scope,people)
    existing=_read(client,[r['performer'] for r in rows.values() if r['performer']])
    targets={}
    for person,row in rows.items():
        targets[person]=(_owned(row,existing.get(row['performer'])) if row['performer'] else
            ensure_person(database,client,report,scope,scene,person,guard=guard))
        if progress:progress()
    return targets


def _record(db,scope,scene,person,performer,payload,precondition,status):
    identifier=operation_id(scope,scene,person,'link')
    db.execute('INSERT OR REPLACE INTO publications VALUES (?,?,?,?,?,?,?,?,?,NULL)',
        (identifier,scope,scene,person,'link',performer,encoded(payload),encoded(precondition),status))
    return identifier


def publish_people(database,client,report,scope,scene,targets,*,guard=None):
    method=getattr(client,'add_scene_people',None)
    if not callable(method):
        for person in targets:
            publish_person(database,client,report,scope,scene,person,reconcile_merges=False,guard=guard)
        return
    if not targets:return
    if guard:guard()
    db=database.connection;rows=_rows(database,scope,targets)
    # Revalidate ownership immediately before this scene's ADD, including
    # already-linked profiles. Never share these snapshots across scenes.
    live=_read(client,list(targets.values()))
    for person,performer in targets.items():
        if rows[person]['performer']!=performer or _owned(rows[person],live.get(performer))!=performer:
            raise PendingRecognition('Recognition performer binding changed before publication.')
    scene_row=client.scene(scene)
    existing={checked_id(item)['id'] for item in scene_row['performers']}
    missing=[];observed=[];unresolved=False
    for person,performer in targets.items():
        pending=db.execute('SELECT * FROM publications WHERE id=?',(operation_id(scope,scene,person,'link'),)).fetchone()
        prior=json.loads(pending['payload']) if pending else {}
        if performer in existing:
            managed=bool(pending and pending['status'] in ('pending','unknown','confirmed','observed')
                and prior.get('managed',True) and prior.get('active',True))
            observed.append((person,performer,dict(performer=performer,input_contract=LINK_INPUT_CONTRACT,
                managed=managed,active=True),json.loads(pending['precondition']) if pending else {}))
        else:
            precondition=dict(performers=sorted(existing))
            if pending and pending['status'] in ('pending','unknown'):
                if not legacy_rejected_link(pending,client,performer):
                    unresolved=True;continue
                precondition['legacy_validation_failure']='BulkUpdateIds.values rejected with HTTP 400'
            missing.append((person,performer,dict(performer=performer,input_contract=LINK_INPUT_CONTRACT,
                managed=True,active=True),precondition))
    with db:
        for person,performer,payload,precondition in observed:
            _record(db,scope,scene,person,performer,payload,precondition,'observed')
    if unresolved:raise PendingRecognition('Earlier additive scene link is unresolved; no blind retry.')
    # Bound each remote write and its recovery group independently.
    for start in range(0,len(missing),32):
        batch=missing[start:start+32]
        if guard:guard()
        if start:
            current=_rows(database,scope,[entry[0] for entry in batch])
            refreshed=_read(client,[entry[1] for entry in batch])
            for person,performer,_,_ in batch:
                if current[person]['performer']!=performer or _owned(current[person],refreshed.get(performer))!=performer:
                    raise PendingRecognition('Recognition performer binding changed between publication batches.')
        with db:
            identifiers=[_record(db,scope,scene,*entry,'pending') for entry in batch]
        from stash_ai.runtime.mutations import write
        try:
            def dispatch():
                with before_dispatch():
                    if guard:guard()
                return method(scene,[entry[1] for entry in batch])
            write(report,'recognition_link_batch',dispatch)
        except BaseException as error:
            from .diagnostics import message
            from stash_ai.runtime.cleanup import cleanup
            status='failed' if getattr(error,'write_outcome',None)=='failed' else 'unknown'
            def record():
                with db:db.executemany('UPDATE publications SET status=?,error=? WHERE id=?',
                    [(status,message(error,getattr(client,'api_key',''))[:1000],i) for i in identifiers])
            cleanup([('record recognition batch outcome',record)])
            raise
        with db:db.executemany("UPDATE publications SET status='confirmed',error=NULL WHERE id=?",[(i,) for i in identifiers])
        confirmed={checked_id(item)['id'] for item in client.scene(scene)['performers']}
        if not {entry[1] for entry in batch}<=confirmed:
            raise PendingRecognition('Scene links disappeared after acknowledgement; rerun to reconcile.')
