"""Retire disabled body routes before face replay binds stable identity labels."""
import json
from .database import encoded


def _restore_faces(db,alias,proof,root):
    """Only undo an explicit downgrade; unrelated later decisions stay intact."""
    from .reference_roles import observation_sql,STRICT_REASONS
    source=alias['source']
    ids=proof.get('face_witnesses',{}).get(source,[])
    for row in db.execute('SELECT c.* FROM candidates c WHERE c.person=? AND '
            "json_extract(c.payload,'$.reason')='body_global_identity' AND ("
            "json_extract(c.payload,'$.body_merge_origin.person')=? OR "+observation_sql('c')+
            ' IN (SELECT value FROM json_each(?)))',(root,source,encoded(ids))).fetchall():
        payload=json.loads(row['payload']);origin=payload.get('body_merge_origin')
        if origin and origin['person']!=source:continue
        reason=(origin or {}).get('reason') or payload.get('prior_reason')
        if not reason or reason=='body_global_identity':continue
        payload.update(reason=reason,reference_role=(origin or {}).get('reference_role') or
                       ('strict' if reason in STRICT_REASONS else 'attachment'),
                       reference_selected=bool((origin or {}).get('reference_selected',False)))
        payload.pop('body_merge_origin',None)
        db.execute('UPDATE candidates SET person=?,payload=? WHERE id=?',(source,encoded(payload),row['id']))
        db.execute('UPDATE members SET person=? WHERE observation=? AND person=?',(source,payload['observation'],root))


def retire_disabled(database,selected,config):
    if not selected:return
    from .body_identity import enabled
    routes=set()
    if not enabled(config):routes.update(('body_scene','body_global'))
    elif not config.data['sapiensid']['global_matching']['enabled']:routes.add('body_global')
    if not routes:return
    db=database.connection
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    selection={item['scene'] for item in selected}
    from .body_global import _scenes
    from .identity_merges import canonical
    pending=[]
    for row in list(db.execute('SELECT a.* FROM identity_merges a JOIN people p ON p.id=a.source '
                              'WHERE p.scope=? AND p.namespace=?',(scope,namespace))):
        proof=json.loads(row['evidence'])
        if proof.get('route') not in routes:continue
        involved=set(proof.get('scenes',()))|_scenes(db,[row['source'],row['target']])
        if involved and involved<=selection:
            pending.append((row,proof,canonical(db,row['target'])))
    for row,proof,root in pending:
        _restore_faces(db,row,proof,root)
        db.execute('DELETE FROM identity_merges WHERE source=?',(row['source'],))

