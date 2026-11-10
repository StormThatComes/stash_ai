"""Consume consolidation requests within an explicit retained-library scope.

Requests nominate scenes only. Current scans and the whole frozen catalogue are
replayed before any merge; stored identity labels and scores are never applied.
Publication outside the caller's selection remains durable publication debt.
"""
import json
from .config import digest
from .scan_heads import predicate as current_scan


def expand(db,selected,config):
    settings=config.data['sapiensid']
    policy=settings['consolidation']
    if not selected or not settings['enabled'] or policy['scope']!='retained':return selected
    first=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    if first is None:raise ValueError('Consolidation requires an existing selected scan.')
    scope,namespace=first
    row=db.execute('SELECT value FROM metadata WHERE key=?',('hybrid-pending:'+digest([scope,namespace]),)).fetchone()
    if row is not None and len(row[0].encode())>settings['execution']['plan_bytes']:
        raise ValueError('Consolidation queue exceeds plan_bytes.')
    request=json.loads(row[0]) if row is not None else dict(format=1,scope=scope,namespace=namespace,pending=[])
    if (request.get('format')!=1 or request.get('scope')!=scope or request.get('namespace')!=namespace
            or not isinstance(request.get('pending'),list)):
        raise ValueError('Invalid recognition consolidation queue.')
    chosen={s['scene']:s for s in selected};allowed=set(policy['scene_ids'])
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    scan_policy=scan_identity(config,namespace)
    for component in request['pending']:
        scans=component.get('required_scans')
        if not isinstance(scans,list) or not all(isinstance(s,str) for s in scans):
            raise ValueError('Invalid consolidation scan addresses.')
        additions={};eligible=True
        for identifier in scans:
            old=db.execute('SELECT scope,namespace,scene FROM scans WHERE id=?',(identifier,)).fetchone()
            if old is None or tuple(old[:2])!=(scope,namespace):
                eligible=False;break
            scene=old['scene']
            if scene in chosen:continue
            if allowed and scene not in allowed:eligible=False;break
            current=db.execute("SELECT s.id,s.status,s.policy_sha FROM scans s WHERE s.scope=? AND s.namespace=? AND s.scene=? "
                'AND '+current_scan(),(scope,namespace,scene)).fetchone()
            if current is None or current['status']!='complete':eligible=False;break
            compatible=db.execute('SELECT 1 FROM scans s WHERE s.id=? AND (s.policy_sha=? OR NOT EXISTS (SELECT 1 FROM observations o '
                'WHERE o.scan=s.id AND o.rejected IS NULL AND NOT '+fallback_sql(db,config,namespace)+'))',
                (current['id'],scan_policy)).fetchone()
            if compatible is None:eligible=False;break
            additions[scene]=dict(scene=scene,scan=current['id'])
        if eligible and len(chosen)+len(additions)<=policy['max_scenes']:chosen.update(additions)
    if (len(chosen) < policy['max_scenes'] and settings['global_matching']['enabled']
            and db.execute('PRAGMA user_version').fetchone()[0] >= 17):
        from .alias_scene_index import retained_components
        from .body_observations import contract
        from .body_identity import policy as body_policy
        stamp = digest([body_policy(config),settings['global_matching']])
        for scenes in retained_components(db,scope,namespace,stamp):
            missing = set(scenes) - chosen.keys()
            if not missing or (allowed and not missing <= allowed) or len(missing)+len(chosen)>policy['max_scenes']:
                continue
            from .database import encoded
            rows = db.execute("SELECT s.scene,s.id FROM scans s WHERE s.scope=? AND s.namespace=? AND "
                "s.scene IN (SELECT value FROM json_each(?)) AND s.status='complete' AND s.policy_sha=? AND "
                +current_scan()+" AND s.total=(SELECT count(*) FROM body_cues b WHERE b.scan=s.id AND b.contract=?)",
                (scope,namespace,encoded(sorted(missing)),scan_policy,contract(config)))
            additions = {scene:dict(scene=scene,scan=scan) for scene,scan in rows}
            if additions.keys()==missing:
                chosen.update(additions)
            if len(chosen)>=policy['max_scenes']:
                break
    return sorted(chosen.values(),key=lambda item:(int(item['scene']),item['scan']))
