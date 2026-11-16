"""Directional, isolated attachment experiments against the retained catalogue."""
from collections import Counter
from contextlib import closing
from types import SimpleNamespace
import json

from .clustering import _groups, matching_policy
from .database import encoded
from .inspection_pair_replay import _fresh, sources
from .inspection_runtime import check_cancelled
from .models import namespace
from .reference_roles import observation_sql
from .reference_usage import ReferenceUsage
from .report_snapshot import snapshot


def view_queries(db, config, selected, source, keys, scope):
    from .auxiliary import _attach_confirmed_views
    from .global_views import catalogue, recover
    from .pose_policy import query_sql, restricted
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    cfg = config.data['clustering']
    policy = matching_policy(config)
    # The hypothetical source has no reference authority in this trial.
    db.execute('DELETE FROM members WHERE person=?',(source,))
    eligible = db.execute('SELECT o.id FROM observations o WHERE id IN (SELECT value FROM json_each(?)) '
                          'AND o.rejected IS NULL AND o.embedding IS NOT NULL AND '+query_sql('o',cfg),(encoded(keys),))
    for (key,) in eligible:
        db.execute("UPDATE candidates SET status='withheld',person=NULL,payload=json_object('observation',?,'reason','candidate_group') "
                   f"WHERE policy=? AND {observation_sql('candidates')}=?", (key,policy,key))
    ns=namespace(config)
    usage=ReferenceUsage(cfg['max_reference_bytes'])
    scan_policy=scan_identity(config,ns)
    anchors=_groups(db,scope,ns,cfg,usage=usage,scan_policy=scan_policy,compatibility=fallback_sql(db,config,ns))
    views=catalogue(db,config,scope,ns,usage=usage,policy=policy,scan_policy=scan_policy,complete=True)
    if cfg['anchor_view_matching']:
        recover(SimpleNamespace(connection=db),selected,config,views,anchors,policy)
    if cfg['auxiliary_matching'] and cfg['view_prototypes_enabled'] and restricted(cfg):
        _attach_confirmed_views(db,selected,config,policy,anchors,usage,None)
    db.commit()


def simulate(config, scope, ids, family, revision):
    from .inspection_merge_paths import _cancellable
    from .offline_evaluation import _ram, _selection
    from .trial_inputs import prepare
    from .run_cache import scope as cache_scope
    from .scratch_space import scope as scratch_scope
    from .evidence_pixels import alias_scope
    from .hybrid_queries import recover
    from .hybrid_query_observer import observe_queries
    if family not in ('view','hybrid') or len(ids)!=2 or ids[0]==ids[1]:
        raise ValueError('Select one attachment mechanism and two different identities.')
    if family=='hybrid' and not config.data['sapiensid']['enabled']:
        raise ValueError('Hybrid is disabled in the loaded YAML.')
    _fresh(config,revision)
    source,target=ids
    with (_cancellable(), scratch_scope(config), cache_scope(config),
          snapshot(config,source,scope=scope,other_identifiers=[target]) as seed, closing(_ram(config)) as db):
        _fresh(config,revision)
        all_rows=sources(seed,config,scope,ids)
        rows={k:r for k,r in all_rows.items() if r['source']==source and r['kind']=='face'}
        if not rows:
            raise ValueError('The source has no current face queries. Inspect the body-track timeline.')
        _,scans=_selection(seed,config,{r['scene'] for r in rows.values()},scope)
        selected=[dict(scene=s['scene'],scan=s['id']) for s in scans]
        prepare(db,seed)
        reasons={}
        class Observer:
            def begin(self,db,banks,settings,births):
                return lambda row,face,body: None
            def decision(self,key,stage,reason,details):
                if key in rows:
                    reasons[key]=reason
        with alias_scope(db,scope,namespace(config)):
            if family=='view':
                view_queries(db,config,selected,source,list(rows),scope)
            else:
                with observe_queries(Observer()):
                    recover(SimpleNamespace(connection=db),selected,config,births=(source,))
        result=[]
        policy=matching_policy(config)
        for key,row in rows.items():
            check_cancelled()
            member=db.execute('SELECT person FROM members WHERE observation=?',(key,)).fetchone()
            owner=member[0] if member else None
            decision=db.execute(f"SELECT payload FROM candidates c WHERE {observation_sql('c')}=? AND policy=? "
                                'ORDER BY (person IS ?) DESC,rowid DESC LIMIT 1',(key,policy,owner)).fetchone()
            proof=json.loads(decision[0]) if decision else {}
            reason=proof.get('auxiliary_view_probe') if owner is None else None
            reason=reasons.get(key) or reason or proof.get('reason') or 'no current decision'
            signal=(proof.get('hybrid_query') or {}).get('signal')
            state='target' if owner==target else 'source' if owner==source else 'unassigned' if owner is None else 'other identity'
            result.append(dict(row,state=state,owner=owner,reason=reason,signal=signal))
        result.sort(key=lambda r:(int(r['scene']),r['cue'],r['id']))
        _fresh(config,revision)
    from .inspection_advice import _name
    codes = {r['reason'] for r in result if r['state']!='target'}
    keys = []
    if family=='view':
        if any('support' in code for code in codes):
            keys.append('clustering.view_attach_min_support')
        if any('cohesion' in code or 'threshold' in code for code in codes):
            keys.extend(('clustering.view_cohesion_threshold','clustering.view_match_threshold'))
        if any('prototype' in code for code in codes):
            keys.append('clustering.view_min_support')
    elif any('support' in code for code in codes):
        keys.append('sapiensid.matching.min_support')
    advice = ('Review '+', '.join(_name(config,key) for key in keys)+'.' if keys else '')
    return dict(rows=result, advice=advice, counts=dict(Counter(r['state'] for r in result)),
                bodies=sum(r['source']==source and r['kind']=='body' for r in all_rows.values()))


def timeline(reader,scan,ids):
    from .body_identity import available,policy
    from .body_observations import contract
    from .inspection_runtime import open_readonly
    with closing(open_readonly(reader.config.database)) as db:
        if not available(db):
            return []
        rows=list(db.execute("SELECT b.id,b.cue,b.face,b.quality,m.person,m.status,m.proof,f.person AS face_owner "
            "FROM body_observations b LEFT JOIN body_members m ON m.observation=b.id AND m.policy=? "
            "LEFT JOIN members f ON f.observation=b.face WHERE b.scan=? AND b.contract=? ORDER BY b.cue,b.id",
            (policy(reader.config),scan['id'],contract(reader.config))))
        cuts=db.execute('SELECT value FROM metadata WHERE key=?',('cue-view-shots:'+scan['id'],)).fetchone()
        starts={a for a,b in json.loads(cuts[0]) if b is not None} if cuts else set()
    index={r['id']:r for r in rows}
    result=[]
    for row in rows:
        p=json.loads(row['proof'] or '{}')
        q=json.loads(row['quality'] or '{}').get('quality',{})
        blocker=p.get('track_blocker') or {}
        root=p.get('open_face_root') or {}
        refs=[k for k in (root.get('observation'),p.get('local_face_root'),p.get('root'),p.get('gap_predecessor'),
                          root.get('returning')) if isinstance(k,str) and k in index]
        owners=({root.get('owner'),row['face_owner']} | {index[k]['face_owner'] for k in refs})-{None}
        peer=blocker.get('peer')
        if isinstance(peer,str) and peer in index and peer not in refs:
            refs.append(peer)
        owner=row['person'] if row['status']=='accepted' else None
        related=owner in ids or bool(set(ids)&owners)
        shot=q.get('view_shot')
        cut=row['cue'] in starts or (isinstance(shot,int) and shot==row['cue'])
        loss=len(owners)==1 and bool(owners.intersection(ids)) and owner not in owners
        # An unassigned body in a shared cue alone cannot establish a broken track.
        labels=[s for s in ('Shot boundary' if cut else None,
                ('Face '+('A' if row['face_owner']==ids[0] else 'B')) if row['face_owner'] in ids else None,
                p.get('perspective') or q.get('orientation'),p.get('reason'),blocker.get('reason')) if s]
        result.append(dict(id=row['id'],scene=scan['scene'],cue=row['cue'],kind='body',face=row['face'],owner=owner,
                           related=related,loss=loss,blocked=bool(blocker),refs=refs,reason=' · '.join(labels) or 'No recorded track decision'))
    return result
