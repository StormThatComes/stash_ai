"""Frozen, complete-component person consolidation; missing faces are not contradictions."""
import json
import time

from .body_identity import independent,policy,available
from .body_observations import contract
from .database import encoded
from .identity_merges import canonical
from .scan_heads import predicate as current_scan
from .progress import tracking
from .body_global_banks import banks

ROUTES=('body_global','body_scene')


def defaults():
    return dict(enabled=False,match_threshold=.90,cohesion_threshold=.85,
                ambiguity_margin=.10,min_support=3,max_references=32)


def report(db,config,scope,namespace,*,focus=None):
    """Retain candidate proofs and requested diagnostic pairs, not an N² graph."""
    result=dict(enabled=config.data['sapiensid']['global_matching']['enabled'],contract=contract(config),
                pairs=[],decisions=[],nearest_competitors=[],pair_coverage='candidates_and_requested')
    if not result['enabled'] or not available(db):return result
    from .incomplete_rivals import protect
    result=protect(db,scope,namespace,plan(db,config,banks(db,config,scope,namespace),focus=focus))
    from .body_reference_gaps import protect as protect_contracts
    return protect_contracts(db,config,scope,namespace,result)


def plan(db,config,bank,*,focus=None):
    """Plan on a frozen metadata catalogue; source evidence is only read."""
    from .body_global_search import Search,exact_pairs
    from .memory_size import Reservation,size
    settings=config.data['sapiensid']['global_matching']
    result=dict(enabled=settings['enabled'],contract=contract(config),pairs=[],decisions=[],
                nearest_competitors=[],pair_coverage='candidates_and_requested')
    focus={} if focus is None else dict(focus) if isinstance(focus,dict) else {p:0 for p in focus}
    # Solver group labels can be arbitrary; the tiled lookup uses dense integers.
    labels={label:i for i,label in enumerate(dict.fromkeys(focus.values()))}
    focus={p:labels[g] for p,g in focus.items() if p in bank}
    pairs={};edges={p:set() for p in bank};plan_bytes=size(edges)+size(focus)
    limit=config.data['sapiensid']['execution']['plan_bytes'];reservation=Reservation(limit)
    def reserve(extra=0):
        nonlocal plan_bytes
        plan_bytes+=extra
        if plan_bytes>limit:raise ValueError('Global body plan exceeds sapiensid.execution.plan_bytes.')
        reservation.set(plan_bytes)
    reserve()
    try:
        with Search(db,config,bank) as search:
            for pair in exact_pairs(search.nominations(focus),bank,config,search.evidence):
                a,b=pair['people']
                reserve(size(pair)+512)
                pairs[a,b]=pair;result['pairs'].append(pair)
                if not pair['blockers']:edges[a].add(b);edges[b].add(a)
            components=[];seen=set();membership={}
            for start in sorted(bank):
                if start in seen or not edges[start]:continue
                pending=[start];component=set()
                while pending:
                    current=pending.pop()
                    if current in component:continue
                    component.add(current);pending.extend(edges[current]-component)
                seen.update(component)
                membership.update((p,len(components)) for p in component)
                components.append(sorted(component))
            reserve(size(components)+size(membership))
            rivals,goal_rivals=search.nearest([membership,focus],pairs)
            result['nearest_competitors']=list(goal_rivals.values())
            for people in components:
                blockers=[];nearest=-1.
                if any(b not in edges[a] for a in people for b in people if a!=b):
                    blockers.append('incomplete_body_component')
                for a in people:
                    runner=rivals.get(a,{}).get('score',-1.);nearest=max(nearest,runner)
                    inside=float('inf')
                    for b in people:
                        if a==b:continue
                        key=tuple(sorted((a,b)));known=pairs.get(key)
                        score=(known['score'] if known is not None else
                            search.evidence.compare(bank[key[0]]['refs'],bank[key[1]]['refs'])[0])
                        inside=min(inside,score)
                    if inside-runner<settings['ambiguity_margin'] and 'body_ambiguity' not in blockers:
                        blockers.append('body_ambiguity')
                decision=dict(people=people,outcome='blocked' if blockers else 'merge',blockers=blockers,
                    nearest_rival=nearest,cores={p:[r['id'] for core in bank[p]['cores'] for r in core] for p in people},
                    face_witnesses={p:bank[p]['face_ids'] for p in people})
                if any(bank[p].get('provisional') for p in people):
                    decision['confirmation_witnesses']={p:[r['id'] for r in bank[p]['witnesses']] for p in people}
                reserve(size(decision));result['decisions'].append(decision)
            result['search']=dict(search.stats)
        reserve(size(result['nearest_competitors'])+size(result['search']))
        if len(encoded(result))>limit:raise ValueError('Global body proof exceeds sapiensid.execution.plan_bytes.')
        return result
    finally:
        reservation.set(0)


def _scenes(db,people):
    encoded_people=encoded(list(people))
    bodies=db.execute('SELECT DISTINCT s.scene FROM body_members m JOIN body_observations b ON b.id=m.observation '
        'JOIN scans s ON s.id=b.scan WHERE m.person IN (SELECT value FROM json_each(?)) AND '+current_scan(),(encoded_people,))
    faces=db.execute('SELECT DISTINCT s.scene FROM members m JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) AND '+current_scan(),(encoded_people,))
    scenes={r[0] for r in bodies}|{r[0] for r in faces}
    for row in db.execute('SELECT target,evidence FROM identity_merges'):
        proof=json.loads(row['evidence'])
        if proof.get('route') in ROUTES and canonical(db,row['target']) in people:scenes.update(proof.get('scenes',()))
    return scenes


def reconcile(database,selected,config):
    """Caller owns the same transaction as local assignments; no partial plan."""
    db=database.connection;selection={s['scene'] for s in selected}
    first=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    scope,namespace=first
    for row in list(db.execute('SELECT m.* FROM identity_merges m JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=?',(scope,namespace))):
        proof=json.loads(row['evidence'])
        if proof.get('route') not in ROUTES:continue
        scenes=set(proof.get('scenes',()))|_scenes(db,[row['source'],row['target']])
        if scenes and scenes<=selection:
            db.execute('DELETE FROM identity_merges WHERE source=?',(row['source'],))
        else:
            # Recreated source cores need all original scenes to revalidate an
            # old alias. Do not publish a retired source or blindly trust it.
            db.execute("UPDATE body_members SET person=NULL,status='withheld',role='auxiliary',proof=? "
                'WHERE person=? AND observation IN (SELECT id FROM body_observations WHERE scan IN (SELECT value FROM json_each(?)))',
                (encoded(dict(reason='body_alias_requires_complete_selection',source=row['source'])),row['source'],encoded([s['scan'] for s in selected])))
    result=report(db,config,scope,namespace)
    from .publication_queue import enqueue
    proof_bytes=0
    for decision in result['decisions']:
        if decision['outcome']!='merge':continue
        people=decision['people'];scenes=_scenes(db,people)
        if not scenes<=selection:
            decision.update(outcome='blocked',blockers=['identity_outside_selected_scenes'],required_scenes=sorted(scenes,key=int));continue
        rows=list(db.execute('SELECT p.id,p.created,EXISTS(SELECT 1 FROM members WHERE person=p.id) AS has_face FROM people p '
            'WHERE p.id IN (SELECT value FROM json_each(?)) ORDER BY has_face DESC,p.created,p.id',(encoded(people),)))
        target=rows[0]['id']
        provisional=bool(db.execute("SELECT 1 FROM body_members WHERE status='provisional' AND person IN (SELECT value FROM json_each(?)) LIMIT 1",(encoded(people),)).fetchone())
        proof=dict(decision,complete_body_witnesses=True,body_witness_format=2,provisional_confirmation=provisional,route='body_global',scenes=sorted(scenes,key=int),contract=contract(config),policy=policy(config))
        proof_bytes+=len(encoded(proof))*(len(rows)-1)
        if proof_bytes>config.data['sapiensid']['execution']['plan_bytes']:
            raise ValueError('Body alias proofs exceed sapiensid.execution.plan_bytes.')
        for source in (r['id'] for r in rows[1:]):
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
            db.execute("UPDATE body_members SET person=?,role='auxiliary',proof=json_set(proof,'$.global_merge',json(?)) WHERE person=?",
                       (target,encoded(dict(source=source,target=target,route='body_global')),source))
            # Original face anchors retain vetoes in the ordinary hybrid system;
            # a body-based alias cannot promote them into new face references.
            db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
            db.execute("UPDATE candidates SET person=?,payload=json_set(payload,"
                "'$.body_merge_origin',json(COALESCE(json_extract(payload,'$.body_merge_origin'),json_object("
                "'person',person,'reason',json_extract(payload,'$.reason'),'reference_role',json_extract(payload,'$.reference_role'),"
                "'reference_selected',json_extract(payload,'$.reference_selected')))),"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','body_global_identity',"
                "'$.reference_role','attachment','$.reference_selected',json('false')) WHERE person=?",(target,source))
        # Cross-scene proof can confirm individually provisional scene cores.
        db.execute("UPDATE body_members SET status='accepted' WHERE person=? AND status='provisional'",(target,))
        for scene in scenes:enqueue(db,scope,namespace,scene,config,'body_global_merge')
    # A body-born identity may become entirely associated with an existing face
    # identity within the scene. Record that promotion as an explicit alias.
    origins={r[0] for r in db.execute('SELECT DISTINCT i.person FROM body_origins i JOIN body_observations b ON b.id=i.observation '
        'JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND '+current_scan(),(scope,namespace))}
    for source in sorted(origins):
        if canonical(db,source)!=source or db.execute('SELECT 1 FROM members WHERE person=? LIMIT 1',(source,)).fetchone():continue
        rows=list(db.execute('SELECT b.id,b.scan,b.cue,b.seconds,b.quality,m.person,m.status,s.scene FROM body_origins i '
            'JOIN body_observations b ON b.id=i.observation JOIN scans s ON s.id=b.scan LEFT JOIN body_members m ON m.observation=b.id '
            'WHERE i.person=? AND b.contract=? AND '+current_scan(),(source,contract(config))))
        targets={r['person'] for r in rows}
        if not rows or len(targets)!=1 or None in targets or source in targets or any(r['status']!='accepted' for r in rows):continue
        target=next(iter(targets));scenes={r['scene'] for r in rows}
        if not scenes<=selection or not db.execute('SELECT 1 FROM members WHERE person=? LIMIT 1',(target,)).fetchone():continue
        if len(independent(rows,config.data['sapiensid']['identities']))<config.data['sapiensid']['identities']['min_support']:continue
        proof=dict(route='body_scene',scenes=sorted(scenes,key=int),cores={source:[r['id'] for r in rows]},
                   contract=contract(config),reason='body_core_confirmed_by_face_bank')
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
    return result
