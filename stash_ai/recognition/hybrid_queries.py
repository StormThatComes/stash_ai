"""Frozen hybrid query recovery, including original merged appearances.

Recovered rows are attachments only. All current strict identities compete,
including identities lacking person evidence. Original cores never acquire new
merge votes or train references through this route.
"""
from collections import defaultdict
import numpy as np
from .database import encoded
from .hybrid_evidence import catalogue, scores
from .person_signal import contract
from .clustering import Group, co_presence_allows, vector, matching_policy
from .independent_evidence import Votes
from .pose_policy import query_sql
from .reference_roles import observation_sql


def _recover(database, selected, config, *, births=()):
    if not selected:return 0
    db=database.connection
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    identities=catalogue(db,config,scope,namespace)
    settings=config.data['sapiensid']['matching']
    banks=[]
    for identity in identities:
        cores=[(identity.observations,identity.faces,identity.persons)]+list(identity.query_cores)
        for ids,faces,bodies in cores:
            if len(faces):banks.append((identity,Group(identity.person,list(faces),True,ids),bodies,faces))
    groups=[group for _,group,_,_ in banks]
    from .hybrid_query_index import QueryIndex
    index=QueryIndex(banks,config.data['sapiensid']['execution']['reference_bytes'])
    policy=matching_policy(config);claims=[];votes=defaultdict(Votes)
    rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,e.embedding AS body,c.id AS candidate,c.person AS prior_owner FROM candidates c '
        'JOIN observations o ON o.id='+observation_sql('c')+
        " LEFT JOIN person_evidence e ON e.observation=o.id AND e.contract=? AND e.status='ready' "
        "WHERE c.policy=? AND (c.status='withheld' OR c.person IN (SELECT value FROM json_each(?))) AND o.rejected IS NULL "
        'AND o.scan IN (SELECT value FROM json_each(?)) AND '+query_sql('o',config.data['clustering'])+
        ' ORDER BY o.scan,o.cue,o.id',(contract(config),policy,encoded(list(births)),encoded([s['scan'] for s in selected])))
    used=0
    source_counts=dict(db.execute('SELECT person,COUNT(*) FROM members WHERE person IN '
        '(SELECT value FROM json_each(?)) GROUP BY person',(encoded(list(births)),)))
    for row in rows:
        face=vector(row);body=np.frombuffer(row['body'],dtype='<f4') if row['body'] is not None else None
        if body is not None and (body.shape!=(4096,) or not np.isfinite(body).all() or abs(float(np.linalg.norm(body))-1)>.001):
            raise ValueError('Invalid retained SapiensID query feature.')
        choices=[];face_choices=[]
        face_batch=np.asarray([face],dtype='float64')
        body_batch=np.asarray([body],dtype='float64') if body is not None else None
        for identity,group,bodies,faces in index.candidates(face,settings['face_floor']):
            if identity.person in births:continue
            f=scores(face_batch,faces)
            if f is None or f['minimum']<settings['face_floor']:continue
            admissible=(not identity.guard_face_missing and f['minimum']>=config.data['clustering']['cohesion_threshold'] and f['mean']>=config.data['recognition']['match_threshold'])
            face_choices.append((f['mean'],identity.person,admissible,group))
            if body is None:continue
            measures=[scores(body_batch,bodies)] if len(bodies) else []
            measures=[m for m in measures if m is not None]
            score=max((m['mean'] for m in measures),default=None)
            admissible=bool(measures and not identity.guard_missing and
                len(identity.independent)>=settings['min_support'] and
                any(m['minimum']>=settings['person_threshold'] for m in measures))
            choices.append((score,identity.person,admissible,group))
        # Prefer a strict face proof; body scores never compete with face scores.
        winner=choose(face_choices,config.data['recognition']['ambiguity_margin'])
        signal='face'
        if winner is None:
            winner=choose(choices,settings['ambiguity_margin']);signal='face_and_body'
        if winner is None:continue
        score,owner,_,group,rival_score=winner
        if not co_presence_allows(db,row,group,groups,face,config,
                peer_threshold=settings['face_floor'],cohesion_threshold=settings['face_floor']):continue
        proof=dict(format=1,route='hybrid_query',frozen=True,contract=contract(config),
                   core_observations=list(group.sample_ids),signal=signal,score=score,rival_score=rival_score,
                   reference_role='attachment',reference_selected=False)
        vote={k:row[k] for k in ('scan','seconds','quality')}
        used+=2*(len(encoded(proof).encode())+len(encoded(vote).encode()))+2048
        if used>config.data['sapiensid']['execution']['plan_bytes']:
            raise ValueError('SapiensID query plan_bytes budget exceeded.')
        claims.append((row['candidate'],row['id'],owner,proof,row['prior_owner'],vote))
    accepted=0
    birth_claims=defaultdict(list)
    for claim in claims:
        if claim[4] in births:birth_claims[claim[4]].append(claim)
    admissible_births={source for source,items in birth_claims.items()
        if len(items)==source_counts.get(source,0) and len({c[2] for c in items})==1}
    for candidate,observation,owner,proof,prior,row in claims:
        if prior not in births or prior in admissible_births:votes[owner].add(row)
    with db:
        for candidate,observation,owner,proof,prior,row in claims:
            if prior in births and prior not in admissible_births:continue
            if len(votes[owner])<(config.data['clustering']['auxiliary_min_support'] if proof['signal']=='face' else settings['min_support']):continue
            db.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','hybrid_query',"
                "'$.hybrid_query',json(?),'$.reference_role','attachment','$.reference_selected',json('false')) WHERE id=?",
                (owner,encoded(proof),candidate))
            if prior in births:db.execute('DELETE FROM members WHERE observation=?',(observation,))
            db.execute('INSERT OR IGNORE INTO members VALUES (?,?)',(observation,owner));accepted+=1
    return accepted


def recover(database, selected, config, *, births=()):
    from .evidence_pixels import alias_scope
    with alias_scope(database.connection):
        return _recover(database, selected, config, births=births)


def choose(choices,margin):
    """Keep unknown rivals explicit and never let a rejected own core hide a valid one."""
    owners={}
    for choice in choices:
        owners.setdefault(choice[1],[]).append(choice)
    valid=[max((c for c in values if c[2] and c[0] is not None),key=lambda c:c[0],default=None)
           for values in owners.values()]
    valid=[c for c in valid if c is not None]
    if not valid:return None
    best=max(valid,key=lambda c:(c[0],c[1]))
    rivals=[c for owner,values in owners.items() if owner!=best[1] for c in values]
    if any(c[0] is None for c in rivals):return None
    rival=max((c[0] for c in rivals),default=None)
    if rival is not None and best[0]-rival<margin:return None
    return (*best,rival)
