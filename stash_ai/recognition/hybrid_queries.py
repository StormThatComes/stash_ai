from .shared_person_evidence import expression
body_embedding = expression('embedding')
"""Frozen hybrid query recovery, including original merged appearances.

Recovered rows are attachments only. All current strict identities compete,
including identities lacking person evidence. Original cores never acquire new
merge votes or train references through this route.
"""
from .resource_errors import ResourceLimit
from collections import defaultdict
import numpy as np
from .database import encoded
from .hybrid_evidence import catalogue, scores
from .person_signal import contract
from .clustering import Group, co_presence_allows, vector, matching_policy
from .independent_evidence import Votes
from .pose_policy import query_sql
from .reference_roles import observation_sql


def _recover(database, selected, config, claims, *, births=()):
    if not selected:return 0
    db=database.connection
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    identities=catalogue(db,config,scope,namespace)
    settings=config.data['sapiensid']['matching']
    banks=[];core_modes={}
    for identity in identities:
        cores=[(identity.observations,identity.faces,identity.persons)]+list(identity.query_cores)
        for ids,faces,bodies in cores:
            if len(faces):banks.append((identity,Group(identity.person,list(faces),True,ids),bodies,faces))
        core_modes.update((id(core[2]),modes) for core,modes in zip(identity.query_cores,identity.query_core_modes))
    groups=[group for _,group,_,_ in banks]
    from .hybrid_query_index import QueryIndex
    from .hybrid_query_observer import query_observer, query_decision
    observe=query_observer(db,banks,config,births,selected)
    policy=matching_policy(config)
    rows=db.execute(f'SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,{body_embedding} AS body,e.metadata AS person_metadata,c.id AS candidate,c.person AS prior_owner FROM candidates c '
        'JOIN observations o ON o.id='+observation_sql('c')+
        " LEFT JOIN person_evidence e ON e.observation=o.id AND e.contract=? AND e.status='ready' "
        "WHERE c.policy=? AND (c.status='withheld' OR c.person IN (SELECT value FROM json_each(?))) AND o.rejected IS NULL "
        'AND o.scan IN (SELECT value FROM json_each(?)) AND '+query_sql('o',config.data['clustering'])+
        ' ORDER BY o.scan,o.cue,o.id',(contract(config),policy,encoded(list(births)),encoded([s['scan'] for s in selected])))
    source_counts=dict(db.execute('SELECT person,COUNT(*) FROM members WHERE person IN '
        '(SELECT value FROM json_each(?)) GROUP BY person',(encoded(list(births)),)))
    def queries():
        from .pose_policy import is_anchor
        for row in rows:
            if config.data['clustering']['auxiliary_matching'] or is_anchor(row['quality'],config.data['clustering']):
                yield row,vector(row)
    from contextlib import closing
    with closing(QueryIndex(banks,config.data['sapiensid']['execution']['reference_bytes'])) as index:
        for row,face,candidates in index.batch(queries(),settings['face_floor']):
            body=np.frombuffer(row['body'],dtype='<f4') if row['body'] is not None else None
            if body is not None and (body.shape!=(4096,) or not np.isfinite(body).all() or abs(float(np.linalg.norm(body))-1)>.001):
                raise ValueError('Invalid retained SapiensID query feature.')
            choices=[];face_choices=[];query_modes={};body_candidates=[];minimums={}
            face_batch=np.asarray([face],dtype='float64')
            body_batch=np.asarray([body],dtype='float64') if body is not None else None
            if observe is not None:
                observe(row,face_batch,body_batch)
            for identity,group,bodies,faces in candidates:
                if identity.person in births:continue
                f=scores(face_batch,faces)
                if f is None or f['minimum']<settings['face_floor']:continue
                admissible=(not identity.guard_face_missing and f['minimum']>=config.data['clustering']['cohesion_threshold'] and f['mean']>=config.data['recognition']['match_threshold'])
                minimums['face', identity.person, id(group)] = f['minimum']
                face_choices.append((f['mean'],identity.person,admissible,group))
                if body is not None:body_candidates.append((identity,group,bodies))
            # Prefer a strict face proof; body scores never compete with face scores.
            winner=choose(face_choices,config.data['recognition']['ambiguity_margin'])
            signal='face'
            if winner is None:
                for identity,group,bodies in body_candidates:
                    from .hybrid_modes import query_score
                    modes=identity.body_modes if bodies is identity.persons else core_modes.get(id(bodies),())
                    measures=[query_score(body_batch,bodies,settings,modes)] if len(bodies) else []
                    measures=[m for m in measures if m is not None]
                    if measures:
                        query_modes[identity.person,id(group)]=measures[0]['mode_index']
                        minimums['face_and_body', identity.person, id(group)] = measures[0]['minimum']
                    score=max((m['mean'] for m in measures),default=None)
                    admissible=bool(measures and not identity.guard_missing and (not identity.unsupported_body_modes or
                        (bodies is identity.persons and bool(modes))) and
                        len(identity.independent)>=settings['min_support'] and
                        any(m['minimum']>=settings['person_threshold'] for m in measures))
                    choices.append((score,identity.person,admissible,group))
                    if identity.unsupported_body_modes:
                        # Pending appearances have no certified upper score for another owner.
                        choices.append((None,identity.person,False,group))
                winner=choose(choices,settings['ambiguity_margin']);signal='face_and_body'
            if winner is None:
                if observe is not None:query_decision(row['id'],'choice','no_unambiguous_query_winner',
                    face_choices=len(face_choices),body_choices=len(choices))
                continue
            score,owner,_,group,rival_score=winner
            if not co_presence_allows(db,row,group,groups,face,config,
                    peer_threshold=settings['face_floor'],cohesion_threshold=settings['face_floor']):
                if observe is not None:query_decision(row['id'],'copresence','copresence_conflict',owner=owner)
                continue
            competing = [(value, peer) for value,peer,_,_ in (face_choices if signal=='face' else choices)
                         if peer!=owner and value is not None and value==rival_score]
            proof=dict(format=1,route='hybrid_query',frozen=True,contract=contract(config),
                       core_observations=list(group.sample_ids),signal=signal,score=score,rival_score=rival_score,
                       reference_role='attachment',reference_selected=False,
                       body_mode=query_modes.get((owner,id(group))) if signal=='face_and_body' else None,
                       minimum=minimums.get((signal,owner,id(group))),
                       match_threshold=config.data['recognition']['match_threshold'] if signal=='face' else None,
                       cohesion_threshold=config.data['clustering']['cohesion_threshold'] if signal=='face' else settings['person_threshold'],
                       ambiguity_margin=config.data['recognition']['ambiguity_margin'] if signal=='face' else settings['ambiguity_margin'],
                       rival=min(peer for _,peer in competing) if competing else None, independent_query_support=None)
            from .independent_evidence import body_vote
            vote=(body_vote(row) if signal=='face_and_body' else
                  {k:row[k] for k in ('scan','seconds','quality')})
            claim=(row['candidate'],row['id'],owner,proof,row['prior_owner'],vote)
            family=((owner,'face') if signal=='face' else
                    (owner,'face_and_body',tuple(proof['core_observations']),proof['body_mode']))
            claims.add(row,claim,family)
    # Resolve against every new claim and existing accepted face in each cue,
    # even when the winning queries used different original merged cores.
    from .decision_resolution import conflicting_claims
    for scan,cue in claims.db.execute('SELECT scan,cue FROM cues ORDER BY scan,cue'):
        for row in db.execute('SELECT o.id,o.scan,o.cue,o.geometry,o.dimensions,o.embedding,m.person '
                'FROM observations o JOIN members m ON m.observation=o.id '
                'WHERE o.scan=? AND o.cue=? AND o.rejected IS NULL',(scan,cue)):
            claims.conflict(row)
    threshold=max(config.data['recognition']['match_threshold'],config.data['recognition']['cooccurrence_merge_threshold'])
    for row in conflicting_claims(claims.db.execute('SELECT * FROM conflicts ORDER BY person,scan,cue,id'),
            threshold,deduplicate=config.data['clustering'].get('deduplicate_detections',False)):
        claims.db.execute('DELETE FROM claims WHERE observation=?',(row['id'],))
    accepted=0
    def family(claim):
        _,_,owner,proof,_,_=claim
        return ((owner,'face') if proof['signal']=='face' else
                (owner,'face_and_body',tuple(proof['core_observations']),proof['body_mode']))
    def minimum(claim):
        return config.data['clustering']['auxiliary_min_support'] if claim[3]['signal']=='face' else settings['min_support']
    complete=claims.resolve(source_counts,minimum)
    with db:
        for candidate,observation,owner,proof,prior,row in claims.rows():
            if prior in births and prior not in complete:
                if observe is not None:query_decision(observation,'birth','incomplete_birth_claim',owner=owner)
                continue
            support=claims.support(family((candidate,observation,owner,proof,prior,row)))
            if not claims.accepted(observation):
                if observe is not None:query_decision(observation,'support',
                    'auxiliary_insufficient_support' if proof['signal']=='face' else 'query_support',
                    owner=owner,support=support)
                continue
            proof['independent_query_support']=support
            db.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','hybrid_query',"
                "'$.hybrid_query',json(?),'$.reference_role','attachment','$.reference_selected',json('false')) WHERE id=?",
                (owner,encoded(proof),candidate))
            if prior in births:db.execute('DELETE FROM members WHERE observation=?',(observation,))
            db.execute('INSERT OR IGNORE INTO members VALUES (?,?)',(observation,owner));accepted+=1
    return accepted


def recover(database, selected, config, *, births=()):
    from .evidence_pixels import alias_scope
    if not selected:return 0
    context=database.connection.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    if context is None:raise ValueError('Missing selected recognition scan.')
    from .query_claims import Claims
    with alias_scope(database.connection,*context,budget=config.data['clustering']['max_reference_bytes']), Claims(config) as claims:
        return _recover(database, selected, config, claims, births=births)


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
