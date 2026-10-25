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


def _recover(database, selected, config):
    if not selected:return 0
    db=database.connection
    scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    identities=catalogue(db,config,scope,namespace)
    settings=config.data['sapiensid']['matching']
    groups=[]
    for identity in identities:
        faces=list(identity.faces)
        if identity.guard_faces is not None:faces+=list(identity.guard_faces)
        groups.append(Group(identity.person,faces,True))
    policy=matching_policy(config);claims=[];votes=defaultdict(Votes)
    rows=db.execute('SELECT o.*,e.embedding AS body,c.id AS candidate FROM candidates c '
        'JOIN observations o ON o.id='+observation_sql('c')+
        ' JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
        "WHERE c.policy=? AND c.status='withheld' AND o.rejected IS NULL AND e.status='ready' "
        'AND o.scan IN (SELECT value FROM json_each(?)) AND '+query_sql('o',config.data['clustering'])+
        ' ORDER BY o.scan,o.cue,o.id',(contract(config),policy,encoded([s['scan'] for s in selected])))
    used=0
    for row in rows:
        face=vector(row);body=np.frombuffer(row['body'],dtype='<f4')
        if body.shape!=(4096,) or not np.isfinite(body).all() or abs(float(np.linalg.norm(body))-1)>.001:
            raise ValueError('Invalid retained SapiensID query feature.')
        choices=[]
        for identity,group in zip(identities,groups):
            f=scores(np.asarray([face]),np.asarray(group.samples))
            if f is None or f['minimum']<settings['face_floor']:continue
            banks=[identity.persons]
            if identity.guard_persons is not None:banks.append(identity.guard_persons)
            measures=[scores(np.asarray([body]),bank) for bank in banks if len(bank)]
            measures=[m for m in measures if m is not None]
            score=max((m['mean'] for m in measures),default=1.)
            admissible=bool(measures and not identity.guard_missing and
                len(identity.independent)>=settings['min_support'] and
                any(m['minimum']>=settings['person_threshold'] for m in measures))
            choices.append((score,identity.person,admissible,group))
        choices.sort(key=lambda x:(-x[0],x[1]))
        if not choices or not choices[0][2]:continue
        score,owner,_,group=choices[0]
        if len(choices)>1 and score-choices[1][0]<settings['ambiguity_margin']:continue
        if not co_presence_allows(db,row,group,groups,face,config,
                peer_threshold=settings['face_floor'],cohesion_threshold=settings['face_floor']):continue
        proof=dict(format=1,route='hybrid_query',frozen=True,contract=contract(config),
                   score=score,rival_score=choices[1][0] if len(choices)>1 else None,
                   reference_role='attachment',reference_selected=False)
        used+=len(encoded(proof))+len(row['quality'])+512
        if used>config.data['sapiensid']['execution']['plan_bytes']:
            raise ValueError('SapiensID query plan_bytes budget exceeded.')
        claims.append((row['candidate'],row['id'],owner,proof));votes[owner].add(row)
    accepted=0
    with db:
        for candidate,observation,owner,proof in claims:
            if len(votes[owner])<settings['min_support']:continue
            db.execute("UPDATE candidates SET person=?,status='accepted',payload=json_set(payload,"
                "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','hybrid_query',"
                "'$.hybrid_query',json(?),'$.reference_role','attachment','$.reference_selected',json('false')) WHERE id=?",
                (owner,encoded(proof),candidate))
            db.execute('INSERT OR IGNORE INTO members VALUES (?,?)',(observation,owner));accepted+=1
    return accepted


def recover(database, selected, config):
    from .evidence_pixels import alias_scope
    with alias_scope(database.connection):
        return _recover(database, selected, config)
