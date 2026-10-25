"""Frozen, quality-qualified supplemental evidence shared by runs and reports.

Cosines from different encoders are never averaged together. One physical
scene/timestamp supplies at most one vote; attachment-only samples never train
this catalogue. Unknown evidence is not a negative match.
"""
from dataclasses import dataclass
from itertools import combinations
import json

from .person_signal import contract, available
from .database import encoded
from .progress import tracking


@dataclass(frozen=True)
class Identity:
    person: str
    observations: tuple
    faces: object
    persons: object
    independent: tuple
    missing: tuple = ()


def catalogue(db, config, scope, namespace):
    import numpy as np
    from .clustering import _groups
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    groups=_groups(db,scope,namespace,config.data['clustering'],
                   scan_policy=scan_identity(config,namespace),compatibility=fallback_sql(db,config,namespace))
    result=[];used=0;present=available(db);expected=contract(config)
    with tracking(groups,'Read hybrid references',unit='identity') as tracked:
        for group in tracked:
            ids=tuple(group.sample_ids);vectors=[];keys=[];missing=[]
            rows={r['id']:r for r in db.execute('SELECT o.id,o.seconds,s.scene FROM observations o '
                'JOIN scans s ON s.id=o.scan WHERE o.id IN (SELECT value FROM json_each(?))',(encoded(ids),))}
            for observation in ids:
                row=db.execute('SELECT status,embedding FROM person_evidence WHERE observation=? AND contract=?',
                               (observation,expected)).fetchone() if present else None
                if row is None or row['status']!='ready':
                    missing.append(observation);continue
                value=np.frombuffer(row['embedding'],dtype='<f4')
                if value.shape!=(4096,) or not np.isfinite(value).all() or abs(float(np.linalg.norm(value))-1)>.001:
                    raise ValueError('Invalid retained SapiensID feature: '+observation)
                used+=value.nbytes
                if used>config.data['sapiensid']['execution']['reference_bytes']:
                    raise ValueError('SapiensID reference_bytes budget exceeded.')
                vectors.append(value);keys.append((rows[observation]['scene'],rows[observation]['seconds']))
            result.append(Identity(group.identifier,ids,np.asarray(group.samples),
                np.asarray(vectors) if vectors else np.empty((0,4096)),tuple(sorted(set(keys))),tuple(missing)))
    return tuple(result)


def scores(left,right):
    """Exact blocked reduction: no full reference cross-product allocation."""
    import numpy as np
    if not len(left) or not len(right):return None
    low=1.;total=0.;count=0
    for start in range(0,len(left),16):
        values=left[start:start+16] @ right.T
        low=min(low,float(values.min()));total+=float(np.sum(values,dtype='float64'));count+=values.size
    return dict(minimum=low,mean=total/count)


def pair(left,right,settings,*,copresent=False,reflection_threshold=1.):
    face=scores(left.faces,right.faces);person=scores(left.persons,right.persons)
    blockers=[]
    if left.missing or right.missing or person is None:blockers.append('missing_person_evidence')
    if min(len(left.independent),len(right.independent))<settings['min_support']:
        blockers.append('insufficient_independent_support')
    if face is None or face['minimum']<settings['face_floor']:blockers.append('face_contradiction')
    if person is not None and person['minimum']<settings['person_threshold']:blockers.append('person_below_threshold')
    if copresent and (face is None or face['minimum']<reflection_threshold):blockers.append('copresence_conflict')
    return dict(format=1,left=left.person,right=right.person,face=face,person=person,
        observations=[list(left.observations),list(right.observations)],
        support=[len(left.independent),len(right.independent)],
        missing=[list(left.missing),list(right.missing)],blockers=blockers,
        thresholds=dict(settings),cooccurring=copresent)


def plan(db,config,identities,*,focus=None):
    """Complete frozen components; all outsiders remain competitors.

    Candidate nomination accepts either face or person similarity. Exact paired
    evidence and independent support decide acceptance. No output is recycled
    as input during the pass.
    """
    from .global_views import _cooccurs
    settings=config.data['sapiensid']['matching'];face_threshold=config.data['recognition']['match_threshold']
    owned={x.person:x for x in identities};edges={};proofs={};rivals={p:[] for p in owned}
    with tracking(combinations(sorted(owned),2),'Compare hybrid identities',unit='pair') as pairs:
        for a,b in pairs:
            left,right=owned[a],owned[b]
            face=scores(left.faces,right.faces);person=scores(left.persons,right.persons)
            # Weak rivals still participate even if they cannot support a merge.
            if person is not None:
                rivals[a].append((b,person['mean']));rivals[b].append((a,person['mean']))
            elif face is not None and face['mean']>=face_threshold:
                rivals[a].append((b,1.));rivals[b].append((a,1.))
            if ((face is None or face['mean']<settings['face_floor']) and
                    (person is None or person['mean']<settings['person_threshold']-settings['ambiguity_margin'])):
                continue
            proof=pair(left,right,settings,copresent=_cooccurs(db,a,b,config),
                       reflection_threshold=config.data['recognition']['cooccurrence_merge_threshold'])
            proofs[a,b]=proof
            if not proof['blockers']:edges[a,b]=proof['person']['mean']
    adjacency={p:set() for p in owned}
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    visited=set();decisions=[]
    for seed in sorted(owned):
        if seed in visited:continue
        pending=[seed];component=set()
        while pending:
            p=pending.pop()
            if p in component:continue
            component.add(p);pending.extend(adjacency[p]-component)
        visited.update(component)
        if focus is not None and not component.intersection(focus):continue
        people=sorted(component);blockers=[];nearest=None
        if len(people)<2:blockers.append('no_supported_pair')
        elif any(tuple(sorted((a,b))) not in edges for a,b in combinations(people,2)):
            blockers.append('incomplete_component')
        else:
            for a in people:
                internal=min(edges[tuple(sorted((a,b)))] for b in people if b!=a)
                for rival,score in rivals[a]:
                    if rival in component:continue
                    gap=internal-score
                    entry=dict(identity=a,rival=rival,score=score,gap=gap)
                    if nearest is None or gap<nearest['gap']:nearest=entry
            if nearest is not None and nearest['gap']<settings['ambiguity_margin']:
                blockers.append('competing_identity')
        decisions.append(dict(people=people,outcome='blocked' if blockers else 'merge',blockers=blockers,
            nearest_rival=nearest,frozen=True,contract=contract(config),
            proofs=[proofs[a,b] for a,b in combinations(people,2) if (a,b) in proofs]))
    return dict(format=1,kind='hybrid_identity_plan',decisions=decisions,
                pairs=[v for (a,b),v in proofs.items() if focus is None or a in focus or b in focus],
                identities=len(owned),enabled=config.data['sapiensid']['enabled'])


def report(db,config,scope,namespace,*,focus=None):
    if not config.data['sapiensid']['enabled']:
        return dict(enabled=False)
    return plan(db,config,catalogue(db,config,scope,namespace),focus=focus)
