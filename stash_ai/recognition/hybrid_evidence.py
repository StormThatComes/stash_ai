"""Frozen, quality-qualified supplemental evidence shared by runs and reports.

Cosines from different encoders are never averaged together. One physical
scene/timestamp supplies at most one vote; attachment-only samples never train
this catalogue. Unknown evidence is not a negative match.
"""
from dataclasses import dataclass
from itertools import combinations

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
    guard_observations: tuple = ()
    guard_missing: tuple = ()
    guard_faces: object = None
    guard_persons: object = None


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
            ids=tuple(group.sample_ids);vectors=[];missing=[]
            from .independent_evidence import Votes
            votes=Votes()
            if present:
                query=('SELECT o.id,o.scan,o.quality,o.seconds,s.scene,e.status,e.embedding FROM observations o '
                    'JOIN scans s ON s.id=o.scan LEFT JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
                    'WHERE o.id IN (SELECT value FROM json_each(?))')
                parameters=(expected,encoded(ids))
            else:
                query=('SELECT o.id,o.scan,o.quality,o.seconds,s.scene,NULL AS status,NULL AS embedding FROM observations o '
                    'JOIN scans s ON s.id=o.scan WHERE o.id IN (SELECT value FROM json_each(?))')
                parameters=(encoded(ids),)
            rows={r['id']:r for r in db.execute(query,parameters)}
            for observation in ids:
                row=rows.get(observation)
                if row is None or row['status']!='ready':
                    missing.append(observation);continue
                value=np.frombuffer(row['embedding'],dtype='<f4')
                if value.shape!=(4096,) or not np.isfinite(value).all() or abs(float(np.linalg.norm(value))-1)>.001:
                    raise ValueError('Invalid retained SapiensID feature: '+observation)
                used+=value.nbytes
                if used>config.data['sapiensid']['execution']['reference_bytes']:
                    raise ValueError('SapiensID reference_bytes budget exceeded.')
                vectors.append(value);votes.add(row)
            result.append(Identity(group.identifier,ids,np.asarray(group.samples),
                np.asarray(vectors) if vectors else np.empty((0,4096)),tuple(range(len(votes))),tuple(missing)))
    from .hybrid_guards import retain
    return retain(db,config,scope,namespace,tuple(result))


def scores(left,right):
    """Exact blocked reduction: no full reference cross-product allocation."""
    import numpy as np
    if not len(left) or not len(right):return None
    low=1.;total=0.;count=0
    for start in range(0,len(left),16):
        values=left[start:start+16].astype('float64') @ right.astype('float64').T
        low=min(low,float(values.min()));total+=float(np.sum(values,dtype='float64'));count+=values.size
    return dict(minimum=low,mean=total/count)


def pair(left,right,settings,*,copresent=False,reflection_threshold=1.):
    face=scores(left.faces,right.faces);person=scores(left.persons,right.persons)
    blockers=[]
    # Former source cores are vetoes only: no extra votes or nominations.
    for kind,measurement in (('faces',face),('persons',person)):
        a,b=getattr(left,'guard_'+kind),getattr(right,'guard_'+kind)
        for first,second in ((a,getattr(right,kind)),(getattr(left,kind),b),(a,b)):
            if first is None or second is None:continue
            guard=scores(first,second)
            if guard is not None and measurement is not None:
                measurement['minimum']=min(measurement['minimum'],guard['minimum'])
    if left.guard_missing or right.guard_missing:blockers.append('missing_original_core_evidence')
    if left.missing or right.missing or person is None:blockers.append('missing_person_evidence')
    if min(len(left.independent),len(right.independent))<settings['min_support']:
        blockers.append('insufficient_independent_support')
    if face is None or face['minimum']<settings['face_floor']:blockers.append('face_contradiction')
    if person is not None and person['minimum']<settings['person_threshold']:blockers.append('person_below_threshold')
    if copresent and (face is None or face['minimum']<reflection_threshold):blockers.append('copresence_conflict')
    return dict(format=1,left=left.person,right=right.person,face=face,person=person,
        observations=[list(left.observations),list(right.observations)],
        guard_observations=[list(left.guard_observations),list(right.guard_observations)],
        missing_original_cores=[list(left.guard_missing),list(right.guard_missing)],
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
    from .hybrid_lookup import Lookup
    owned={x.person:x for x in identities};edges={};proofs={}
    limits=config.data['sapiensid']['execution']
    lookup=Lookup([owned[p] for p in sorted(owned)],limits['reference_bytes'])
    candidates=used=0
    with tracking(lookup.pairs(),'Nominate hybrid pairs',unit='pair',total=len(owned)*(len(owned)-1)//2) as pairs:
        for a,b,face_mean,person_mean in pairs:
            # A tiny overnomination tolerance protects threshold-edge witnesses
            # from centroid reduction rounding. Exact witness gates decide.
            if ((face_mean is None or face_mean+1e-6<settings['face_floor']) and
                    (person_mean is None or person_mean+1e-6<settings['person_threshold']-settings['ambiguity_margin'])):
                continue
            candidates+=1
            if candidates>limits['max_candidate_pairs']:
                raise ValueError('SapiensID max_candidate_pairs budget exceeded; no hybrid plan applied.')
            proof=pair(owned[a],owned[b],settings,copresent=_cooccurs(db,a,b,config),
                       reflection_threshold=config.data['recognition']['cooccurrence_merge_threshold'])
            used+=len(encoded(proof).encode())
            if used>limits['plan_bytes']:
                raise ValueError('SapiensID plan_bytes budget exceeded; no hybrid plan applied.')
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
        decisions.append(dict(people=people,outcome='blocked' if blockers else 'merge',blockers=blockers,
            nearest_rival=nearest,frozen=True,contract=contract(config),
            proofs=[proofs[a,b] for a,b in combinations(people,2) if (a,b) in proofs]))
    # Second bounded pass: weak/missing-evidence outsiders still compete, but
    # their N-squared scores are never retained in Python dictionaries.
    eligible={p:d for d in decisions if not d['blockers'] for p in d['people']}
    internal={p:min(edges[tuple(sorted((p,q)))] for q in d['people'] if q!=p)
              for p,d in eligible.items()}
    with tracking(lookup.pairs(),'Check hybrid competitors',unit='pair',total=len(owned)*(len(owned)-1)//2) as comparisons:
        for a,b,face_mean,person_mean in comparisons:
            score=person_mean
            if score is None and face_mean is not None and face_mean>=face_threshold:score=1.
            if score is None:continue
            for person,rival in ((a,b),(b,a)):
                decision=eligible.get(person)
                if decision is None or rival in decision['people']:continue
                gap=internal[person]-score
                entry=dict(identity=person,rival=rival,score=score,gap=gap)
                nearest=decision['nearest_rival']
                if nearest is None or (gap,person,rival)<(nearest['gap'],nearest['identity'],nearest['rival']):
                    decision['nearest_rival']=entry
    for decision in decisions:
        nearest=decision['nearest_rival']
        if nearest is not None and nearest['gap']<settings['ambiguity_margin']:
            decision.update(outcome='blocked',blockers=['competing_identity'])
    return dict(format=1,kind='hybrid_identity_plan',decisions=decisions,
                candidate_pairs=candidates,proof_bytes=used,comparison_block=64,
                pairs=[v for (a,b),v in proofs.items() if focus is None or a in focus or b in focus],
                identities=len(owned),enabled=config.data['sapiensid']['enabled'])


def report(db,config,scope,namespace,*,focus=None):
    if not config.data['sapiensid']['enabled']:
        return dict(enabled=False)
    return plan(db,config,catalogue(db,config,scope,namespace),focus=focus)
