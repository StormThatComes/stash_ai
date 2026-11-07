"""Frozen, quality-qualified supplemental evidence shared by runs and reports.

Cosines from different encoders are never averaged together. One physical
scene/timestamp supplies at most one vote; attachment-only samples never train
this catalogue. Unknown evidence is not a negative match.
"""
from .resource_errors import ResourceLimit
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
    person_observations: tuple = ()
    guard_person_observations: tuple = ()
    guard_face_missing: tuple = ()
    query_cores: tuple = ()
    query_core_modes: tuple = ()
    body_modes: tuple = ()
    body_witness_observations: tuple = ()
    unsupported_body_modes: bool = False


def _catalogue(db, config, scope, namespace):
    import numpy as np
    from .clustering import _groups
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    groups=_groups(db,scope,namespace,config.data['clustering'],
                   scan_policy=scan_identity(config,namespace),compatibility=fallback_sql(db,config,namespace))
    groups=list(groups)
    result=[];used=0;present=available(db);expected=contract(config)
    from .hybrid_references import catalogue_rows
    body_catalogue=catalogue_rows(db,config,scope,namespace,[g.identifier for g in groups]) if present else {}
    ready={r[0] for r in db.execute("SELECT e.observation FROM person_evidence e JOIN observations o ON o.id=e.observation "
        "JOIN scans s ON s.id=o.scan WHERE e.contract=? AND e.status='ready' AND s.scope=? AND s.namespace=?",
        (expected,scope,namespace))} if present else set()
    with tracking(groups,'Read hybrid references',unit='identity') as tracked:
        for group in tracked:
            ids=tuple(group.sample_ids)
            from .hybrid_modes import body_modes
            records=body_catalogue.get(group.identifier,[])
            from .independent_evidence import Votes
            votes=Votes(); vectors=[]; body_ids=[]
            for row in records:
                value=np.frombuffer(row['embedding'],dtype='<f4')
                if value.shape!=(4096,) or not np.isfinite(value).all() or abs(float(np.linalg.norm(value))-1)>.001:
                    raise ValueError('Invalid retained SapiensID feature: '+row['id'])
                used+=value.nbytes
                if used>config.data['sapiensid']['execution']['reference_bytes']:
                    raise ResourceLimit('SapiensID reference_bytes budget exceeded.')
                vectors.append(value);votes.add(row);body_ids.append(row['id'])
            missing=tuple(i for i in ids if i not in ready)
            result.append(Identity(group.identifier,ids,np.asarray(group.samples),
                np.asarray(vectors) if vectors else np.empty((0,4096)),tuple(range(getattr(records,'support',len(votes)))),
                missing,person_observations=tuple(body_ids),
                body_modes=records.modes if hasattr(records,'modes') else body_modes(records,config),
                body_witness_observations=getattr(records,'witnesses',tuple(body_ids)),
                unsupported_body_modes=getattr(records,'unsupported_modes',False)))
    from .hybrid_guards import retain
    return retain(db,config,scope,namespace,tuple(result))


def scores(left,right):
    """Exact blocked reduction: no full reference cross-product allocation."""
    import numpy as np
    if not len(left) or not len(right):return None
    low=1.;total=0.;count=0
    from .disk_body_modes import Matrix
    if isinstance(left,Matrix) or isinstance(right,Matrix):
        for start in range(0,len(left),16):
            a=np.asarray(left[start:start+16],dtype='float64')
            wa=left.weights(slice(start,start+16)) if isinstance(left,Matrix) else np.ones(len(a))
            for other in range(0,len(right),64):
                b=np.asarray(right[other:other+64],dtype='float64')
                wb=right.weights(slice(other,other+64)) if isinstance(right,Matrix) else np.ones(len(b))
                values=a @ b.T
                low=min(low,float(values.min()))
                total+=float((values*wa[:,None]*wb[None,:]).sum(dtype='float64'));count+=float(wa.sum()*wb.sum())
        if not count:raise ValueError('Body mode has no independent scoring witnesses.')
        return dict(minimum=low,mean=total/count)
    from .run_cache import packed
    from .run_cache import products
    for start in range(0,len(left),16):
        a=packed(left if len(left)<=16 else left[start:start+16])
        for other in range(0,len(right),64):
            b=packed(right if len(right)<=64 else right[other:other+64])
            values=products(a,b)
            low=min(low,float(values.min()));total+=float(np.sum(values,dtype='float64'));count+=values.size
    return dict(minimum=low,mean=total/count)


def preliminary_blocker(left,right,settings):
    """Only omit body scoring when merger and rival eligibility are both known."""
    import numpy as np
    floor=settings['face_floor']
    if not len(left.faces) or not len(right.faces):
        return 'face_contradiction'
    for a,b in zip(left.faces[:2],right.faces[:2]):
        if np.dot(a.astype('float64'),b.astype('float64')) < floor-1e-6:
            return 'face_contradiction'
    if min(len(left.independent),len(right.independent)) >= settings['min_support']:
        return None
    # Under-supported owners can still be rivals. Resolve every face veto
    # (including inherited witnesses) before omitting their body proof.
    for a,b in ((left.faces,right.faces),(left.guard_faces,right.faces),
                (left.faces,right.guard_faces),(left.guard_faces,right.guard_faces)):
        if a is not None and b is not None:
            measurement=scores(a,b)
            if measurement is not None and measurement['minimum']<floor:
                return 'face_contradiction'
    return 'insufficient_independent_support'


def pair(left,right,settings,*,copresent=False,reflection_threshold=1.):
    face=scores(left.faces,right.faces)
    from .hybrid_modes import covered_score
    mode_score=covered_score(left.body_modes,right.body_modes,settings)
    person=mode_score if mode_score is not None else scores(left.persons,right.persons)
    blockers=[]
    if left.unsupported_body_modes or right.unsupported_body_modes:
        blockers.append('unsupported_body_mode')
    if left.body_modes and right.body_modes and mode_score is None:
        blockers.append('incomplete_body_mode_coverage')
    # Former source cores are vetoes only: no extra votes or nominations.
    for kind,measurement in (('faces',face),('persons',person)):
        a,b=getattr(left,'guard_'+kind),getattr(right,'guard_'+kind)
        if kind=='persons' and mode_score is not None:
            from .hybrid_modes import guard_modes,query_score
            for bank,modes in ((a,right.body_modes),(b,left.body_modes)):
                if bank is None:continue
                for mode in guard_modes(bank,settings['person_threshold']):
                    guard=query_score(mode,bank,settings,modes)
                    measurement['minimum']=min(measurement['minimum'],guard['minimum'])
            # Current endpoints cannot hide a contradiction between inherited
            # cores. Every inherited mode needs reciprocal coverage as well.
            if a is not None and b is not None:
                inherited=covered_score(guard_modes(a,settings['person_threshold']),
                                        guard_modes(b,settings['person_threshold']),settings)
                if inherited is None:blockers.append('original_body_core_contradiction')
                else:measurement['minimum']=min(measurement['minimum'],inherited['minimum'])
            continue
        for first,second in ((a,getattr(right,kind)),(getattr(left,kind),b),(a,b)):
            if first is None or second is None:continue
            guard=scores(first,second)
            if guard is not None and measurement is not None:
                measurement['minimum']=min(measurement['minimum'],guard['minimum'])
    if left.guard_missing or right.guard_missing:blockers.append('missing_original_core_evidence')
    if person is None or any(i.missing and len(i.independent)<settings['min_support'] for i in (left,right)):
        blockers.append('missing_person_evidence')
    if min(len(left.independent),len(right.independent))<settings['min_support']:
        blockers.append('insufficient_independent_support')
    if face is None or face['minimum']<settings['face_floor']:blockers.append('face_contradiction')
    if person is not None and person['minimum']<settings['person_threshold']:blockers.append('person_below_threshold')
    if copresent and (face is None or face['minimum']<reflection_threshold):blockers.append('copresence_conflict')
    return dict(format=1,left=left.person,right=right.person,face=face,person=person,
        observations=[list(left.observations),list(right.observations)],
        person_observations=[list(left.person_observations),list(right.person_observations)],
        body_witness_observations=[list(i.body_witness_observations or i.person_observations) for i in (left,right)],
        guard_observations=[list(left.guard_observations),list(right.guard_observations)],
        guard_person_observations=[list(left.guard_person_observations),list(right.guard_person_observations)],
        missing_original_cores=[list(left.guard_missing),list(right.guard_missing)],
        support=[len(left.independent),len(right.independent)],
        missing=[list(left.missing),list(right.missing)],blockers=blockers,
        thresholds=dict(settings),cooccurring=copresent)


def plan(db,config,identities,*,focus=None,compact=False):
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
    contradictions=set()
    candidates=used=0
    def nominations():
        nonlocal candidates
        from .derived_cache import current as derived
        from .hybrid_delta import nominations as reusable_nominations
        floor=settings['person_threshold']-settings['ambiguity_margin']
        store=derived() if compact else None
        stream=(reusable_nominations(lookup,store,settings['face_floor'],floor,config) if compact and len(owned)>=2
                else lookup.pairs(face_floor=settings['face_floor'],person_floor=floor))
        with tracking(stream,'Nominate hybrid pairs',unit='nominated pair') as pairs:
            for a,b,face_mean,person_mean in pairs:
                if ((face_mean is None or face_mean+1e-6<settings['face_floor']) and
                    (person_mean is None or person_mean+1e-6<settings['person_threshold']-settings['ambiguity_margin'])):
                    continue
                candidates+=1
                if candidates>limits['max_candidate_pairs']:
                    raise ResourceLimit('SapiensID max_candidate_pairs budget exceeded; no hybrid plan applied.')
                if min(candidates,64)*256>limits['plan_bytes']:
                    raise ResourceLimit('SapiensID plan_bytes budget exceeded by nominations.')
                yield a,b
    from .hybrid_jobs import proofs as exact_proofs
    for a,b,proof in exact_proofs(db,owned,nominations(),config,compact=compact):
        if compact and proof['blockers']:
            if 'face_contradiction' not in proof['blockers']:continue
            contradictions.add((a,b))
            used+=128
        else:
            proofs[a,b]=proof
            used+=len(encoded(proof).encode())+128
        if used>limits['plan_bytes']:
            raise ResourceLimit('SapiensID plan_bytes budget exceeded; no hybrid plan applied.')
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
    # Reuse complete, outsider-separated prefixes from the immutable graph.
    # Overlapping incompatible proposals veto each other, never win by order.
    from .frozen_subgroups import Edges, nonoverlapping
    graph=Edges(limits['plan_bytes'])
    for (a,b),score in edges.items():graph.add(a,b,score)
    repairs=[]
    for decision in decisions:
        if decision['blockers']==['incomplete_component']:
            repairs.extend(graph.proposals(decision['people'],settings['person_threshold'],settings['ambiguity_margin']))
    repairs=[people for people in repairs if all(
        min(graph.score(p,q) for q in people if q!=p)-graph.score(p,r)>=settings['ambiguity_margin']
        for p in people for r in owned if r not in people)]
    for people in nonoverlapping(repairs):
        decisions.append(dict(people=list(people),outcome='merge',blockers=[],nearest_rival=None,
            frozen=True,contract=contract(config),partition_strategy='supported_complete_subgroup',
            proofs=[proofs[a,b] for a,b in combinations(people,2)]))
    # Second bounded pass: weak/missing-evidence outsiders still compete, but
    # their N-squared scores are never retained in Python dictionaries.
    eligible={p:d for d in decisions if not d['blockers'] for p in d['people']}
    internal={p:min(edges[tuple(sorted((p,q)))] for q in d['people'] if q!=p)
              for p,d in eligible.items()}
    from .hybrid_modes import RivalBounds
    bounds=RivalBounds(min(limits['reference_bytes']//32,8*1024**2))
    with tracking(lookup.pairs(incident=set(eligible)),'Check hybrid competitors',unit='nominated pair') as comparisons:
        for a,b,face_mean,person_mean in comparisons:
            # Known face contradiction is distinct from unknown body evidence.
            key=tuple(sorted((a,b)))
            if key in contradictions:continue
            witness=proofs.get(key)
            from .competitor_eligibility import hybrid as eligible_rival
            if not eligible_rival(witness, face_mean, settings['face_floor']):continue
            # A mean below the complete-link floor proves at least one veto,
            # including pairs screened out of positive nomination.
            from .hybrid_modes import competitor_score
            rivals=[(person,eligible[person]['nearest_rival']) for person,rival in ((a,b),(b,a))
                    if person in eligible and rival not in eligible[person]['people']]
            if not rivals:continue
            if all(known is not None for _,known in rivals) and owned[a].body_modes and owned[b].body_modes:
                upper=bounds.upper(owned[a],owned[b])
                if upper is not None and all(internal[person]-upper>known['gap'] for person,known in rivals):continue
            score=competitor_score(owned[a],owned[b],person_mean)
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
                pairs=[] if compact else [v for (a,b),v in proofs.items() if focus is None or a in focus or b in focus],
                identities=len(owned),enabled=config.data['sapiensid']['enabled'])


def report(db,config,scope,namespace,*,focus=None,compact=False):
    if not config.data['sapiensid']['enabled']:
        return dict(enabled=False)
    from .incomplete_rivals import protect
    return protect(db,scope,namespace,plan(db,config,catalogue(db,config,scope,namespace),focus=focus,compact=compact))


def catalogue(db, config, scope, namespace):
    from .evidence_pixels import alias_scope
    from .run_cache import current, database_key, MISSING
    from .config import digest, plain
    cache=current();generation=database_key(db)
    key=('hybrid-catalogue',generation,scope,namespace,digest(plain(config.data)))
    found=cache.get(key) if generation is not None else MISSING
    if found is not MISSING:return found
    with alias_scope(db,scope,namespace,budget=config.data['sapiensid']['execution']['reference_bytes']):
        found=_catalogue(db, config, scope, namespace)
    size=0
    for item in found:
        arrays=[item.faces,item.persons,item.guard_faces,item.guard_persons,*item.body_modes]
        arrays.extend(v for _,faces,bodies in item.query_cores for v in (faces,bodies))
        for array in arrays:
            if array is not None:
                from .disk_body_modes import Matrix,resident_bytes
                if not isinstance(array,Matrix):array.flags.writeable=False
                size+=resident_bytes(array)*2+256
        size+=1024+sum(len(v)+128 for v in item.observations+item.person_observations+item.guard_observations+item.body_witness_observations)
    if generation is not None:
        # Construction can populate disposable dependency digests. Cache the
        # finished generation so those writes do not force another catalogue.
        key=('hybrid-catalogue',database_key(db),scope,namespace,digest(plain(config.data)))
        cache.put(key,found,size)
    return found
