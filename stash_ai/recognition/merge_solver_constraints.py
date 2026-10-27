"""Bounded conjunction/disjunction experiments on frozen witnesses."""
from itertools import combinations, islice
import math
from .config import digest
from .merge_report_policy import parameter, _floor
from .merge_solver_search import PREFIX


def above(key, value):
    return int(value)+1 if key.endswith('support') else round(math.floor(value*1e6)/1e6+.000001,6)


def joint_separation(config, positives, negatives, limit=32):
    """Desired routes require all gates; each unwanted route must fail any gate.

    Preserve per-rival vectors: maxima from different rivals are not a witness.
    The bounded frontier is a proposal generator, not an infeasibility proof.
    """
    upper = {}
    for row in positives:
        for key,value in row.items():
            if value < (1 if key.endswith('support') else 0):
                continue
            upper[key] = min(upper.get(key,value), int(value) if key.endswith('support') else _floor(value))
    if not upper:return dict(proposals=[],unresolved=[],truncated=False)
    base = {k:min(parameter(config.data,k),v) for k,v in upper.items()}
    states, unresolved, truncated = [base], [], False
    def cost(p):
        return sum(abs(v-parameter(config.data,k))/max(1.,abs(parameter(config.data,k))) for k,v in p.items())
    for index,rival in enumerate(negatives):
        expanded = []
        for patch in states:
            if any(patch[k] > value for k,value in rival.items() if k in patch):
                expanded.append(patch)
                continue
            for key,value in rival.items():
                if key not in upper:continue
                reject = above(key,value)
                if reject <= upper[key]:expanded.append(dict(patch,**{key:max(patch[key],reject)}))
        if not expanded:
            unresolved.append(index)
            continue  # Keep best-effort states; this is not a global impossibility claim.
        unique = sorted({digest(p):p for p in expanded}.values(),key=lambda p:(cost(p),digest(p)))
        truncated |= len(unique)>limit
        states = unique[:limit]
    # Prefer routes satisfying more unwanted witnesses, then a smaller mutation.
    states.sort(key=lambda p:(-sum(any(p.get(k,-math.inf)>v for k,v in r.items()) for r in negatives),cost(p),digest(p)))
    return dict(proposals=states,unresolved=unresolved,truncated=truncated)


def component_proposals(config, hybrid, limit=16):
    """Strongest fragment bridges first; complete-core production checks still apply."""
    edges = []
    for entry in hybrid.get('pairs', ()):
        if entry['relation'] != 'within_set':continue
        proof = entry.get('mode_boundary_evidence') or entry
        if any(proof.get('missing_original_cores',())) or not all(proof.get(k) for k in ('face','person')):continue
        if min(proof['face']['minimum'],proof['person']['minimum'])<0 or min(proof['support'])<1:continue
        patch = {PREFIX+'face_floor':min(config.data['sapiensid']['matching']['face_floor'],_floor(proof['face']['minimum'])),
                 PREFIX+'person_threshold':min(config.data['sapiensid']['matching']['person_threshold'],_floor(proof['person']['minimum'])),
                 PREFIX+'min_support':min(config.data['sapiensid']['matching']['min_support'],min(proof['support']))}
        edges.append((-(proof['face']['minimum']+proof['person']['minimum']),proof['left'],proof['right'],patch))
    roots, combined, result = {}, {}, []
    def root(v):
        while roots.get(v,v)!=v:v=roots[v]
        return v
    for _,a,b,patch in sorted(edges):
        a,b = root(a),root(b)
        if a==b:continue
        roots[b]=a
        result.append(patch)
        for k,v in patch.items():combined[k]=min(combined.get(k,v),v)
        result.append(dict(combined))
        if len(result)>=limit:break
    return list({digest(p):p for p in result}.values())


def crossovers(changes, parents, limit=12):
    """Keep both resolutions of conflicting values, with bounded combinations."""
    emitted=set()
    for other in parents:
        conflicts=[k for k in sorted(changes.keys() & other.keys()) if changes[k]!=other[k]]
        patches=[dict(changes,**other),dict(other,**changes)]
        if len(conflicts)>1:
            for key in conflicts[:4]:patches.append({**changes,**other,key:changes[key]})
        for patch in patches:
            key=digest(patch)
            if key in emitted or patch==changes:continue
            emitted.add(key);yield patch
            if len(emitted)>=limit:return


def coupled_refinement(config, changes, options):
    from .merge_solver_search import refinement
    singles=list(refinement(config,changes,options))
    by_key={}
    for patch in singles:
        key=next(k for k in patch if patch[k]!=changes[k])
        by_key.setdefault(key,[]).append(patch[key])
    groups=((PREFIX+'face_floor',PREFIX+'person_threshold',PREFIX+'ambiguity_margin',PREFIX+'min_support'),
            ('recognition.match_threshold','clustering.cohesion_threshold'),
            ('clustering.view_match_threshold','clustering.view_cohesion_threshold'),
            tuple(k for k in changes if k.startswith('clustering.min_')))
    seen=set()
    for group in groups:
        for keys in combinations([k for k in group if k in by_key],2):
            values=[sorted(by_key[k],key=lambda v:abs(float(v)-float(changes[k])))[:3] for k in keys]
            for a in values[0]:
                for b in values[1]:
                    patch=dict(changes,**dict(zip(keys,(a,b))))
                    key=digest(patch)
                    if key not in seen:
                        seen.add(key);yield patch
                    if len(seen)>=32:return
