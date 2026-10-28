"""Bounded conjunction/disjunction experiments on frozen witnesses."""
from itertools import combinations, islice
import math
from .config import digest
from .merge_report_policy import parameter, _floor
from .merge_solver_search import PREFIX


def above(key, value):
    from .threshold_boundaries import strictly_above
    return strictly_above(value, integer=key.endswith('support'))


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


def alternative_separation(config, obligations, negatives, limit=32):
    """AND of obligations, OR of routes within each obligation.

    The bounded states are acceptance upper bounds, not accepted identities.
    A fresh production replay remains necessary for support and competition.
    """
    states = [{}]
    def cost(p):
        return sum(abs(v-parameter(config.data,k))/max(1.,abs(parameter(config.data,k))) for k,v in p.items())
    for routes in obligations:
        if not routes:
            continue
        expanded = {}
        for state in states:
            for route in routes:
                combined = dict(state)
                for key, value in route.items():
                    combined[key] = min(combined.get(key, value), value)
                expanded[digest(combined)] = combined
        states = sorted(expanded.values(), key=lambda p: (cost(p), digest(p)))[:limit]
    proposals = []
    for state in states:
        proposals.extend(joint_separation(config, [state], negatives, limit=limit)['proposals'])
    unique = {digest(p): p for p in proposals}
    return sorted(unique.values(), key=lambda p: (
        -sum(any(p.get(k, -math.inf) > v for k, v in n.items()) for n in negatives), cost(p), digest(p)))[:limit]


def component_proposals(config, hybrid, limit=16, *, targets=None):
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
        from .merge_solver_targets import bridge_gain
        gain = bridge_gain(proof['left'], proof['right'], targets or {})
        rivals = [r['score'] for r in hybrid.get('nearest_competitors', ())
                  if r['identity'] in (proof['left'], proof['right'])]
        margin = proof['person'].get('mean', proof['person']['minimum'])-max(rivals, default=0.)
        if rivals and margin >= 0:
            patch[PREFIX+'ambiguity_margin'] = min(config.data['sapiensid']['matching']['ambiguity_margin'], _floor(margin))
        if proof.get('cooccurring'):
            patch['recognition.cooccurrence_merge_threshold'] = min(
                config.data['recognition']['cooccurrence_merge_threshold'], _floor(proof['face']['minimum']))
        edges.append((-gain, -margin, -(proof['face']['minimum']+proof['person']['minimum']),proof['left'],proof['right'],patch))
    roots, combined, result = {}, {}, []
    def root(v):
        while roots.get(v,v)!=v:v=roots[v]
        return v
    for _,_,_,a,b,patch in sorted(edges):
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


def learned_groups(feedback):
    """Discover interacting gates on the same actual blocked query or pair."""
    from collections import defaultdict, Counter
    witnesses = defaultdict(set)
    for row in feedback.get('provenance', {}).get('blocker_trace', ()):
        key = ('query', row['observation']) if row.get('observation') else ('pair', row.get('left'), row.get('right'))
        if key == ('pair', None, None):
            continue
        witnesses[key].update(row.get('parameters', ()))
    for row in feedback.get('quality', ()):
        witnesses['query', row['observation']].update(row.get('patch', {}))
    counts = Counter(tuple(sorted(keys)) for keys in witnesses.values() if len(keys) >= 2)
    return [keys for keys, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def interaction_proposals(config, changes, options, feedback, limit=24):
    from itertools import product
    from .merge_solver_domains import registry, normalize
    domains, seen = registry(config), set()
    for group in learned_groups(feedback):
        keys = [k for k in group if k in domains and parameter(config.data, k) is not None]
        if len(keys) < 2:
            continue
        # Keep up to three simultaneous blockers, including cross-family gates.
        for selected in combinations(keys, min(3, len(keys))):
            alternatives = []
            for key in selected:
                current = changes.get(key, parameter(config.data, key))
                measured = [domains[key].project(v) for v in options.get(key, ()) if v != current]
                values = sorted(set(measured), key=lambda v: (
                    (float(v)-float(current))*domains[key].relax <= 0, abs(float(v)-float(current)), float(v)))
                alternatives.append(values[:2])
            for values in product(*alternatives):
                patch = normalize(config, dict(changes, **dict(zip(selected, values))))
                signature = digest(patch)
                if patch and signature not in seen:
                    seen.add(signature)
                    yield patch
                if len(seen) >= limit:
                    return


def coupled_refinement(config, changes, options, feedback=None):
    from .merge_solver_search import refinement
    singles=list(refinement(config,changes,options))
    by_key={}
    for patch in singles:
        key=next(k for k in patch if patch[k]!=changes[k])
        by_key.setdefault(key,[]).append(patch[key])
    groups=tuple(learned_groups(feedback or {}))+((PREFIX+'face_floor',PREFIX+'person_threshold',PREFIX+'ambiguity_margin',PREFIX+'min_support'),
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
