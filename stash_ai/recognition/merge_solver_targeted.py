"""Set-local measured routes, followed by full production replay.

Historical performers are diagnostic addresses, never mandatory identities.
The planner proposes global parameter patches; it cannot assign memberships or
certify transitive merges, competitor separation or global infeasibility.
"""
from collections import Counter, defaultdict

from .config import digest
from .merge_goal_sets import assess_set
from .merge_report_policy import parameter, _floor
from .merge_solver_domains import normalize
from .merge_solver_metrics import magnitude


def combine(*patches):
    result = {}
    for patch in patches:
        for key, value in patch.items():
            choose = max if key.endswith('pose_deviation') or type(value) is bool else min
            result[key] = choose(result.get(key, value), value)
    return result


def plan(db, config, goals, frozen, membership, feedback, hybrid):
    """Reuse bounded measurements from this exact rebuilt configuration."""
    from .global_views import _cooccurs
    from .pose_policy import restricted
    from .merge_solver_diagnostics import round_robin
    from .merge_solver_constraints import joint_separation
    strict = goals.get('goal_policy', {}).get('name') == 'strict'
    groups, by_observation, owners = {}, {}, defaultdict(set)
    original = {g['index']:g for g in goals['groups']}
    for source, group in zip(frozen.groups, goals['groups']):
        index = group['index']
        addresses = source['addresses']
        destinations = {membership[oid] for oid in addresses if membership.get(oid) is not None}
        measured = assess_set(source['evidence'], membership, membership, destinations)
        target = measured['set_target']
        missing = [s for s, support in measured['scene_support'].items() if not support.get(target)]
        raw = [r for m in group['members'] for r in m['observations']]
        satisfied = (bool(raw) and all(membership.get(r['id']) == target and target is not None for r in raw)
                     if strict else measured['identity_fully_merged'])
        members = []
        for member in group['members']:
            rows = member['observations']
            members.append(dict(performer=member['performer_id'] or member['identity_id'],
                observations=len(rows), assigned=sum(membership.get(r['id']) is not None for r in rows),
                destinations=sorted({membership[r['id']] for r in rows if membership.get(r['id']) is not None}),
                _observations={r['id'] for r in rows}, observed_blockers=Counter()))
        groups[index] = dict(set=index, target=target, satisfied=satisfied, fragments=sorted(destinations),
            fragment_scenes={d:[s for s,v in measured['scene_support'].items() if v.get(d)] for d in destinations},
            missing_target_scenes=missing, core_deficit=max(0, measured['set_core_required']-
                measured['set_core_destinations'].get(target, 0)), performers=members,
            observed_blockers=Counter(), candidates=[], seen=set(), truncated=False)
        for oid in addresses:by_observation[oid] = index
        for destination in destinations:owners[destination].add(index)

    def pair_set(a, b):
        indices = owners.get(a, set()) & owners.get(b, set())
        return next(iter(indices)) if len(indices) == 1 and a != b else None

    def lower(key, value):
        return min(parameter(config.data, key), _floor(value))

    quality = defaultdict(dict)
    quality_ids = defaultdict(list)
    for row in feedback.get('quality', []) + hybrid.get('body_quality', []):
        oid = row['observation']; index = by_observation.get(oid)
        if index is None or groups[index]['satisfied'] or not row.get('patch'):
            continue
        # Retain only missing/off-target evidence. No need to restore every
        # historical performer once its pooled set already meets the goal.
        if groups[index]['target'] is not None and membership.get(oid) == groups[index]['target']:
            continue
        quality[index] = combine(quality[index], row['patch'])
        quality_ids[index].append(oid)

    def add(index, route, patch, *, observations=(), fragments=(), blockers=(), gain=1):
        if index not in groups or groups[index]['satisfied']:
            return
        try:
            patch = normalize(config, patch)
        except ValueError:
            return
        group = groups[index]
        key = digest(patch)
        if not patch:
            return
        previous = next((p for p in group['candidates'] if p['patch']==patch),None)
        if previous is not None:
            if previous['gain']>=gain:return
            group['candidates'].remove(previous)
        if key not in group['seen'] and len(group['seen'])>=256:
            group['truncated']=True
            return
        group['seen'].add(key)
        involved = [m['performer_id'] or m['identity_id'] for m in original[index]['members']
                    if any(r['id'] in observations or membership.get(r['id']) in fragments for r in m['observations'])]
        item = dict(set=index, route=route, patch=patch, observations=sorted(set(observations))[:16],
                    fragments=sorted(set(fragments)), performers=involved, blockers=sorted(set(blockers)), gain=gain)
        group['candidates'].append(item)
        group['candidates'].sort(key=lambda p: (p['route']=='admission', -p['gain'],
            magnitude([dict(parameter=k,current=parameter(config.data,k),proposed=v) for k,v in p['patch'].items()]),
            len(p['patch']), digest(p['patch'])))
        del group['candidates'][16:]

    for index, patch in quality.items():
        add(index, 'admission', patch, observations=quality_ids[index], blockers=patch,
            gain=1+len(groups[index]['missing_target_scenes']))

    bridges = defaultdict(list)

    def bridge(index, route, patch, a, b, blockers=(), observations=()):
        if index is None:return
        group = groups[index]
        covered = set(group['fragment_scenes'].get(a,())) | set(group['fragment_scenes'].get(b,()))
        gain = max(1, len(covered & set(group['missing_target_scenes'])) + int(group['target'] in (a,b)))
        # Different edges can require the same YAML patch. Keep their endpoints
        # until spanning proposals are built; deduplicate only replay candidates.
        bridges[index,route].append(dict(patch=patch,fragments=(a,b),observations=observations,blockers=blockers))
        add(index, route, patch, fragments=(a,b), observations=observations, blockers=blockers, gain=gain)
        if quality[index]:
            add(index, route+'_admission', combine(patch, quality[index]), fragments=(a,b),
                observations=list(observations)+quality_ids[index], blockers=list(blockers)+list(quality[index]), gain=gain)

    for pair in hybrid.get('face_pairs', []):
        a, b = pair['left'], pair['right']; index = pair_set(a, b)
        if index is None or groups[index]['satisfied'] or pair['minimum'] < 0:continue
        patch = {key: lower(key, pair['minimum']) for key in
                 ('recognition.match_threshold', 'clustering.cohesion_threshold')}
        if _cooccurs(db, a, b, config):
            patch['recognition.cooccurrence_merge_threshold'] = lower('recognition.cooccurrence_merge_threshold', pair['minimum'])
        if restricted(config.data['clustering']):patch['clustering.anchor_view_matching'] = True
        bridge(index, 'face', patch, a, b, blockers=[k for k,v in patch.items() if parameter(config.data,k)!=v],
               observations=pair['observations'])

    for pair in hybrid.get('pairs', []):
        if pair['relation'] != 'within_set':continue
        p = pair.get('mode_boundary_evidence') or pair
        a, b = p['left'], p['right']; index = pair_set(a, b)
        if (index is None or groups[index]['satisfied'] or any(p.get('missing_original_cores', ())) or
                'missing_original_core_evidence' in p['blockers'] or
                not p.get('face') or not p.get('person') or min(p['support']) < 1 or
                min(p['face']['minimum'], p['person']['minimum']) < 0):continue
        prefix = 'sapiensid.matching.'
        patch = {prefix+'face_floor': lower(prefix+'face_floor',p['face']['minimum']),
                 prefix+'person_threshold': lower(prefix+'person_threshold',p['person']['minimum']),
                 prefix+'min_support': min(parameter(config.data,prefix+'min_support'), min(p['support']))}
        gaps = [p['person']['mean']-r['score'] for r in hybrid.get('nearest_competitors',[]) if r['identity'] in (a,b)]
        if gaps and min(gaps) < 0:continue
        if gaps:patch[prefix+'ambiguity_margin'] = lower(prefix+'ambiguity_margin',min(gaps))
        if p['cooccurring']:
            patch['recognition.cooccurrence_merge_threshold'] = lower('recognition.cooccurrence_merge_threshold',p['face']['minimum'])
        bridge(index, 'hybrid', patch, a, b, p['blockers'])

    bodies = hybrid.get('independent_person', {})
    for p in bodies.get('pairs', []):
        if p.get('relation') != 'within_set':continue
        a,b = p['people']; index = pair_set(a,b)
        if (index is None or groups[index]['satisfied'] or min(p['score'],p['minimum']) < 0 or min(p['support']) < 1
                or set(p['blockers']) & {'person_copresence','face_contradiction','missing_original_body_witness'}):continue
        prefix = 'sapiensid.global_matching.'
        patch = {prefix+'match_threshold':lower(prefix+'match_threshold',p['score']),
                 prefix+'cohesion_threshold':lower(prefix+'cohesion_threshold',p['minimum']),
                 prefix+'min_support':min(parameter(config.data,prefix+'min_support'),min(p['support']))}
        rivals = [r['score'] for r in bodies.get('nearest_competitors',[]) if r['identity'] in (a,b)]
        rivals += [r['score'] for r in bodies.get('pairs',[]) if r.get('relation') != 'within_set' and set(r['people']) & {a,b}]
        if rivals and max(rivals) > p['score']:continue
        if rivals:patch[prefix+'ambiguity_margin'] = lower(prefix+'ambiguity_margin',p['score']-max(rivals))
        bridge(index, 'body', patch, a, b, p['blockers'])

    query = feedback.get('query', {})
    # Hybrid observers cannot run without a body bank. Reuse the existing
    # retained-face fallback measurements and keep their set/performer address.
    for q in query.get('face_routes', []):
        oid=q['observation'];index=by_observation.get(oid)
        if index is None or index not in owners.get(q['identity'], ()):continue
        minimal={k:v for k,v in q['patch'].items() if '.view_' not in k}
        for patch in (minimal,q['patch']):
            add(index,'face_query',combine(patch,quality[index]),observations=[oid]+quality_ids[index],
                fragments=[q['identity']],blockers=list(patch)+list(quality[index]),gain=2)
    for q in query.get('queries', []):
        oid = q['observation']; index = by_observation.get(oid)
        if (index is None or groups[index]['satisfied'] or 'identity' not in q or
                membership.get(oid) == q['identity'] or index not in owners.get(q['identity'], ())):continue
        face_support = min(q['support'], q.get('face_query_support',0))
        face_gap = q.get('face_gap')
        if face_support and q['face']['minimum'] >= 0 and (face_gap is None or face_gap >= 0):
            patch = {key:lower(key,q['face']['minimum']) for key in
                     ('recognition.match_threshold','clustering.cohesion_threshold')}
            for key in ('recognition.min_identity_support','clustering.auxiliary_min_support'):
                patch[key] = min(parameter(config.data,key), face_support)
            if face_gap is not None:patch['recognition.ambiguity_margin'] = lower('recognition.ambiguity_margin',face_gap)
            for gates in (patch,combine(patch,quality[index])):
                add(index,'face_query',gates,observations=[oid],
                    fragments=[q['identity']],blockers=['face_query_thresholds'],gain=2)
        if (q.get('missing_original_core') or
                q.get('unknown_competitor') or q.get('gap') is not None and q['gap'] < 0):continue
        support = min(q['support'], q.get('query_support', 0))
        if not support or not q.get('person') or min(q['face']['minimum'],q['person']['minimum']) < 0:continue
        prefix = 'sapiensid.matching.'
        patch = {prefix+'face_floor':lower(prefix+'face_floor',q['face']['minimum']),
                 prefix+'person_threshold':lower(prefix+'person_threshold',q['person']['minimum']),
                 prefix+'min_support':min(parameter(config.data,prefix+'min_support'),support)}
        if q.get('gap') is not None:patch[prefix+'ambiguity_margin'] = lower(prefix+'ambiguity_margin',q['gap'])
        # Use complete per-rival witnesses, never mixed coordinate maxima.
        candidates = joint_separation(config,[patch],query.get('negative_witnesses',[]),limit=8)['proposals']
        for candidate in candidates[:2]:
            for gates in (candidate,combine(candidate,quality[index])):
                add(index,'query',gates,observations=[oid],
                    fragments=[q['identity']],blockers=q['blockers'],gain=2)

    # A few complementary bridges may clear several gaps together. A spanning
    # proposal is just an experiment: complete-core validation still runs.
    for index, group in groups.items():
        for route in ('face','hybrid','body'):
            edges = sorted(bridges[index,route],key=lambda p:magnitude([
                dict(parameter=k,current=parameter(config.data,k),proposed=v) for k,v in p['patch'].items()]))
            roots, chosen = {}, []
            def root(a):
                while a in roots:a=roots[a]
                return a
            for p in edges:
                a,b=map(root,p['fragments'])
                if a==b:continue
                roots[b]=a;chosen.append(p)
            if len(chosen)>1:
                add(index,route+'_set_bridges',combine(*(p['patch'] for p in chosen),quality[index]),
                    fragments=[f for p in chosen for f in p['fragments']],
                    observations=[o for p in chosen for o in p['observations']],
                    blockers=[b for p in chosen for b in p['blockers']],gain=1+len(chosen))
    for row in feedback.get('provenance',{}).get('blocker_trace',[]):
        indices = {by_observation[row['observation']]} if row.get('observation') in by_observation else set()
        for key in ('left','right','target'):indices.update(owners.get(row.get(key),set()))
        for index in indices:
            groups[index]['observed_blockers'][row['reason']] += 1
            for member in groups[index]['performers']:
                if (row.get('observation') in member['_observations'] or
                        any(row.get(k) in member['destinations'] for k in ('left','right','target'))):
                    member['observed_blockers'][row['reason']] += 1
    candidates = list(round_robin({i:g['candidates'] for i,g in groups.items()}))
    for group in groups.values():
        group['candidate_count'] = len(group.pop('candidates'))
        group.pop('seen')
        group['observed_blockers'] = [dict(reason=r,count=n) for r,n in group['observed_blockers'].most_common(16)]
        for member in group['performers']:
            member.pop('_observations')
            member['observed_blockers'] = [dict(reason=r,count=n) for r,n in member['observed_blockers'].most_common(8)]
    return dict(sets=list(groups.values()), candidates=candidates[:64],
        truncated=len(candidates)>64 or any(g['truncated'] for g in groups.values()),
        interpretation='Measured routes on this replay only. Performers are addresses, not preservation obligations. '
                       'Global production replay must verify every candidate, including all rivals and exclusions.')


class TargetedSearch:
    def __init__(self, budget):
        self.budget, self.executed, self.turn = budget, 0, 0
        self.seen = set()

    def next(self, parent, feedback, parent_changes):
        if self.executed >= self.budget:return None
        by_set = defaultdict(list)
        for item in feedback.get('targeted',{}).get('candidates',[]):
            if (parent,digest(item['patch'])) not in self.seen:by_set[item['set']].append(item)
        if not by_set:return None
        indices = sorted(by_set)
        index = next((i for i in indices if i >= self.turn), indices[0])
        item = by_set[index][0]
        self.turn = index+1
        self.seen.add((parent,digest(item['patch'])))
        return dict(item, patch=dict(parent_changes,**item['patch']))
