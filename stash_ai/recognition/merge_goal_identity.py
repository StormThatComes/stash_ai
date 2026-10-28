"""Identity-level goals on frozen, noisy membership evidence.

Reference eligibility is a confidence hint, not a truth label. Matching never
receives these goals; only the offline evaluator and proposal diagnostics do.
"""
from collections import Counter, defaultdict
from math import floor

POLICY = dict(name='identity', version=2, preservation='set_scene_coverage', core_outlier_fraction=.2,
              minimum_core_outlier_allowance=1, minimum_core_support=2,
              soft_loss_fraction=.2, minimum_soft_loss_allowance=3,
              protected_scene_support=2)


def required(total):
    """A small minority may be noisy; never retain zero or only one of many."""
    return max(min(POLICY['minimum_core_support'], total),
               total-max(POLICY['minimum_core_outlier_allowance'], floor(total*POLICY['core_outlier_fraction'])))


def units(rows):
    from .independent_evidence import representatives, unit
    from .config import digest
    chosen = {unit(r) for r in representatives(rows)}
    grouped = defaultdict(list)
    for row in rows:
        key = unit(row)
        if key in chosen:
            grouped[key].append(dict(id=row['id'], scene=row['scene']))
    return [dict(id=digest(key), observations=sorted(grouped[key], key=lambda r: r['id']))
            for key in sorted(grouped, key=repr)]


def freeze(rows):
    """Snapshot support and roles once, before a trial can change admission gates."""
    primary = [r for r in rows if r['goal_role'] == 'reference']
    soft = [r for r in rows if r['goal_role'] != 'reference']
    fallback = not primary
    core_units, soft_units = units(rows if fallback else primary), units(soft)
    scenes = Counter(scene for u in core_units for scene in {r['scene'] for r in u['observations']})
    return dict(basis='fallback_membership' if fallback else 'baseline_reference_eligible',
                core_units=core_units, soft_units=soft_units, core_required=required(len(core_units)),
                protected_scenes={s: required(n) for s, n in sorted(scenes.items())
                                  if n >= POLICY['protected_scene_support']},
                soft_loss_allowance=min(len(soft_units), max(POLICY['minimum_soft_loss_allowance'],
                    floor(len(soft_units)*POLICY['soft_loss_fraction']))))


def votes(units, observations, predicted):
    counts, scenes = Counter(), defaultdict(Counter)
    for unit in units:
        rows = [r for r in unit['observations'] if r['id'] in observations and predicted.get(r['id']) is not None]
        destinations = {predicted[r['id']] for r in rows}
        if len(destinations) != 1:
            continue  # Conflicting duplicates cannot vote for two identities.
        destination = next(iter(destinations))
        counts[destination] += 1
        for scene in {r['scene'] for r in rows}:
            scenes[scene][destination] += 1
    return counts, scenes


def member_result(member, observations, predicted):
    evidence = member['identity_evidence']
    total = len(evidence['core_units'])
    counts, scenes = votes(evidence['core_units'], observations, predicted)
    candidates = [d for d, n in counts.items() if n >= evidence['core_required'] and
                  all(scenes[s][d] >= need for s, need in evidence['protected_scenes'].items())]
    destination = candidates[0] if len(candidates) == 1 and total else None
    soft, _ = votes(evidence['soft_units'], observations, predicted)
    soft_total = len(evidence['soft_units'])
    soft_loss = soft_total-soft.get(destination, 0) if destination is not None else soft_total-sum(soft.values())
    # More evidence than the permitted minority constitutes a real conflict,
    # even when no destination has enough support to represent the performer.
    significant = {d: n for d, n in counts.items() if n > total-evidence['core_required']}
    return dict(basis=evidence['basis'], represented=destination is not None,
                destination=destination, core_units=total, core_assigned=sum(counts.values()),
                core_destinations=dict(counts), significant_destinations=significant,
                core_required=evidence['core_required'],
                protected_scenes=evidence['protected_scenes'],
                scene_support={s: dict(v) for s, v in scenes.items()},
                soft_units=soft_total, soft_lost_units=soft_loss,
                soft_loss_allowance=evidence['soft_loss_allowance'],
                soft_budget_excess=max(0, soft_loss-evidence['soft_loss_allowance']))


def apply(goals, status, observations, predicted):
    """Keep member/strict diagnostics; preserve the declared person as a set."""
    from .merge_goal_sets import assess_set, set_destinations
    status.update(goal_policy=dict(POLICY), strict_disjoint=status['disjoint'],
                  strict_conflicts=status['conflicts'], strict_admissible=status['admissible'],
                  strict_fully_solved=status['fully_solved'])
    owners, raw_owners = defaultdict(dict), defaultdict(set)
    for source, group in zip(goals['groups'], status['groups']):
        for destination in group['resulting_identities']:
            raw_owners[destination].add(group['index'])
        components, fractions, squares = Counter(), Counter(), Counter()
        represented = assigned = 0
        for original, member in zip(source['members'], group['members']):
            evidence = member_result(original, observations, predicted)
            member['identity_evidence'] = evidence
            represented += evidence['represented']
            if evidence['represented']:
                components[evidence['destination']] += 1
            total = evidence['core_units']
            assigned += evidence['core_assigned']/total if total else 0.
            for destination, count in evidence['core_destinations'].items():
                fraction = count/total
                fractions[destination] += fraction
                squares[destination] += fraction*fraction
            if (evidence['basis'] != 'fallback_membership' or
                    source['identity_evidence']['basis'] == 'fallback_membership'):
                for destination, count in evidence['significant_destinations'].items():
                    sets = owners[destination]
                    sets[group['index']] = sets.get(group['index'], 0)+count
        n = group['total_performers']
        consolidated = max(components.values(), default=0)
        group.update(represented_performers=represented, representation_fraction=represented/n,
                     consolidated_performers=consolidated, identity_fraction=consolidated/n,
                     identity_components=sorted(components.values(), reverse=True),
                     core_retention=assigned/n, core_shared_coverage=max(fractions.values(), default=0.)/n,
                     core_pair_fraction=(sum(v*v-squares[k] for k, v in fractions.items())/(n*(n-1))
                                         if n > 1 else float(bool(represented))))
    for source, group in zip(goals['groups'], status['groups']):
        destinations = set_destinations(group['index'], raw_owners, owners)
        group.update(assess_set(source['identity_evidence'], observations, predicted, destinations))
    conflicts = []
    raw_groups = {g['index']: g for g in status['groups']}
    for destination, sets in sorted(owners.items()):
        if len(sets) > 1:
            values = list(sets.values())
            raw = [raw_groups[s]['resulting_identities'].get(destination, 0) for s in sets]
            conflicts.append(dict(identity=destination, sets=sorted(sets),
                evidence_pairs=(sum(values)**2-sum(v*v for v in values))//2,
                observation_pairs=(sum(raw)**2-sum(v*v for v in raw))//2))
    preserved = all(g['set_evidence_preserved'] for g in status['groups'])
    status.update(disjoint=not conflicts, conflicts=conflicts,
                  cross_set_evidence_pairs=sum(c['evidence_pairs'] for c in conflicts),
                  evidence_preserved=preserved, admissible=preserved and not conflicts,
                  fully_solved=preserved and not conflicts and all(g['identity_fully_merged'] for g in status['groups']))
    return status


def objective(status):
    groups = status['groups']
    def worst(key): return min(g[key] for g in groups)
    def mean(key): return sum(g[key] for g in groups)/len(groups)
    return (status['admissible'], status['fully_solved'], status['disjoint'],
            worst('scene_coverage_fraction'), mean('scene_coverage_fraction'),
            -status['cross_set_evidence_pairs'],
            worst('set_shared_coverage'), mean('set_shared_coverage'),
            worst('set_pair_fraction'), mean('set_pair_fraction'),
            -mean('soft_loss_fraction'), -sum(g['soft_budget_excess'] for g in groups),
            mean('set_core_retention'))


def unmet(status):
    result = []
    for group in status['groups']:
        if not group['required_scenes']:
            result.append(dict(set=group['index'], reason='empty_identity_set'))
        if group['missing_scenes']:
            result.append(dict(set=group['index'], reason='set_scene_not_preserved',
                               scenes=group['missing_scenes']))
        if not group['identity_fully_merged']:
            result.append(dict(reason='incomplete_identity_set', set=group['index'],
                               core_required=group['set_core_required'],
                               core_support=group['set_core_destinations'], scene_support=group['scene_support']))
    result.extend(dict(reason='cross_set_merge', **c) for c in status['conflicts'])
    return result


def diagnostic_goals(goals):
    """Include scene fallback queries so search can recover soft-only scenes."""
    if goals.get('goal_policy', {}).get('name') != 'identity':
        return goals
    groups = []
    for group in goals['groups']:
        ids = {r['id'] for scene in group['identity_evidence']['scenes'].values()
               for unit in scene['scoring_units'] for r in unit['observations']}
        groups.append(dict(group, members=[dict(m, observations=[r for r in m['observations'] if r['id'] in ids])
                                           for m in group['members']]))
    return dict(goals, groups=groups)


def destination_owners(goals, predicted):
    """Label rebuilt destinations consistently for pair and query proposals."""
    owners, raw = defaultdict(set), defaultdict(set)
    for group in goals['groups']:
        for member in group['members']:
            for row in member['observations']:
                if predicted.get(row['id']) is not None:
                    raw[predicted[row['id']]].add(group['index'])
            if goals.get('goal_policy', {}).get('name') == 'identity':
                evidence = member_result(member, predicted, predicted)
                destinations = (evidence['significant_destinations']
                    if evidence['basis'] != 'fallback_membership' or
                    group['identity_evidence']['basis'] == 'fallback_membership' else {})
            else:
                destinations = {predicted[r['id']] for r in member['observations'] if predicted.get(r['id']) is not None}
            for destination in destinations:
                owners[destination].add(group['index'])
    if goals.get('goal_policy', {}).get('name') == 'identity':
        # Match the evaluator: substantial core owns a destination; otherwise
        # an exclusive soft fragment can still guide recovery of its set.
        for destination, sets in raw.items():
            if not owners.get(destination) and len(sets) == 1:
                owners[destination] = sets
    return owners
