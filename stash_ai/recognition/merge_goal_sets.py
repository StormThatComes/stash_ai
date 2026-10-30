"""Preservation and consolidation of one declared person across a whole set."""
from collections import Counter, defaultdict

from .merge_goal_identity import freeze, units, votes, required


def freeze_set(rows):
    """Freeze pooled evidence before replay, including every original scene.

    A weak historical fragment does not become mandatory core when another
    fragment already supplies references. Independent units are pooled before
    counting, so duplicating or repartitioning memberships cannot add votes.
    """
    evidence = freeze(rows)
    scenes = defaultdict(list)
    for row in rows:
        scenes[row['scene']].append(row)
    return dict(evidence, scenes={scene: dict(
        membership_units=units(selected),
        scoring_units=units([r for r in selected if r['goal_role'] == 'reference'] or selected))
        for scene, selected in sorted(scenes.items())})


def set_destinations(index, raw_owners, core_owners):
    """A foreign person's core cannot preserve this person's scene coverage.

    An exclusively owned soft fragment may still cover a scene. Ambiguous soft
    overlaps without a substantial core owner cannot establish ownership.
    """
    return {d for d, owners in raw_owners.items()
            if (set(core_owners[d]) if core_owners.get(d) else owners) == {index}}


def ownership(goals, observations, predicted):
    """Substantial scene-local core owns a destination, independent of members.

    Pool/deduplicate before counting. A small scene cannot be diluted by a large
    scene of the same person. Soft-only scenes cannot manufacture core ownership.
    """
    core_owners, raw_owners = defaultdict(dict), defaultdict(set)
    for group in goals['groups']:
        index = group['index']
        for member in group['members']:
            for row in member['observations']:
                destination = predicted.get(row['id'])
                if row['id'] in observations and destination is not None:
                    raw_owners[destination].add(index)
        evidence = group['identity_evidence']
        if goals.get('goal_policy', {}).get('continuity') == 'face_or_body':
            for scene in evidence['scenes'].values():
                for unit in scene['membership_units']:
                    for row in unit['observations']:
                        destination = predicted.get(row['id'])
                        if row['id'] in observations and destination is not None:
                            raw_owners[destination].add(index)
        core = evidence['core_units']
        total_counts, _ = votes(core, observations, predicted)
        from .merge_goal_index import current
        index_cache = current()
        if index_cache is not None:
            cohorts = index_cache.cohorts(evidence)
        else:
            cohorts = defaultdict(list)
            for unit in core:
                for scene in {r['scene'] for r in unit['observations']}:
                    cohorts[scene].append(dict(unit, observations=[r for r in unit['observations'] if r['scene'] == scene]))
        significant = set()
        for cohort in cohorts.values():
            counts, _ = votes(cohort, observations, predicted)
            allowance = len(cohort)-required(len(cohort))
            significant.update(d for d, count in counts.items() if count > allowance and total_counts[d])
        for destination in significant:
            core_owners[destination][index] = total_counts[destination]
    return core_owners, raw_owners


def assess_set(evidence, observations, predicted, destinations):
    """Separate scene preservation, common-destination progress and soft loss."""
    raw_core, _ = votes(evidence['core_units'], observations, predicted)
    core = Counter({d: n for d, n in raw_core.items() if d in destinations})
    scene_support, shared, raw_shared = {}, Counter(), Counter()
    squared, raw_squared = Counter(), Counter()
    within, raw_within = [], []
    for scene, frozen in evidence['scenes'].items():
        counts, _ = votes(frozen['membership_units'], observations, predicted)
        scene_support[scene] = {d: n for d, n in counts.items() if d in destinations}
        scores, _ = votes(frozen['scoring_units'], observations, predicted)
        total_scores = len(frozen['scoring_units'])
        raw_within.append(sum(n*(n-1) for n in scores.values())/(total_scores*(total_scores-1))
                          if total_scores > 1 else float(bool(scores)))
        within.append(sum(n*(n-1) for d, n in scores.items() if d in destinations)/(total_scores*(total_scores-1))
                      if total_scores > 1 else float(any(d in destinations for d in scores)))
        for destination, count in scores.items():
            fraction = count/total_scores
            raw_shared[destination] += fraction
            raw_squared[destination] += fraction*fraction
            if destination in destinations:
                shared[destination] += fraction
                squared[destination] += fraction*fraction
    missing = [s for s, support in scene_support.items() if not support]
    preserved = bool(scene_support) and not missing
    # Coverage across several destinations is admissible progress, not a full
    # merge. One destination must cover every scene and retain pooled core quorum.
    complete = {d for d, n in core.items() if n >= evidence['core_required'] and
                all(support.get(d, 0) for support in scene_support.values())}
    candidates = complete or shared.keys()
    rank = {d: (shared[d], core[d]) for d in candidates}
    best = max(rank.values(), default=None)
    tied = sorted(d for d, value in rank.items() if value == best)
    soft, _ = votes(evidence['soft_units'], observations, predicted)
    soft_total = len(evidence['soft_units'])
    lost = soft_total-sum(n for d, n in soft.items() if d in destinations)
    total, scenes = len(evidence['core_units']), len(scene_support)
    def pairs(fractions, squares, local):
        # Every ordered pair of distinct scenes has equal weight. For a single
        # scene retain the independent-unit pair score (no self-pair inflation).
        return (max(0., sum(v*v-squares[d] for d, v in fractions.items()))/(scenes*(scenes-1))
                if scenes > 1 else (local[0] if local else 0.))
    return dict(set_evidence_preserved=preserved, identity_fully_merged=bool(complete),
                required_scenes=scenes, covered_scenes=scenes-len(missing), missing_scenes=missing,
                scene_coverage_fraction=(scenes-len(missing))/scenes if scenes else 0.,
                scene_support=scene_support, set_destinations=sorted(destinations),
                set_target=tied[0] if tied else None, set_target_ties=tied,
                set_core_units=total, set_core_required=evidence['core_required'],
                set_core_destinations=dict(core), set_core_retention=sum(core.values())/total if total else 0.,
                set_shared_coverage=max(shared.values(), default=0.)/scenes if scenes else 0.,
                set_pair_fraction=pairs(shared, squared, within),
                # The exploratory frontier deliberately keeps overmerges for
                # later repair. These raw scores never select the winner.
                raw_set_pair_fraction=pairs(raw_shared, raw_squared, raw_within),
                raw_set_shared_coverage=max(raw_shared.values(), default=0.)/scenes if scenes else 0.,
                soft_lost_units=lost, soft_loss_fraction=lost/soft_total if soft_total else 0.,
                soft_loss_allowance=evidence['soft_loss_allowance'],
                soft_budget_excess=max(0, lost-evidence['soft_loss_allowance']))
