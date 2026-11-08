"""Explicit matched comparisons over frozen, labelled VTT evidence."""
from collections import Counter, defaultdict
from dataclasses import replace
from .config import plain
from .body_identity import scene_bonus, enabled


def variants(config):
    if not enabled(config):
        raise ValueError('Scene strategy comparison requires body identity matching.')
    positive = scene_bonus(config.data) or config.data['clustering']['same_scene_match_bonus'] or .03
    result = {}
    for tracks in (False,True):
        for mode in ('off','fixed','evidence'):
            data = plain(config.data)
            data['sapiensid']['identities'].update(tracklet_matching=tracks,
                scene_bonus_mode='evidence' if mode=='evidence' else 'fixed',
                scene_match_bonus=0. if mode=='off' else positive)
            result[f"tracks-{int(tracks)}-{mode}"] = replace(config,data=data)
    return result


def physical_predictions(labels,locations,ids,decisions):
    seen = defaultdict(set)
    for oid in ids:
        owners = seen[(*locations[oid],labels[oid])]
        value = decisions[oid]['accepted_identity']
        if value is not None:
            owners.add(value)
    return {key:next(iter(owners)) if len(owners)==1 else None for key,owners in seen.items()}


def omission_damage(labels,locations,ids,before,after):
    left = physical_predictions(labels,locations,ids,before)
    right = physical_predictions(labels,locations,ids,after)
    by_truth, by_owner, common_truth, common_owner, right_truth = (Counter() for _ in range(5))
    for key,a in left.items():
        truth,b = key[-1],right[key]
        if a is not None:
            by_truth[truth,a] += 1
        if b is not None:
            by_owner[b] += 1
            right_truth[truth,b] += 1
        if a is not None and b is not None:
            common_truth[truth,a,b] += 1
            common_owner[a,b] += 1
    def pairs(counts):
        return sum(n*(n-1)//2 for n in counts.values())
    return dict(lost_correct_pairs=pairs(by_truth)-pairs(common_truth),
                new_false_joins=pairs(by_owner)-pairs(common_owner)-pairs(right_truth)+pairs(common_truth))


def strategy_summary(config,metrics,labels,locations,decisions):
    settings = config.data['sapiensid']['identities']
    identities = defaultdict(set)
    for (*_,truth),owner in physical_predictions(labels,locations,labels,decisions).items():
        if owner is not None:
            identities[truth].add(owner)
    seq = metrics['sequences']
    return dict(tracklet_matching=settings['tracklet_matching'],scene_match_bonus=settings['scene_match_bonus'],
                scene_bonus_mode=settings['scene_bonus_mode'],physical_coverage=metrics['physical']['coverage'],
                fragments_per_performer={p:len(identities[p]) for p in sorted(set(labels.values()))},
                own_id_turns=seq['own_id_turns'],turns=seq['turns'],endpoint_returns=seq['endpoint_returns'],
                partner_swaps=seq['contact_interior']['partner'],contacts=seq['crossings'])
