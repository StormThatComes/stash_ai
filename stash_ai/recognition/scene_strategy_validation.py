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
            data['sapiensid']['identities'].update(tracklet_matching=tracks, endpoint_matching=tracks,
                scene_bonus_mode='evidence' if mode=='evidence' else 'fixed',
                scene_match_bonus=0. if mode=='off' else positive)
            result[f"tracks-{int(tracks)}-{mode}"] = replace(config,data=data)
    for mode in ('off', 'fixed', 'evidence'):
        tracks = result[f'tracks-1-{mode}']
        data = plain(tracks.data)
        data['sapiensid']['identities']['endpoint_matching'] = False
        result[f'paths-only-{mode}'] = replace(config, data=data)
    for mode in ('off','fixed','evidence'):
        off = result[f'tracks-0-{mode}']
        data = plain(off.data)
        data['sapiensid']['identities']['endpoint_matching'] = True
        result[f'endpoints-only-{mode}'] = replace(config,data=data)
    return result


STRATEGY_KEYS = tuple('sapiensid.identities.'+key for key in (
    'tracklet_matching','endpoint_matching','scene_bonus_mode','scene_match_bonus'))


def training_profiles(config):
    from .merge_report_policy import parameter
    for name,variant in variants(config).items():
        yield name,{k:parameter(variant.data,k) for k in STRATEGY_KEYS}


def tuning_profiles(config, families):
    from .merge_report_policy import parameter
    # Alternate families so a short remaining trial budget is shared fairly.
    for key in ('sapiensid.identities.match_threshold','sapiensid.identities.track_similarity_threshold',
                'recognition.match_threshold'):
        for step in (-.02,.02):
            for name,trial,patch in families:
                value = max(0.,min(1.,parameter(config.data,key)+step))
                yield name,trial,dict(patch,**{key:value})


def physical_owner(values):
    owners = set(values) - {None}
    return next(iter(owners)) if len(owners) == 1 else None


def physical_predictions(labels,locations,ids,decisions):
    seen = defaultdict(set)
    for oid in ids:
        owners = seen[(*locations[oid],labels[oid])]
        value = decisions[oid]['accepted_identity']
        if value is not None:
            owners.add(value)
    return {key:physical_owner(owners) for key,owners in seen.items()}


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
    return dict(tracklet_matching=settings['tracklet_matching'],endpoint_matching=settings.get('endpoint_matching'),
                physical_false_joins=metrics['physical']['false_merge_pairs'],scene_match_bonus=settings['scene_match_bonus'],
                scene_bonus_mode=settings['scene_bonus_mode'],physical_coverage=metrics['physical']['coverage'],
                slices={name:{k:values[k] for k in ('observations','coverage','extra_fragments','false_merge_pairs')}
                        for name,values in metrics['physical']['slices'].items()},
                fragments_per_performer={p:len(identities[p]) for p in sorted(set(labels.values()))},
                own_id_turns=seq['own_id_turns'],turns=seq['turns'],endpoint_returns=seq['endpoint_returns'],
                partner_swaps=seq['contact_interior']['partner'],contacts=seq['crossings'])
