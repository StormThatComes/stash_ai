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
    # Keep paths/endpoints identical in the four modality controls. Screen
    # them first even when the remaining trial budget is small.
    controls = {}
    for body in ('off', 'evidence'):
        name = f'tracks-1-{body}'
        for face in (False, True):
            data = plain(result[name].data)
            data['clustering']['same_scene_match_bonus'] = (
                config.data['clustering']['same_scene_match_bonus'] or .03) if face else 0.
            label = name if face else name+'-noface'
            controls[label] = replace(config,data=data)
    for body_first in (False,True):
        original = result['tracks-1-evidence']
        data = plain(original.data)
        data['sapiensid']['identities']['body_first_matching'] = body_first
        controls['body-first' if body_first else 'face-first'] = replace(original,data=data)
    return {**controls, **{name:v for name,v in result.items() if name not in controls}}


STRATEGY_KEYS = tuple('sapiensid.identities.'+key for key in (
    'tracklet_matching','endpoint_matching','scene_bonus_mode','body_first_matching','scene_match_bonus')) + ('clustering.same_scene_match_bonus',)


def training_profiles(config):
    from .merge_report_policy import parameter
    for name,variant in variants(config).items():
        yield name,{k:parameter(variant.data,k) for k in STRATEGY_KEYS}


def measured_context(settings, decisions, expected, forbidden):
    """Return strict feasible boundaries for labelled roster returns and exclusions."""
    from .threshold_boundaries import at_or_above, at_or_below
    positive, negative = [], []
    for row in decisions:
        probe = row.get('scene_probe')
        if not probe or probe['weight']<=0:
            continue
        owner, oid = probe['owner'], row['observation']
        need = (settings['match_threshold']-probe['score'])/probe['weight']
        gap = probe['score']-probe['runner']
        if owner in forbidden.get(oid,()):
            negative.append((need,gap))
        elif owner == expected.get(oid):
            positive.append((need,gap))
    if not positive:
        return []
    bonus = at_or_above(max(0., max(need for need,_ in positive)))
    margin = at_or_below(min(1., min(gap for _,gap in positive)))
    if bonus>.1 or margin<0 or any(need<=bonus and gap>=margin for need,gap in negative):
        return []
    strict = {'sapiensid.identities.scene_match_bonus':bonus,
              'sapiensid.identities.ambiguity_margin':margin}
    return [strict]


def tuning_profiles(config, families, boundaries=None):
    from .merge_report_policy import parameter
    for name,trial,patch in families:
        for update in (boundaries or {}).get(trial,()):
            yield name,trial,dict(patch,**update)
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


def projection_errors(labels, locations, ids, decisions):
    """Count physical conflicts and shared ownership of distinct people in a cue."""
    by_cue = defaultdict(lambda: defaultdict(set))
    for oid in ids:
        owner = decisions[oid]['accepted_identity']
        if owner is not None:
            by_cue[locations[oid]][labels[oid]].add(owner)
    conflicts, swaps = set(), set()
    for cue, people in by_cue.items():
        by_owner = defaultdict(list)
        for truth, owners in people.items():
            if len(owners) > 1:
                conflicts.add((cue, truth))
            for owner in owners:
                for peer in by_owner[owner]:
                    swaps.add((cue, *sorted((truth, peer))))
                by_owner[owner].append(truth)
    return conflicts, swaps


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
    conflicts, swaps = projection_errors(labels, locations, ids, before)
    new_conflicts, new_swaps = projection_errors(labels, locations, ids, after)
    return dict(new_projection_conflicts=len(new_conflicts-conflicts), new_partner_claims=len(new_swaps-swaps),
                lost_correct_pairs=pairs(by_truth)-pairs(common_truth),
                new_false_joins=pairs(by_owner)-pairs(common_owner)-pairs(right_truth)+pairs(common_truth))


def strategy_summary(config,metrics,labels,locations,decisions):
    settings = config.data['sapiensid']['identities']
    identities = defaultdict(set)
    for (*_,truth),owner in physical_predictions(labels,locations,labels,decisions).items():
        if owner is not None:
            identities[truth].add(owner)
    seq = metrics['sequences']
    return dict(body_first_matching=settings.get('body_first_matching',False),tracklet_matching=settings['tracklet_matching'],endpoint_matching=settings.get('endpoint_matching'),
                face_scene_bonus=config.data['clustering']['same_scene_match_bonus'],
                physical_false_joins=metrics['physical']['false_merge_pairs'],scene_match_bonus=settings['scene_match_bonus'],
                scene_bonus_mode=settings['scene_bonus_mode'],physical_coverage=metrics['physical']['coverage'],
                slices={name:{k:values[k] for k in ('observations','coverage','extra_fragments','false_merge_pairs')}
                        for name,values in metrics['physical']['slices'].items()},
                fragments_per_performer={p:len(identities[p]) for p in sorted(set(labels.values()))},
                own_id_turns=seq['own_id_turns'],turns=seq['turns'],endpoint_returns=seq['endpoint_returns'],
                partner_swaps=seq['contact_interior']['partner'],contacts=seq['crossings'])
