"""Scene-only body fragments and independent local context for anonymous owners."""
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from .body_identity import copresent, independent, quality, scene_bonus
from .body_origins import physical_key


def preserve_fragments(record, rows, decisions, settings, scores, guard):
    """Retain coherent near misses as provisional satellites, never training seeds."""
    # Retain the strongest neighbour of each sample. Store O(n) edges even
    # when many observations in one scene look alike. Freeze all rivals before pruning.
    groups = {r['id']:[r] for r in rows}
    owners = {key:key for key in groups}
    nearest = {}
    ordered = sorted(rows, key=lambda r:r['id'])
    for i, row in enumerate(ordered):
        for peer in ordered[i+1:]:
            a, b = row['id'], peer['id']
            if copresent([row],groups[b]) or not guard.allows(row,groups[b]):
                continue
            score, minimum = scores.compare([row],groups[b])
            if score < settings['match_threshold'] or minimum < settings['cohesion_threshold']:
                continue
            edge = (-score, a, b)
            for oid in (a,b):
                if oid not in nearest or edge < nearest[oid]:
                    nearest[oid] = edge
    for _, a, b in sorted(set(nearest.values())):
        first, second = owners[a], owners[b]
        if first == second:
            continue
        left, right = groups[first], groups[second]
        if copresent(left,right) or not all(guard.allows(r,right) for r in left):
            continue
        mean, low = scores.compare(left,right)
        if mean >= settings['match_threshold'] and low >= settings['cohesion_threshold']:
            left.extend(right)
            for r in right:
                owners[r['id']] = first
            del groups[second]
    for original in groups.values():
        group = [r for r in original if all(
            scores.compare([r],original)[1]-scores.compare([r],other)[0]>=settings['ambiguity_margin']
            for other in groups.values() if other is not original and not copresent([r],other)
            and guard.allows(r,other))]
        group.sort(key=lambda r:r['id'])
        votes = independent(group,settings)
        if len(votes)<max(2,settings['attach_min_support']):
            continue
        owner = str(uuid5(NAMESPACE_URL,'near-body-fragment:'+physical_key(record,group[0])))
        for row in group:
            prior = decisions[row['id']]
            if prior.get('reason')=='ambiguous_or_insufficient_face_body_bank':
                prior.update(person=owner,role='auxiliary',status='provisional',reason='anonymous_body_track',
                             fragment_support=len(votes))


def attach_with_scene_support(rows, groups, seeds, decisions, config, scores, guard):
    """Use other shots of accepted body-born cores as a frozen scene roster."""
    settings = config.data['sapiensid']['identities']
    bonus = scene_bonus(config.data)
    if not groups:
        return
    from .body_seed_modes import FrozenSeeds
    banks = {decisions[g[0]['id']]['person']:g for g in groups}
    confirmed = {p for p,g in banks.items() if decisions[g[0]['id']]['status']=='accepted'}
    if not confirmed:
        return
    from .contact_anchors import body_pair_queries, joint_body_claims
    pairs = body_pair_queries(rows,decisions)
    use_roster = bonus and any(decisions[r['id']]['status']!='accepted' and
                               quality(r).get('view_shot') is not None for r in rows)
    if not pairs and not use_roster:
        return
    modes = FrozenSeeds({**seeds,**banks},scores,settings)
    if pairs:
        positive = confirmed | set(seeds)
        rankings = {}
        for row in pairs:
            rankings[row['id']] = [(mean,low,p,mode,good and p in positive,rival)
                                    for mean,low,p,mode,good,rival in modes.rank(row,guard)]
        joint_body_claims(rows,decisions,rankings,settings)
    if not use_roster:
        return
    minimum = max(2,settings['attach_min_support'],config.data['clustering']['same_scene_min_support'])
    from .roster_presence import Presence, entrant_score
    from .crowd_context import pair
    import json
    shots = {r['cue']:quality(r).get('view_shot') for r in rows}
    lookup = {r['id']:r for r in rows}
    by_shot = {}
    presence = {}
    entrant_banks = {}
    proposals = {}
    components = defaultdict(list)
    for row in rows:
        decision = decisions[row['id']]
        if decision['status']=='accepted' or decision.get('reason') not in (
                'body_scene_core','body_anchor_quality','body_anchor_keypoint_rules','body_anchor_pixels',
                'ambiguous_or_insufficient_face_body_bank','anonymous_body_track'):
            continue
        q = quality(row)
        shot = q.get('view_shot')
        if decision.get('person'):
            components[decision['person']].append(row['id'])
        # This local roster is a faceless recovery route. A measured face uses
        # the ordinary face veto banks, including all external witness history.
        if (shot is None or row.get('face_person') or row.get('_face_vector') is not None
                or q.get('query_only') or q.get('expected_people',0)>q.get('detected_people',1)):
            continue
        rank = modes.rank(row,guard)
        choices = []
        for _,_,p,_,_,_ in rank:
            if p not in confirmed or p==decision['person']:
                continue
            for index,mode in enumerate(modes.modes[p]):
                key = p,index,shot
                if key not in by_shot:
                    by_shot[key] = independent((r for r in mode if quality(r).get('view_shot') not in (None,shot)),settings)
                witnesses = by_shot[key]
                if len(witnesses)<minimum:
                    continue
                mean,low = scores.compare([row],witnesses)
                weight = 1.
                if settings.get('scene_bonus_mode')=='evidence':
                    if key not in presence:
                        presence[key] = Presence(witnesses,shots)
                    weight,_ = presence[key].weight(row)
                # An entrant may have independent body evidence of its own.
                # Same-shot members of this query fragment cannot vote for themselves.
                own = decision.get('person'),shot
                if own not in entrant_banks:
                    entrant_banks[own] = independent((r for r in banks.get(own[0],())
                                                     if shots[r['cue']] not in (None,shot)),settings)
                entrant = entrant_score(scores,row,entrant_banks[own],settings,bonus)
                runner = max(entrant,max((item[5] for item in rank if item[2] not in (p,decision['person'])),default=-1.))
                if (mean>=settings['match_threshold']-bonus*weight and low>=settings['cohesion_threshold']
                        and mean-runner>=settings['ambiguity_margin']):
                    choices.append((mean,p,runner,weight,[r['id'] for r in witnesses[:minimum]],entrant))
        if choices and len({c[1] for c in choices})==1:
            proposals[row['id']] = max(choices)
    # Preserve a provisional fragment as a unit until every member agrees.
    for ids in components.values():
        if any(i not in proposals for i in ids) or len({proposals[i][1] for i in ids})!=1:
            allowed = set()
            if settings.get('tracklet_matching'):
                from .body_tracklets import split_members
                def occupied(row, target):
                    geom = json.loads(row['geometry'])
                    return any(peer['cue']==row['cue'] and peer['id']!=row['id'] and
                               pair([geom,json.loads(peer['geometry'])]) is not None
                               for peer in modes.seeds.get(target,()))
                targets = {oid:proposals[oid][1] for oid in ids if oid in proposals}
                allowed = split_members([lookup[i] for i in ids],targets,occupied,settings)
            for i in set(ids)-allowed:
                proposals.pop(i,None)
    for oid,(mean,owner,runner,weight,witnesses,entrant) in proposals.items():
        decisions[oid].update(person=owner,status='accepted',role='auxiliary',reason='scene_body_attachment',
                              score=mean,runner=runner,entrant_score=entrant,threshold=settings['match_threshold']-bonus*weight,
                              required_margin=settings['ambiguity_margin'],witness_ids=witnesses)
        if mean<settings['match_threshold']:
            decisions[oid]['roster_context'] = dict(owner=owner,bonus=bonus*weight,witnesses=witnesses)
