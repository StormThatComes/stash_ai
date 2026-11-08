"""Scene-only body fragments and independent local context for anonymous owners."""
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from .body_identity import copresent, independent, quality, scene_bonus
from .body_origins import physical_key


def preserve_fragments(record, rows, decisions, settings, scores, guard):
    """Retain coherent near misses as provisional satellites, never training seeds."""
    groups = []
    for row in rows:
        ranked = []
        for group in groups:
            if not copresent([row],group) and guard.allows(row,group):
                mean, low = scores.compare([row],group)
                ranked.append((mean,low,group))
        ranked.sort(key=lambda item:item[0],reverse=True)
        if not ranked or ranked[0][0]<settings['match_threshold'] or ranked[0][1]<settings['cohesion_threshold']:
            groups.append([row])
        elif len(ranked)==1 or ranked[0][0]-ranked[1][0]>=settings['ambiguity_margin']:
            ranked[0][2].append(row)
    for group in groups:
        votes = independent(group,settings)
        if len(votes)<max(2,settings['attach_min_support']):
            continue
        # A rival discovered later may invalidate this proposal.
        if any(scores.compare([r],group)[1]-scores.compare([r],other)[0]<settings['ambiguity_margin']
               for r in group for other in groups if other is not group and not copresent([r],other)
               and guard.allows(r,other)):
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
    if not groups or not bonus:
        return
    from .body_seed_modes import FrozenSeeds
    banks = {decisions[g[0]['id']]['person']:g for g in groups}
    confirmed = {p for p,g in banks.items() if decisions[g[0]['id']]['status']=='accepted'}
    if not confirmed or not any(decisions[r['id']]['status']!='accepted' and
                                    quality(r).get('view_shot') is not None for r in rows):
        return
    minimum = max(2,settings['attach_min_support'],config.data['clustering']['same_scene_min_support'])
    modes = FrozenSeeds({**seeds,**banks},scores,settings)
    by_shot = {}
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
                    distance = min(abs(r['cue']-row['cue']) for r in witnesses)
                    weight = 1/(1+distance/30)
                runner = max((item[5] for item in rank if item[2] not in (p,decision['person'])),default=-1.)
                if (mean>=settings['match_threshold']-bonus*weight and low>=settings['cohesion_threshold']
                        and mean-runner>=settings['ambiguity_margin']):
                    choices.append((mean,p,runner,weight,[r['id'] for r in witnesses[:minimum]]))
        if choices and len({c[1] for c in choices})==1:
            proposals[row['id']] = max(choices)
    # Preserve a provisional fragment as a unit until every member agrees.
    for ids in components.values():
        if any(i not in proposals for i in ids) or len({proposals[i][1] for i in ids})!=1:
            for i in ids:
                proposals.pop(i,None)
    for oid,(mean,owner,runner,weight,witnesses) in proposals.items():
        decisions[oid].update(person=owner,status='accepted',role='auxiliary',reason='scene_body_attachment',
                              score=mean,runner=runner,witness_ids=witnesses)
        if mean<settings['match_threshold']:
            decisions[oid]['roster_context'] = dict(owner=owner,bonus=bonus*weight,witnesses=witnesses)
