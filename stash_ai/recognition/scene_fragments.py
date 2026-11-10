"""Scene-only body fragments and independent local context for anonymous owners."""
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from .body_identity import copresent, independent, quality, scene_bonus
from .body_origins import physical_key


def plausible_pairs(rows, threshold):
    """Tile singleton nominations. Re-score every possible hit with the production scorer."""
    from .nomination_bounds import ball, upper
    import numpy as np
    width = 16
    blocks = []
    for start in range(0,len(rows),width):
        data = np.stack([r['_vector'] for r in rows[start:start+width]])
        blocks.append(ball(data))
    from functools import lru_cache
    # Bound auxiliary matrices even for an unusually dense scene.
    block_bytes = width * len(rows[0]['_vector']) * 8 if rows else 1

    @lru_cache(maxsize=max(1, min(len(blocks), 8 * 1024**2 // block_bytes)))
    def matrix(index):
        return np.stack([r['_vector'] for r in rows[index*width:(index+1)*width]]).astype('float64')

    for i,first in enumerate(blocks):
        left = matrix(i)
        for j in range(i,len(blocks)):
            second = blocks[j]
            # Conservative float32 dot-product error for unit embeddings.
            error = 2*left.shape[1]*np.finfo(np.float32).eps
            if upper(first,second)+error < threshold:
                continue
            right = left if i == j else matrix(j)
            hits = left @ right.T >= threshold-error
            if i==j:
                hits = np.triu(hits,1)
            for a,b in zip(*np.nonzero(hits)):
                yield rows[i*width+a],rows[j*width+b]


def reconcile_neighbours(groups,settings,scores,guard):
    """Join mutually separated complete-link components; keep only two scores per component."""
    if not any(len(group)>1 for group in groups.values()):
        return
    import numpy as np
    from .nomination_bounds import ball, upper
    best = defaultdict(list)
    keys = sorted(groups)
    bounds = {k:ball(np.stack([r['_vector'] for r in groups[k]])) for k in keys}
    error = 2*len(groups[keys[0]][0]['_vector'])*np.finfo(np.float32).eps
    rival_floor = settings['match_threshold']-settings['ambiguity_margin']
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if upper(bounds[a],bounds[b])+error<rival_floor:
                continue
            left,right = groups[a],groups[b]
            if copresent(left,right) or not all(guard.allows(r,right) for r in left) or not all(
                    guard.allows(r,left) for r in right):
                continue
            mean,low = scores.compare(left,right)
            for own,peer in ((a,b),(b,a)):
                best[own].append((mean,peer,low))
                best[own].sort(key=lambda item:(-item[0],item[1]))
                del best[own][2:]
    def winner(key):
        options = best[key]
        if not options:
            return None
        mean,peer,low = options[0]
        rival = options[1][0] if len(options)>1 else -1.
        if (mean>=settings['match_threshold'] and low>=settings['cohesion_threshold']
                and mean-rival>=settings['ambiguity_margin']):
            return peer
    for a in keys:
        b = winner(a)
        if b and a<b and winner(b)==a:
            groups[a].extend(groups.pop(b))


def preserve_fragments(record, rows, decisions, settings, scores, guard):
    """Retain coherent near misses as provisional satellites, never training seeds."""
    # Retain the strongest neighbour of each sample. Store O(n) edges even
    # when many observations in one scene look alike. Freeze all rivals before pruning.
    groups = {r['id']:[r] for r in rows}
    owners = {key:key for key in groups}
    nearest = {}
    ordered = sorted(rows, key=lambda r:r['id'])
    for row,peer in plausible_pairs(ordered,max(settings['match_threshold'],settings['cohesion_threshold'])):
        a, b = row['id'], peer['id']
        if copresent([row],groups[b]) or not guard.allows(row,groups[b]) or not guard.allows(peer,groups[a]):
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
        if (copresent(left,right) or not all(guard.allows(r,right) for r in left)
                or not all(guard.allows(r,left) for r in right)):
            continue
        mean, low = scores.compare(left,right)
        if mean >= settings['match_threshold'] and low >= settings['cohesion_threshold']:
            left.extend(right)
            for r in right:
                owners[r['id']] = first
            del groups[second]
    reconcile_neighbours(groups,settings,scores,guard)
    for original in groups.values():
        if len(original)<max(2,settings['attach_min_support']):
            continue
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


def attach_with_scene_support(rows, groups, seeds, decisions, config, scores, guard, *, cut_starts=(), entrants=None):
    """Use other shots of accepted body-born cores as a frozen scene roster."""
    settings = config.data['sapiensid']['identities']
    bonus = scene_bonus(config.data)
    if not groups and not seeds:
        return
    from .body_seed_modes import FrozenSeeds
    banks = {decisions[g[0]['id']]['person']:g for g in groups}
    confirmed = {p for p,g in banks.items() if decisions[g[0]['id']]['status']=='accepted'}
    if settings.get('tracklet_matching'):
        for person, witnesses in seeds.items():
            existing = {r['id']:r for r in banks.get(person,())}
            existing.update((r['id'],r) for r in witnesses)
            banks[person] = list(existing.values())
        confirmed.update(seeds)
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
    joint_options = defaultdict(dict)
    competitors = {}
    fragment_rows = defaultdict(list)
    for row in rows:
        person = decisions[row['id']].get('person')
        if person and not quality(row).get('query_only'):
            fragment_rows[person].append(row)
    components = defaultdict(list)
    for row in rows:
        decision = decisions[row['id']]
        if decision['status']=='accepted' or decision.get('reason') not in (
                'body_scene_core','body_anchor_quality','body_anchor_keypoint_rules','body_anchor_pixels',
                'ambiguous_or_insufficient_face_body_bank','anonymous_body_track','body_group_ambiguous'):
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
        competitors[row['id']] = {v[2]:v[5] for v in rank}
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
                        presence[key] = Presence(witnesses,shots,cut_starts)
                    weight,_ = presence[key].weight(row)
                # An entrant may have independent body evidence of its own.
                # Same-shot members of this query fragment cannot vote for themselves.
                own = decision.get('person'),shot
                if own not in entrant_banks:
                    entrant_banks[own] = independent((r for r in fragment_rows.get(own[0],())
                                                     if shots[r['cue']] not in (None,shot)),settings)
                entrant = entrant_score(scores,row,entrant_banks[own],settings,bonus)
                if entrants:
                    entrant = entrants.challenge(row,witnesses,scores,entrant)
                runner = max(entrant,max((item[5] for item in rank if item[2] not in (p,decision['person'])),default=-1.))
                if weight and low>=settings['cohesion_threshold']:
                    probe = decision.get('scene_probe')
                    if probe is None or mean>probe['score']:
                        decision['scene_probe'] = dict(owner=p,score=mean,runner=runner,weight=weight)
                if mean>=settings['match_threshold']-bonus*weight and low>=settings['cohesion_threshold']:
                    option = (mean,p,runner,weight,[r['id'] for r in witnesses[:minimum]],entrant)
                    previous = joint_options[row['id']].get(p)
                    if previous is None or option>previous:
                        joint_options[row['id']][p] = option
                    if mean-runner>=settings['ambiguity_margin']:
                        choices.append(option)
        if choices and len({c[1] for c in choices})==1:
            proposals[row['id']] = max(choices)
    # Competing roster claims in one physical cue must share one assignment.
    from .joint_claims import solve
    by_cue = defaultdict(list)
    for row in rows:
        by_cue[row['cue']].append(row)
    margin = max(1e-6,settings['ambiguity_margin'])
    for members in by_cue.values():
        if len(members)!=2 or not all(r['id'] in competitors for r in members):
            continue
        if pair([json.loads(r['geometry']) for r in members]) is None:
            continue
        ids = [r['id'] for r in members]
        if not (joint_options[ids[0]].keys() & joint_options[ids[1]].keys()):
            continue
        domains = []
        for oid in ids:
            options = joint_options[oid]
            abstain = max([settings['match_threshold']-bonus-margin,*(o[5] for o in options.values())])
            domains.append([(None,abstain),*((p,o[0]) for p,o in options.items())])
            proposals.pop(oid,None)
        result = solve(domains,lambda i,a,j,b:a[0] is None or a[0]!=b[0],margin)
        if result is None:
            continue
        resolved = {}
        for i,(p,score) in enumerate(result):
            if p is None:
                continue
            oid = ids[i]
            _,_,_,weight,witnesses,entrant = joint_options[oid][p]
            explained = result[1-i][0]
            runner = max(entrant,max((v for k,v in competitors[oid].items()
                if k not in (p,decisions[oid]['person'],explained)),default=-1.))
            if score-runner<margin:
                break
            resolved[oid] = (score,p,runner,weight,witnesses,entrant)
        else:
            proposals.update(resolved)
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
