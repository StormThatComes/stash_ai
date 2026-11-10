"""Reconcile an established rear-only core against separately face-bracketed shots."""
from collections import Counter, defaultdict
from itertools import pairwise

from .body_identity import independent
from .turn_transitions import transitions
from .turn_witnesses import TurnWitnesses


def recover(rows, decisions, seeds, retained, by_cue, geometry, metadata, signals, anchors,
            angles, owners, occupied, cuts, settings, scores, guard, scene_bonus):
    roots = {r['id']:p for p,bank in seeds.items() for r in bank}
    if not roots:
        return
    donors = {p:bank for p,bank in owners.items() if p not in seeds and p not in retained
              and len({metadata[r['id']].get('view_shot') for r in bank} - {None}) >= 2
              and any(decisions[r['id']]['reason']=='body_scene_core' for r in bank)
              and all(angles[r['id']]=='back' and not r.get('face_person') and
                      decisions[r['id']]['reason'] in ('body_scene_core','bracketed_scene_core','corroborated_rear_view')
                      for r in bank)}
    if not donors:
        return
    lookup = {r['id']:r for r in rows}
    donor_at = {r['id']:p for p,bank in donors.items() for r in bank}
    # A scene discount only nominates brackets. Acceptance below still requires
    # ordinary same-view similarities and independent roots in distinct shots.
    relaxed = dict(settings,turn_similarity_discount=settings.get('turn_similarity_discount',.06)+scene_bonus)
    confirmed = {oid for oid in anchors if decisions[oid].get('role')=='anchor'}
    edges = transitions(by_cue,geometry,metadata,signals,anchors,angles,relaxed,scores,occupied,cuts,confirmed=confirmed)
    gallery = TurnWitnesses(lookup,settings,scores)
    targets = defaultdict(set)
    bracketed = defaultdict(set)
    for start in sorted(edges.keys()-{edge[0] for edge in edges.values()}):
        lane = [start]
        while lane[-1] in edges:
            lane.append(edges[lane[-1]][0])
        positions = [i for i,oid in enumerate(lane) if oid in roots]
        for left,right in pairwise(positions):
            a,b = lane[left],lane[right]
            middle = lane[left+1:right]
            target = roots[a]
            if not middle or len(middle)>15 or roots[b]!=target or len(independent([lookup[a],lookup[b]],settings))!=2:
                continue
            source = donor_at.get(middle[0])
            if source is None or any(donor_at.get(oid)!=source for oid in middle):
                continue
            if any(metadata[k].get('query_only') or decisions[k].get('view_exclusion') for k in middle):
                continue
            targets[source].add(target)
            bracketed[source,target].update(middle)
            for oid in middle:
                gallery.add(source,target,lookup[oid],(a,b),metadata[oid]['view_shot'])

    occupied_by = {p:{r['cue'] for r in bank} for p,bank in owners.items()}
    margin = max(.025,settings['ambiguity_margin'])
    assignments = defaultdict(set)
    for row in rows:
        owner = decisions[row['id']].get('person')
        if owner is not None:
            assignments[owner,row['cue']].add(row['id'])
    proposals = {}
    for source,options in targets.items():
        if len(options)!=1:
            continue
        target = next(iter(options))
        bank = gallery.by_view[source,target]
        if occupied_by[source] & occupied_by[target] or len({shot for _,_,shot in bank})<2:
            continue
        # Select disjoint face brackets from different shots with coherent rear
        # examples. No recovered member can become a root for this operation.
        pairs = []
        for i,(a,ends,shot) in enumerate(bank):
            for b,other,another in bank[i+1:]:
                if (shot==another or len(independent([lookup[k] for k in (*ends,*other)],settings))!=4
                        or len(independent([a,b],settings))!=2
                        or scores.face_similarity(a['_vector'],b['_vector'])<settings['cohesion_threshold']):
                    continue
                pairs.append(((a,ends,shot),(b,other,another)))
        if not pairs:
            continue
        proofs = {}
        for row in donors[source]:
            oid = row['id']
            if (metadata[oid].get('query_only') or decisions[oid].get('view_exclusion')
                    or decisions[oid]['person'] not in (source,target) or assignments[target,row['cue']]-{oid}
                    or not guard.allows(row,owners[target],target)):
                continue
            shot = metadata[oid].get('view_shot')
            if shot is None:
                continue
            best = None
            for pair in pairs:
                external = [r for r,_,s in pair if s!=shot]
                # The query is not a support vote. Similar-looking crops may be
                # attached, while an exact repeat cannot certify its own label.
                pixel = metadata[oid].get('pixel_digest')
                if not external or any(r['id']==oid or (pixel and pixel==metadata[r['id']].get('pixel_digest'))
                                       for r in external):
                    continue
                low = min(scores.face_similarity(row['_vector'],r['_vector']) for r in external)
                ids = tuple(r['id'] for r in external)
                record = (low, ids, tuple(k for _,ends,_ in pair for k in ends))
                if best is None or record>best:
                    best = record
            if best is None or best[0]<settings['match_threshold']:
                continue
            rival = max((scores.face_similarity(row['_vector'],peer['_vector'])
                         for other,refs in owners.items() if other not in (source,target)
                         and row['cue'] not in occupied_by[other] for peer in refs
                         if anchors.get(peer['id'])==other),default=-1.)
            if best[0]-rival<margin:
                continue
            proofs[oid] = dict(view_score=best[0],view_witnesses=list(best[1]),view_roots=list(best[2]),
                               view_runner=rival,required_margin=margin,original_core=source)
        if len(proofs)<len(donors[source]):
            # Partial moves require their own bracket as well as the external
            # appearance checks above. Never move unsupported satellites.
            verified = {}
            starts = set(proofs)-{edge[0] for oid,edge in edges.items() if oid in proofs}
            for first in starts:
                part = [first]
                while part[-1] in edges and edges[part[-1]][0] in proofs:
                    part.append(edges[part[-1]][0])
                if (set(part)<=bracketed[source,target] and
                        len(independent([lookup[k] for k in part],settings))>=max(2,settings['attach_min_support'])):
                    verified.update((k,proofs[k]) for k in part)
            proofs = verified
        if proofs:
            proposals[source] = target,proofs
    # Two cores cannot simultaneously claim a target in the same physical cue.
    claims = Counter((target,r['cue']) for target,proofs in proposals.values() for r in (lookup[k] for k in proofs))
    for source,(target,proofs) in proposals.items():
        if any(claims[target,lookup[k]['cue']]!=1 for k in proofs):
            continue
        for oid,proof in proofs.items():
            decisions[oid].update(person=target,status='accepted',role='auxiliary',
                                  reason='corroborated_rear_view',**proof)
