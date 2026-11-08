"""Bounded complete-link alternatives for face-rooted tracks."""
VIEW_POLICY = 'independent-held-out-perspectives-v6'


def select(tracks, settings, scores, rank, shot_of):
    ordered = sorted(tracks, key=rank, reverse=True)
    if not ordered:
        return []
    # Try high-quality starting points from different shots, including later shots.
    starts = {}
    for item in ordered:
        starts.setdefault(shot_of(item[2]), item)
    seeds = list(starts.values())
    if len(seeds) > 8:
        seeds = [seeds[round(i*(len(seeds)-1)/7)] for i in range(8)]
    return select_ordered(ordered, seeds, settings, scores, rank, shot_of)


def select_ordered(ordered, seeds, settings, scores, rank, shot_of):
    from itertools import chain
    from .independent_evidence import body_vote, stamp, unit
    from .body_identity import quality
    maximum = max(8, settings['attach_min_support'])
    best, best_rank = [], ()
    for seed in seeds:
        selected, roots, shots = [], set(), set()
        pixels, cues, hashes = set(), set(), []
        for item in chain((seed,), ordered):
            start, end, row = item
            shot = shot_of(row)
            if shot in shots or start in roots or end in roots:
                continue
            vote = body_vote(row)
            pixel, cue = unit(vote), stamp(vote)
            raw_hash = quality(row).get('appearance_hash')
            hashed = int(raw_hash,16) if raw_hash else None
            if pixel in pixels or cue in cues or (hashed is not None and any(
                    (hashed ^ peer).bit_count() < settings['min_appearance_distance'] for peer in hashes)):
                continue
            if any(scores.face_similarity(row['_vector'], other['_vector']) < settings['cohesion_threshold']
                   for _, _, other in selected):
                continue
            selected.append(item)
            roots.update((start,end))
            shots.add(shot)
            pixels.add(pixel)
            cues.add(cue)
            if hashed is not None:
                hashes.append(hashed)
            if len(selected) == maximum:
                break
        # Prefer a bank that also retrieves independent views beyond its own shots.
        # Cap the probe independently of a performer's total scene count.
        recovered = set()
        if len(selected)>=max(2,settings['attach_min_support']):
            for a,b,row in ordered:
                shot = shot_of(row)
                if shot in shots or shot in recovered or a in roots or b in roots:
                    continue
                vote = body_vote(row)
                if unit(vote) in pixels or stamp(vote) in cues:
                    continue
                values = [scores.face_similarity(row['_vector'],other['_vector']) for _,_,other in selected]
                if min(values)>=settings['cohesion_threshold'] and sum(values)/len(values)>=settings['match_threshold']:
                    recovered.add(shot)
                    if len(recovered)>=maximum:
                        break
        priority = (len(selected)>=max(2,settings['attach_min_support']),len(recovered),len(selected), tuple(sorted((rank(item) for item in selected), reverse=True)))
        if priority > best_rank:
            best, best_rank = selected, priority
    return sorted(best,key=lambda item:(item[2]['scan'],item[2]['cue'],item[2]['id']))


def retrieves_independent_witness(kept, candidates, settings, scores, shot_of):
    """Certify an additional mode on pixels and shot roots outside its retained bank."""
    from .independent_evidence import body_vote, stamp, unit
    roots = {root for a,b,_ in kept for root in (a,b)}
    shots = {shot_of(row) for _,_,row in kept}
    pixels = {unit(body_vote(row)) for _,_,row in kept}
    cues = {stamp(body_vote(row)) for _,_,row in kept}
    for a,b,row in candidates:
        vote = body_vote(row)
        if a in roots or b in roots or shot_of(row) in shots or unit(vote) in pixels or stamp(vote) in cues:
            continue
        values = [scores.face_similarity(row['_vector'],other['_vector']) for _,_,other in kept]
        if min(values)>=settings['cohesion_threshold'] and sum(values)/len(values)>=settings['match_threshold']:
            return True
    return False


def verified_extra(kept, candidates, settings, scores, shot_of):
    if retrieves_independent_witness(kept,candidates,settings,scores,shot_of):
        return kept
    if len(kept)>max(2,settings['attach_min_support']):
        for i in range(len(kept)):
            bank = kept[:i]+kept[i+1:]
            if retrieves_independent_witness(bank,[kept[i]],settings,scores,shot_of):
                return bank
    # A rare look with exactly the required support still has a held-out
    # evaluation: freeze all other shots for each query, then refit the full
    # bank only when every independent query is recovered.
    if len(kept)>=max(2,settings['attach_min_support']) and all(
            retrieves_independent_witness(kept[:i]+kept[i+1:],[item],settings,scores,shot_of)
            for i,item in enumerate(kept)):
        return kept
    return []


def modes(tracks, settings, scores, rank, shot_of, maximum=None):
    """Keep additional modes only for previously uncovered, independently rooted looks."""
    remaining = list(tracks)
    index = 0
    while remaining and (maximum is None or index < maximum):
        kept = select(remaining,settings,scores,rank,shot_of)
        if not kept or index and len(kept)<max(2,settings['attach_min_support']):
            break
        original = kept
        if index>=3:
            kept = verified_extra(kept,remaining,settings,scores,shot_of)
        if kept:
            yield kept
        kept = original
        index += 1
        used = {root for a,b,_ in kept for root in (a,b)}
        def uncovered(item):
            if item[0] in used or item[1] in used:
                return False
            values = [scores.face_similarity(item[2]['_vector'],r['_vector']) for _,_,r in kept]
            return min(values)<settings['cohesion_threshold'] or sum(values)/len(values)<settings['match_threshold']
        remaining = [item for item in remaining if uncovered(item)]
        if not remaining:
            break
