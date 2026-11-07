"""Bounded complete-link alternatives for face-rooted tracks."""
def select(tracks, settings, scores, rank, shot_of):
    from .body_identity import independent
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
    maximum = max(8, settings['attach_min_support'])
    best, best_rank = [], ()
    for seed in seeds:
        selected, roots, shots = [], set(), set()
        for item in (seed, *ordered):
            start, end, row = item
            shot = shot_of(row)
            if shot in shots or start in roots or end in roots:
                continue
            if any(scores.face_similarity(row['_vector'], other['_vector']) < settings['cohesion_threshold']
                   for _, _, other in selected):
                continue
            selected.append(item)
            roots.update((start,end))
            shots.add(shot)
            if len(selected) == maximum:
                break
        ids = {r['id'] for r in independent([item[2] for item in selected],settings)}
        selected = [item for item in selected if item[2]['id'] in ids]
        priority = (len(selected), tuple(sorted((rank(item) for item in selected), reverse=True)))
        if priority > best_rank:
            best, best_rank = selected, priority
    return sorted(best,key=lambda item:(item[2]['scan'],item[2]['cue'],item[2]['id']))
