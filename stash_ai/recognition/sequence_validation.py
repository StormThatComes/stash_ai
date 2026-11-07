"""Score labelled turns and crossings by physical cue, independent of UUID spelling."""
from collections import defaultdict
from itertools import combinations


def sequence_metrics(labels, samples, locations, decisions, excluded=()):
    by_scene = defaultdict(lambda: defaultdict(dict))
    for oid,truth in labels.items():
        if oid in excluded:
            continue
        scene,cue = locations[oid]
        rows = by_scene[scene][truth].setdefault(cue,[])
        rows.append((decisions[oid]['accepted_identity'], samples[oid]['slices']))
    counts = defaultdict(lambda: defaultdict(int))

    def episodes(cues, active):
        # Use the nearest labelled observation before and after one contiguous
        # run of hard cues. Missing boundary labels cannot establish a result.
        first = None
        for i,cue in enumerate(cues):
            if active(cue):
                if first is None:
                    first = i
            elif first is not None:
                if first>0:
                    yield cues[first-1:i+1]
                first = None

    def identity(rows):
        owners = {owner for owner,_ in rows}
        return owners.pop() if len(owners)==1 else None

    def tagged(rows, name):
        return any(name in slices for _,slices in rows)

    crossings = correct_pairs = 0
    for people in by_scene.values():
        for person,track in people.items():
            for window in episodes(sorted(track),lambda cue:tagged(track[cue],'back')):
                ids = [identity(track[c]) for c in window]
                counts[person]['turns'] += 1
                counts[person]['one_id_turns'] += int(None not in ids and len(set(ids))==1)
                assigned = [p for p in ids if p is not None]
                counts[person]['id_switches'] += sum(a!=b for a,b in zip(assigned,assigned[1:]))
        # Form pairs only in annotated entangled cues (usually two people).
        present = defaultdict(list)
        for person,track in people.items():
            for cue in track:
                present[cue].append(person)
        pairs = {pair for cue,owners in present.items() if any(tagged(people[p][cue],'entangled') for p in owners)
                 for pair in combinations(sorted(owners),2)}
        for a,b in pairs:
            common = sorted(people[a].keys() & people[b].keys())
            for window in episodes(common,lambda c:any(tagged(people[p][c],'entangled') for p in (a,b))):
                crossings += 1
                expected = [identity(people[p][window[0]]) for p in (a,b)]
                result = [identity(people[p][window[-1]]) for p in (a,b)]
                recovered = None not in expected and len(set(expected))==2 and expected==result
                # A measured swap or mixed projection during contact cannot count
                # as success; abstention inside contact is allowed.
                for person,partner in ((a,expected[1]),(b,expected[0])):
                    if partner is not None and any(owner==partner for c in window[1:-1]
                                                   for owner,_ in people[person][c]):
                        recovered = False
                correct_pairs += int(recovered)
                for person in (a,b):
                    counts[person]['crossings'] += 1
                    counts[person]['recovered_crossings'] += int(recovered)
    def macro(success,total):
        rates = [c.get(success,0)/c[total] for c in counts.values() if c.get(total)]
        return sum(rates)/len(rates) if rates else None
    return dict(turns=sum(c['turns'] for c in counts.values()),
                one_id_turns=sum(c['one_id_turns'] for c in counts.values()),
                id_switches=sum(c['id_switches'] for c in counts.values()),
                crossings=crossings,both_recovered=correct_pairs,
                macro_turn_recovery=macro('one_id_turns','turns'),
                macro_crossing_recovery=macro('recovered_crossings','crossings'),
                per_performer={p:dict(c) for p,c in sorted(counts.items())})
