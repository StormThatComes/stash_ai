"""Score labelled turns and crossings by physical cue, independent of UUID spelling."""
from collections import defaultdict
from itertools import combinations


def sequence_metrics(labels, samples, locations, decisions, excluded=(), *, timing=None, unbound=None):
    from .reference_roles import STRICT_REASONS
    face_roots = defaultdict(set)
    by_scene = defaultdict(lambda: defaultdict(dict))
    for oid,truth in labels.items():
        if oid in excluded:
            continue
        scene,cue = locations[oid]
        rows = by_scene[scene][truth].setdefault(cue,[])
        decision = decisions[oid]
        owner = decision['accepted_identity']
        if (owner and decision.get('evidence_kind') != 'person' and
                (decision.get('evidence') or {}).get('reason') in STRICT_REASONS):
            face_roots[scene,truth].add((cue,owner))
        rows.append((owner, samples[oid]['slices']))
    for (scene, cue, truth), info in (unbound or {}).items():
        by_scene[scene][truth].setdefault(cue, [(None, info.get("slices", ()))])
    counts = defaultdict(lambda: defaultdict(int))

    turn_roots = []
    returned_people = []
    missing_boundaries = 0

    def continuous(scene, a, b):
        if b != a + 1:
            return False
        if not timing:
            return True
        left, right = timing.get((scene,a),(None,None))[1], timing.get((scene,b),(None,None))[1]
        return left is None or right is None or left == right

    def episodes(scene, cues, active):
        nonlocal missing_boundaries
        # Use the nearest labelled observation before and after one contiguous
        # run of hard cues. Missing boundary labels cannot establish a result.
        first = None
        for i,cue in enumerate(cues):
            if active(cue):
                if first is None:
                    first = i
            elif first is not None:
                if first>0:
                    window = cues[first-1:i+1]
                    if all(continuous(scene,a,b) for a,b in zip(window,window[1:])):
                        yield window
                    else:
                        missing_boundaries += 1
                else:
                    missing_boundaries += 1
                first = None
        if first is not None:
            missing_boundaries += 1

    def identity(rows):
        from .scene_strategy_validation import physical_owner
        return physical_owner(owner for owner,_ in rows)

    def tagged(rows, name):
        return any(name in slices for _,slices in rows)

    middle_outcomes = dict(correct=0, withheld=0, partner=0, foreign=0, unknown_reference=0)
    crossings = correct_pairs = 0
    endpoint_returns = foreign_ids = abstentions = 0
    entrant_episodes = entrant_takeovers = 0
    reacquisition = []
    censored = wrong_returns = 0
    for scene,people in by_scene.items():
        for person,track in people.items():
            previous = missing_since = None
            for cue in sorted(track):
                current = identity(track[cue])
                if current is None:
                    if previous and missing_since is None:
                        missing_since = cue
                    continue
                if previous and previous != current:
                    counts[person]['scene_id_switches'] += 1
                if previous == current and missing_since is not None:
                    reacquisition.append(cue-missing_since)
                if missing_since is not None and current!=previous:
                    wrong_returns += 1
                previous, missing_since = current, None
            if missing_since is not None:
                censored += 1
            # A labelled arrival needs a previously visible partner and continuation.
            for cue, observations in track.items():
                if (tagged(observations, 'entrant') and cue-1 not in track and cue+1 in track
                        and any(cue-1 in other and cue in other for other in people.values())
                        and continuous(scene, cue, cue+1)):
                    entrant_episodes += 1
                    arriving = identity(observations)
                    if arriving and any(arriving == identity(other[cue-1]) for key,other in people.items()
                                        if key != person and cue-1 in other and cue not in other):
                        entrant_takeovers += 1
            for window in episodes(scene,sorted(track),lambda cue:tagged(track[cue],'back')):
                ids = [identity(track[c]) for c in window]
                counts[person]['turns'] += 1
                counts[person]['one_id_turns'] += int(None not in ids and len(set(ids))==1)
                roots_by_cue = face_roots[scene,person]
                boundaries = {cue for cue,owner in roots_by_cue if cue in (window[0],window[-1])}
                roots = {owner for cue,owner in roots_by_cue if cue in boundaries}
                if len(boundaries)==1:
                    counts[person]['one_local_root_turns'] += 1
                elif not boundaries and roots_by_cue:
                    counts[person]['remote_root_turns'] += 1
                if len(boundaries)==2:
                    turn_roots.append((scene,person,window[0],window[-1]))
                    counts[person]['anchored_turns'] += 1
                    counts[person]['face_root_conflicts'] += len(roots)>1
                    expected = next(iter(roots)) if len(roots)==1 else None
                    foreign = any(owner==expected for partner,other in people.items() if partner!=person
                                  for cue in window for owner,_ in other.get(cue,()))
                    counts[person]['own_id_turns'] += int(expected is not None and all(p==expected for p in ids) and not foreign)
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
            for window in episodes(scene,common,lambda c:any(tagged(people[p][c],'entangled') for p in (a,b))):
                crossings += 1
                expected = [identity(people[p][window[0]]) for p in (a,b)]
                result = [identity(people[p][window[-1]]) for p in (a,b)]
                recovered = None not in expected and len(set(expected))==2 and expected==result
                # A measured swap or mixed projection during contact cannot count
                # as success; abstention inside contact is allowed.
                endpoint_returns += int(recovered)
                if None not in expected and len(set(expected)) == 2:
                    returned_people.extend((scene,p,window[0],window[-1]) for p,own,last in
                                           zip((a,b),expected,result) if own==last)
                wrong = withheld = False
                for person, own in zip((a,b), expected):
                    middle = [owner for cue in window[1:-1] for owner,_ in people[person][cue]]
                    wrong |= own is not None and any(owner is not None and owner != own for owner in middle)
                    withheld |= any(identity(people[person][cue]) is None for cue in window[1:-1])
                if None in expected or len(set(expected)) != 2:
                    outcome = 'unknown_reference'
                else:
                    swapped = any(owner == expected[1-i] for i,p in enumerate((a,b))
                                  for cue in window[1:-1] for owner,_ in people[p][cue])
                    outcome = 'partner' if swapped else 'foreign' if wrong else 'withheld' if withheld else 'correct'
                middle_outcomes[outcome] += 1
                foreign_ids += int(wrong)
                abstentions += int(withheld)
                recovered = recovered and not wrong
                correct_pairs += int(recovered)
                for person in (a,b):
                    counts[person]['crossings'] += 1
                    counts[person]['recovered_crossings'] += int(recovered)
    def macro(success,total):
        rates = [c.get(success,0)/c[total] for c in counts.values() if c.get(total)]
        return sum(rates)/len(rates) if rates else None
    return dict(entrant_episodes=entrant_episodes, entrant_takeovers=entrant_takeovers,
                scene_id_switches=sum(c['scene_id_switches'] for c in counts.values()),
                unrecovered_at_end=censored, wrong_reacquisitions=wrong_returns, reacquired_episodes=len(reacquisition), reacquisition_cues=sum(reacquisition),
                max_reacquisition_cues=max(reacquisition, default=0), returned_people=sorted(returned_people), rooted_turns=sorted(turn_roots), turns=sum(c['turns'] for c in counts.values()),
                one_id_turns=sum(c['one_id_turns'] for c in counts.values()),
                id_switches=sum(c['id_switches'] for c in counts.values()),
                crossings=crossings,contact_interior=middle_outcomes,both_recovered=correct_pairs,endpoint_returns=endpoint_returns,
                foreign_id_crossings=foreign_ids,abstained_crossings=abstentions,
                unscored_episodes=missing_boundaries,
                anchored_turns=sum(c['anchored_turns'] for c in counts.values()),
                one_local_root_turns=sum(c['one_local_root_turns'] for c in counts.values()),
                remote_root_turns=sum(c['remote_root_turns'] for c in counts.values()),
                own_id_turns=sum(c['own_id_turns'] for c in counts.values()),
                macro_own_id_turn_recovery=macro('own_id_turns','anchored_turns'),
                macro_turn_recovery=macro('one_id_turns','turns'),
                macro_crossing_recovery=macro('recovered_crossings','crossings'),
                per_performer={p:dict(c) for p,c in sorted(counts.items())})
