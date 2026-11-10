"""Replay positive physical labels against independently frozen goal evidence."""
from collections import Counter, defaultdict
from itertools import groupby
import json

from .merge_observation_exclusions import resolve


def freeze(db, config, goals, owners, observations, tracks):
    pairs = list(observations)
    track_labels = {}
    for performer, oid in tracks:
        row = db.execute("SELECT b.scan,m.proof,m.person,m.status FROM body_members m "
                         "JOIN body_observations b ON b.id=m.observation WHERE b.id=?", (oid,)).fetchone()
        if not row:
            raise ValueError(f'Unknown retained body observation {oid}.')
        proof = json.loads(row['proof'])
        fragment = proof.get('anonymous_fragment') or (row['person'] if proof.get('reason')=='anonymous_body_track' else None)
        if not fragment:
            raise ValueError(f'Observation {oid} has no frozen anonymous track.')
        members = db.execute("SELECT b.id FROM body_observations b JOIN body_members m ON m.observation=b.id "
            "WHERE b.scan=? AND (json_extract(m.proof,'$.anonymous_fragment')=? OR "
            "(json_extract(m.proof,'$.reason')='anonymous_body_track' AND m.person=?))", (row['scan'],fragment,fragment))
        for member, in members:
            pairs.append((performer,member))
            track_labels[member] = fragment
    labels = resolve(db, config, pairs, goals['scope'], owners, positive=True)
    claimed = {}
    forbidden = {(e['set'],oid) for e in goals.get('observation_exclusions',()) for oid in e['observation_ids']}
    forbidden_scenes = {(e['set'],e['scene']) for e in goals.get('scene_exclusions',())}
    for item in labels:
        group = item['set']
        fragments = {track_labels[oid] for oid in item['observation_ids'] if oid in track_labels}
        if len(fragments)>1:
            raise ValueError('Overlapping explicit tracks have incompatible physical labels.')
        if fragments:
            item['track'] = fragments.pop()
        for oid in item['observation_ids']:
            if (claimed.setdefault(oid, group)!=group or (group,oid) in forbidden
                    or (group,item['scene']) in forbidden_scenes):
                raise ValueError('Contradictory positive and negative physical labels.')
    goals['positive_observations'] = labels
    targets = {}
    for group in goals['groups']:
        index = group['index']
        if index in claimed.values():
            targets[index] = sorted({r['id'] for m in group['members'] for r in m['observations']
                                     if r['id'] not in claimed})
            if not targets[index]:
                raise ValueError('Positive labels require independent observations of the target performer.')
    goals['positive_reference_ids'] = targets
    units = {}
    for group in goals['groups']:
        index = group['index']
        if index not in targets:
            continue
        evidence = group.get('identity_evidence') or group.get('construction_evidence',{})
        allowed = set(targets[index])
        frozen = []
        for unit in evidence.get('core_units',()):
            addresses = {r['id'] for r in unit['observations']}
            if not claimed.keys().isdisjoint(addresses):
                continue
            available = sorted(allowed & addresses)
            if available:
                frozen.append(available)
        if not frozen:
            by_cue = defaultdict(list)
            for member in group['members']:
                for row in member['observations']:
                    if row['id'] in allowed:
                        by_cue[row['scan'],row['cue']].append(row['id'])
            frozen = list(by_cue.values())
        units[index] = frozen
    goals['positive_reference_units'] = units


def assess(goals, status, observations, predicted):
    labels = goals.get('positive_observations')
    if not labels:
        return
    from .merge_goal_identity import required
    units_by_set = goals.get('positive_reference_units')
    if units_by_set is None:
        units_by_set = {index:[[oid] for oid in ids] for index,ids in goals['positive_reference_ids'].items()}
    targets = {}
    for index,units in units_by_set.items():
        votes = Counter()
        lost = mixed = 0
        for unit in units:
            assigned = {predicted.get(oid) for oid in unit if oid in observations}-{None}
            if len(assigned)==1:
                votes[assigned.pop()] += 1
            elif assigned:
                mixed += 1
            else:
                lost += 1
        quorum = required(len(units))
        winners = [owner for owner,count in votes.items() if count>=quorum]
        targets[int(index)] = (winners[0] if len(winners)==1 else None,
            dict(reference_units=len(units),reference_quorum=quorum,lost_reference_units=lost,mixed_reference_units=mixed))
    buckets = defaultdict(list)
    for item in labels:
        key = item['set'],item['scene'],item.get('track') or tuple(item['requested_observation_ids'])
        buckets[key].append(item)
    results = []
    for (index,scene,_),items in buckets.items():
        expected,detail = targets[index]
        good = wrong = gap = longest = switches = 0
        previous = gap_start = None
        longest_cues = 0
        for item in sorted(items,key=lambda item:item.get('cue',0)):
            found = {predicted.get(oid) for oid in item['observation_ids'] if oid in observations}-{None}
            correct = expected is not None and found=={expected}
            good += correct
            wrong += bool(found and expected is not None and found!={expected})
            gap = 0 if correct else gap+1
            longest = max(longest,gap)
            cue = item.get('cue',0)
            gap_start = None if correct else cue if gap_start is None else gap_start
            if gap_start is not None:
                longest_cues = max(longest_cues,cue-gap_start+1)
            if len(found)==1:
                owner = found.pop()
                switches += previous is not None and previous!=owner
                previous = owner
        result = dict(set=index,scene=scene,observations=items[0]['requested_observation_ids'],
            satisfied=good==len(items),coverage=good/len(items),sample_count=len(items),longest_unmatched_run=longest,
            wrong_owner_samples=wrong,owner_switches=switches,longest_unmatched_cue_span=longest_cues,**detail)
        if 'track' in items[0]:
            result['track'] = items[0]['track']
        results.append(result)
    status['positive_observations'] = results
    if not all(r['satisfied'] for r in results):
        status.update(admissible=False, fully_solved=False)


def measure(db, goals, expected, decisions):
    """Bound diagnostics to user-labelled cues and a small frozen scene roster."""
    import numpy as np
    from .database import encoded
    labels = {oid for item in goals['positive_observations'] for oid in item['observation_ids']}
    selected = {row['observation']:row for row in decisions if row['observation'] in labels}
    sql = ('SELECT id,scan,cue FROM body_observations WHERE id IN (SELECT value FROM json_each(?)) '
           'AND embedding IS NOT NULL ORDER BY scan,cue,id')
    for scan,items in groupby(db.execute(sql,(encoded(sorted(selected)),)),key=lambda r:r[1]):
        addresses = defaultdict(dict)
        for ref,at,owner in db.execute("SELECT b.id,b.cue,m.person FROM body_observations b JOIN body_members m "
                "ON m.observation=b.id WHERE b.scan=? AND m.status='accepted' AND m.role='anchor' "
                "AND b.embedding IS NOT NULL ORDER BY b.cue,b.id",(scan,)):
            bank = addresses[owner]
            if ref not in labels and at not in bank and len(bank)<8:
                bank[at] = ref
        refs = [ref for bank in addresses.values() for ref in bank.values()]
        vectors = {oid:np.frombuffer(blob,dtype='<f4') for oid,blob in db.execute(
            'SELECT id,embedding FROM body_observations WHERE id IN (SELECT value FROM json_each(?))',(encoded(refs),))}
        queries = {oid:cue for oid,_,cue in items}
        for oid,blob in db.execute('SELECT id,embedding FROM body_observations WHERE id IN '
                                   '(SELECT value FROM json_each(?))',(encoded(list(queries)),)):
            query = np.frombuffer(blob,dtype='<f4')
            cue = queries[oid]
            scores = {owner:max(float(query @ vectors[ref]) for at,ref in bank.items() if at!=cue)
                      for owner,bank in addresses.items() if any(at!=cue for at in bank)}
            wanted = expected.get(oid)
            ranked = sorted(((score,p) for p,score in scores.items() if p!=wanted),reverse=True)
            row = selected[oid]
            row['label_probe'] = dict(expected=wanted,score=scores.get(wanted),rival=ranked[0][1] if ranked else None,
                                      runner=ranked[0][0] if ranked else None,
                                      stage=(row.get('track_blocker') or {}).get('reason') or row['reason'],
                                      measurements=row.get('measurements',{}),**row.get('turn_evidence',{}))


def proposals(config, goals, decisions, membership):
    """Propose measured boundaries for missing labels. Replay decides feasibility."""
    from .threshold_boundaries import at_or_below
    from .merge_report_policy import parameter
    from .merge_solver_domains import normalize
    index = {oid:item['set'] for item in goals['positive_observations'] for oid in item['observation_ids']}
    options = {}
    prefix = 'sapiensid.identities.'
    for row in decisions:
        oid = row['observation']
        probe = row.get('label_probe')
        if not probe or not probe['expected'] or membership.get(oid)==probe['expected']:
            continue
        score,runner = probe['score'],probe['runner']
        if score is None or score<=0 or score-(runner if runner is not None else -1)<parameter(config.data,prefix+'ambiguity_margin'):
            continue
        if probe['stage'] in ('different_face_root','unresolved_partner','weak_face_track_conflict','person_copresence'):
            continue
        # A nearest neighbour nominates the owner. Only measurements whose
        # witnesses actually belong to that owner define a bank threshold.
        patch = {}
        witnesses = row.get('witnesses') or []
        measures = row.get('measurements') or {}
        expected = probe['expected']
        if witnesses and all(membership.get(k)==expected for k in witnesses):
            for gate,field in (('match_threshold','score'),('cohesion_threshold','minimum')):
                threshold = measures.get(field)
                if threshold is not None and 0<threshold<parameter(config.data,prefix+gate):
                    patch[prefix+gate] = at_or_below(threshold)
        if not patch and row.get('track_blocker',{}).get('reason')=='body_similarity':
            blocker = row['track_blocker']
            measured = blocker.get('measured')
            if (isinstance(measured,(int,float)) and 0<measured<parameter(config.data,prefix+'track_similarity_threshold')
                    and membership.get(blocker.get('peer'))==expected and
                    measured-blocker.get('partner_score',-1)>=parameter(config.data,prefix+'ambiguity_margin')):
                patch[prefix+'track_similarity_threshold'] = at_or_below(measured)
        if not patch:
            # Try the direct body threshold as a hypothesis; full replay still
            # checks minimum cohesion, all competitors and independent support.
            for gate in ('match_threshold','cohesion_threshold'):
                if score<parameter(config.data,prefix+gate):
                    patch[prefix+gate] = at_or_below(score)
        if not patch:
            continue
        patch = normalize(config,patch)
        if patch:
            key = index[oid],tuple(sorted(patch.items()))
            options.setdefault(key,dict(set=index[oid],observation=oid,stage=probe['stage'],patch=patch))
    # Round-robin the sets before spending a bounded trial budget.
    from .merge_solver_diagnostics import round_robin
    groups = defaultdict(list)
    for item in options.values():
        groups[item['set']].append(item)
    return list(round_robin(groups))[:64]
