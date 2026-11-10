"""Replay positive physical labels against independently frozen goal evidence."""
import json

from .merge_observation_exclusions import resolve


def freeze(db, config, goals, owners, observations, tracks):
    pairs = list(observations)
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
        pairs.extend((performer, member[0]) for member in members)
    labels = resolve(db, config, pairs, goals['scope'], owners)
    claimed = {}
    forbidden = {(e['set'],oid) for e in goals.get('observation_exclusions',()) for oid in e['observation_ids']}
    forbidden_scenes = {(e['set'],e['scene']) for e in goals.get('scene_exclusions',())}
    for item in labels:
        group = item['set']
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


def assess(goals, status, observations, predicted):
    labels = goals.get('positive_observations')
    if not labels:
        return
    targets = {int(index): {predicted.get(oid) for oid in ids if oid in observations}-{None}
               for index,ids in goals['positive_reference_ids'].items()}
    results = []
    for item in labels:
        found = {predicted.get(oid) for oid in item['observation_ids'] if oid in observations}-{None}
        expected = targets[item['set']]
        results.append(dict(set=item['set'], scene=item['scene'], observations=item['requested_observation_ids'],
                            satisfied=len(expected)==1 and found==expected))
    status['positive_observations'] = results
    if not all(r['satisfied'] for r in results):
        status.update(admissible=False, fully_solved=False)


def measure(db, goals, expected, decisions):
    """Bound diagnostics to user-labelled cues and a small frozen scene roster."""
    import numpy as np
    from .database import encoded
    labels = {oid for item in goals['positive_observations'] for oid in item['observation_ids']}
    selected = {row['observation']:row for row in decisions if row['observation'] in labels}
    banks = {}
    for oid,scan,cue,blob in db.execute('SELECT id,scan,cue,embedding FROM body_observations '
            'WHERE id IN (SELECT value FROM json_each(?)) AND embedding IS NOT NULL', (encoded(sorted(selected)),)):
        if scan not in banks:
            banks[scan] = {}
            refs = db.execute("SELECT b.id,b.cue,m.person,b.embedding FROM body_observations b JOIN body_members m "
                "ON m.observation=b.id WHERE b.scan=? AND m.status='accepted' AND m.role='anchor' "
                "AND b.embedding IS NOT NULL ORDER BY b.cue,b.id", (scan,))
            for ref,at,owner,vector in refs:
                bank = banks[scan].setdefault(owner, {})
                if ref not in labels and at not in bank and len(bank)<8:
                    bank[at] = np.frombuffer(vector, dtype='<f4')
        query = np.frombuffer(blob,dtype='<f4')
        scores = {owner:max(float(query @ v) for at,v in bank.items() if at!=cue)
                  for owner,bank in banks[scan].items() if any(at!=cue for at in bank)}
        wanted = expected.get(oid)
        ranked = sorted(((score,p) for p,score in scores.items() if p!=wanted),reverse=True)
        row = selected[oid]
        row['label_probe'] = dict(expected=wanted, score=scores.get(wanted), rival=ranked[0][1] if ranked else None,
                                  runner=ranked[0][0] if ranked else None,
                                  stage=(row.get('track_blocker') or {}).get('reason') or row['reason'])
