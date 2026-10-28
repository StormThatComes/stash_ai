"""Per-set sample diagnostics, independent of the solver objective.

Roles and body availability are frozen from input metadata. Body features share
their face observation's final membership; these counts do not attribute a match
to a particular encoder or claim that a sample trained a reference bank.
"""
from collections import Counter, defaultdict

ROLES = ('anchor', 'auxiliary', 'other')
BODY_STATES = ('ready', 'unavailable', 'missing', 'incompatible', 'disabled')


def freeze(db, config, rows, diagnostics):
    """Read only evidence status/contract; never load vectors or retained crops."""
    from .database import encoded
    from .person_signal import available
    from .pose_policy import is_anchor, query_allows
    body = defaultdict(dict)
    if diagnostics['sapiensid_enabled'] and available(db):
        for oid, contract, status in db.execute(
                'SELECT observation,contract,status FROM person_evidence '
                'WHERE observation IN (SELECT value FROM json_each(?))', (encoded([r['id'] for r in rows]),)):
            body[oid][contract] = status
    for row in rows:
        eligible = row['status'] == 'complete' and row['rejected'] is None
        if eligible and row['reference_authority'] and is_anchor(row['quality'], config.data['clustering']):
            role = 'anchor'
        elif eligible and query_allows(row['quality'], config.data['clustering']) and not is_anchor(row['quality'], config.data['clustering']):
            role = 'auxiliary'
        else:
            role = 'other'
        records = body[row['id']]
        state = (records.get(diagnostics['sapiensid_contract'], 'incompatible' if records else 'missing')
                 if diagnostics['sapiensid_enabled'] else 'disabled')
        row.update(sample_role=role, body_state=state)


def target(group):
    """One common destination for all role counters, preferring identity support."""
    if 'set_target' in group:
        return group['set_target'], group['set_target_ties']
    represented, coverage = Counter(), Counter()
    for member in group['members']:
        evidence = member.get('identity_evidence')
        if evidence is not None:
            counts, total = evidence['core_destinations'], evidence['core_units']
            if evidence['represented']:
                represented[evidence['destination']] += 1
        else:
            counts, total = member['resulting_identities'], member['observations']
            if member['fully_assigned_to_one']:
                represented[next(iter(counts))] += 1
        for destination, count in counts.items():
            coverage[destination] += count/total if total else 0.
    ranks = {d: (represented[d], coverage[d], n) for d,n in group['resulting_identities'].items()}
    best = max(ranks.values(), default=None)
    tied = sorted(d for d,rank in ranks.items() if rank == best)
    return (tied[0] if tied else None), tied


def assignment(rows, observations, predicted, destination, owners):
    present = [r['id'] for r in rows if r['id'] in observations]
    counts = Counter(predicted[k] for k in present if predicted.get(k) is not None)
    assigned = sum(counts.values())
    shared = counts.get(destination, 0)
    return dict(total=len(rows), assigned=assigned, to_target=shared, other=assigned-shared,
                withheld=len(present)-assigned, missing=len(rows)-len(present), fragments=len(counts),
                cross_set=sum(n for d,n in counts.items() if len(owners[d]) > 1))


def attach(goals, result, observations, predicted):
    diagnostics = goals.get('sample_diagnostics')
    if diagnostics is None:
        return result  # Legacy callers without a frozen role snapshot.
    owners = defaultdict(set)
    for group in goals['groups']:
        for member in group['members']:
            for row in member['observations']:
                if row['id'] in observations and predicted.get(row['id']) is not None:
                    owners[predicted[row['id']]].add(group['index'])
    for source, group in zip(goals['groups'], result['groups']):
        rows = [r for m in source['members'] for r in m['observations']]
        destination, tied = target(group)
        face, body = {}, {}
        for role in ROLES:
            selected = [r for r in rows if r['sample_role'] == role]
            face[role] = assignment([r for r in selected if r.get('evidence_kind','face')=='face'], observations, predicted, destination, owners)
            states = Counter(r['body_state'] for r in selected)
            ready = [r for r in selected if r['body_state'] == 'ready']
            body[role] = dict(availability={s: states[s] for s in BODY_STATES},
                             **assignment(ready, observations, predicted, destination, owners))
        group['sample_assignment'] = dict(diagnostics, selector=goals['selector'],
            requested_ids=list(dict.fromkeys(i for m in source['members'] for i in m['requested_ids'])),
            performer_ids=list(dict.fromkeys(m['performer_id'] for m in source['members'] if m['performer_id'] is not None)),
            target_identity=destination, target_ties=tied, face=face, sapiensid=body)
    return result


def lines(group):
    data = group.get('sample_assignment')
    if data is None:
        return []
    import json
    compact = lambda value: json.dumps(value, separators=(',', ':'))
    result = [f"    Members: {data['selector']}_ids={compact(data['requested_ids'])} "
              f"resolved_performer_ids={compact(data['performer_ids'])} roles=input_config "
              f"target={data['target_identity'] or 'none'} target_ties={len(data['target_ties'])}"]
    def counts(item):
        fraction = item['assigned']/item['total'] if item['total'] else 0.
        return (f"assigned={item['assigned']}/{item['total']} ({fraction:.1%}) "
                f"target={item['to_target']}/{item['total']} other={item['other']} "
                f"withheld={item['withheld']} missing_obs={item['missing']} "
                f"cross_set={item['cross_set']} fragments={item['fragments']}")
    for role in ROLES:
        item = data['face'][role]
        if role != 'other' or item['total']:
            result.append(f"    Face {role}: {counts(item)}")
    if not data['sapiensid_enabled']:
        result.append('    SapiensID: disabled')
        return result
    for role in ROLES:
        item = data['sapiensid'][role]
        total = sum(item['availability'].values())
        if role == 'other' and not total:
            continue
        states = item['availability']
        result.append(f"    SapiensID {role}: ready={states['ready']}/{total} {counts(item)} "
                      f"unavailable={states['unavailable']} no_evidence={states['missing']} "
                      f"incompatible={states['incompatible']}")
    return result
