"""Bound the human-readable evidence for the selected solver trial."""
from collections import Counter, defaultdict


def blockers(groups, trace, locations, inputs):
    unresolved = {g['index'] for g in groups if not g.get('identity_fully_merged', g.get('fully_merged'))}
    counts, examples = defaultdict(Counter), {}
    for row in trace:
        oid, reason = row.get('observation'), row.get('reason')
        location = locations.get(oid)
        if location is None or location[0] not in unresolved or not reason:
            continue
        index = location[0]
        counts[index][reason] += 1
        examples.setdefault((index, reason), (location[2], row.get('cue', location[3] if len(location) > 3 else None)))
    result = []
    for group in inputs:
        index = group['index']
        if index not in unresolved:
            continue
        if counts[index]:
            reason, n = counts[index].most_common(1)[0]
            scene, cue = examples[index, reason]
            label = f'{reason} ({n} sampled)'
        elif group['certificates']:
            certificate = group['certificates'][0]
            label, scene, cue = certificate['reason'], certificate.get('scene'), None
        else:
            label, scene, cue = 'no measured blocker', None, None
        address = f' · scene {scene}' if scene is not None else ''
        if cue is not None:
            address += f' cue {cue}'
        result.append(f'set {index + 1}: {label}{address}')
    return result


def short_parameter(value):
    if isinstance(value, (tuple, list, dict)):
        return 'rule profile'
    return f'{value:.4g}' if type(value) is float else str(value)


def holdout_summary(report):
    metrics = report['metrics']
    before, after = metrics['baseline/joint'], metrics['winner/joint']
    parts = []
    for key in ('all', 'back', 'entangled'):
        a, b = before[key], after[key]
        if not a['observations']:
            parts.append(f'{key}: no labels')
            continue
        af, bf = (m.get('pairs_touching_slice', m)['false_merge_pairs'] for m in (a, b))
        parts.append(f"{key} ({a['observations']}): fragments {a['extra_fragments']} → {b['extra_fragments']}"
                     f' · false merge pairs {af} → {bf}')
    stress = report['omissions']
    omissions = ', '.join(f"{w['scene']}:{w['first_cue']}-{w['last_cue']} lost={w['lost_pairs']}"
                          f" joined={w['joined_pairs']}" for w in stress) or 'no series identified'
    return 'Holdout baseline → winner · ' + ' | '.join(parts) + '\n  omitted windows: ' + omissions
