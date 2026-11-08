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
        examples.setdefault((index, reason), (location[2], row.get('cue', location[3] if len(location) > 3 else None),row))
    result = []
    for group in inputs:
        index = group['index']
        if index not in unresolved:
            continue
        if counts[index]:
            reason, n = counts[index].most_common(1)[0]
            scene, cue, example = examples[index, reason]
            label = f'{reason} ({n} sampled)'
            if 'measured' in example and 'limit' in example:
                label += f" · measured {example['measured']:.4g}, threshold {example['limit']:.4g}"
            if example.get('requires_evidence') and cue is not None:
                label += ' · inspect detection audit'
        elif group['certificates']:
            certificate = group['certificates'][0]
            label, scene, cue = certificate['reason'], certificate.get('scene'), None
        else:
            label, scene, cue = 'no measured blocker', None, None
            inputs_for_set = group.get('inputs', {})
            if inputs_for_set.get('body_routes', {}).get('collection'):
                unavailable = Counter(inputs_for_set.get('body_input_counts', {}))
                unavailable.pop('ready_for_matching', None)
                if unavailable:
                    reason, n = unavailable.most_common(1)[0]
                    row = inputs_for_set['body_input_examples'][reason]
                    address = locations.get(row['observation'])
                    label = f'body input {reason} ({n} samples)'
                    scene, cue = address[2] if address else None, row.get('cue')
        address = f' · scene {scene}' if scene is not None else ''
        if cue is not None:
            address += f' cue {cue}'
        result.append(f'set {index + 1}: {label}{address}')
    return result


def parameter_change(change):
    key = change['parameter']
    before = change.get('previous', change.get('current'))
    after = change.get('proposed', change.get('current'))
    while isinstance(before, (dict, list)) and type(before) is type(after):
        if isinstance(before, dict):
            fields = sorted(before.keys() | after.keys())
            field = next((k for k in fields if before.get(k, 'unset') != after.get(k, 'unset')), None)
            if field is None:
                break
            key += '.' + field
            before, after = before.get(field, 'unset'), after.get(field, 'unset')
        else:
            from itertools import zip_longest
            for i, (left, right) in enumerate(zip_longest(before, after, fillvalue='unset')):
                if left != right:
                    key += f'[{i}]'
                    before, after = left, right
                    break
            else:
                break
    def short(value):
        if type(value) is float:
            return f'{value:.6g}'
        text = str(value)
        return text if len(text) <= 40 else text[:39] + '…'
    return f'{key} {short(before)} → {short(after)}'


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
    if 'partners' in before:
        parts.append(f"partner false joins {before['partners']['false_joins']} → {after['partners']['false_joins']}")
    for name,trial in report.get('incremental',{}).items():
        parts.append(f"{name} lost={trial['lost_pairs']} joined={trial['joined_pairs']} coverage changes={trial['coverage_changes']}")
    if 'sequences' in after:
        for metric,total in [('one_id_turns','turns'),('both_recovered','crossings'),('own_id_turns','anchored_turns')]:
            if metric not in after['sequences']:
                continue
            parts.append(f"{metric} {before['sequences'][metric]} → {after['sequences'][metric]}/{after['sequences'][total]}")
        for metric in ('foreign_id_crossings','abstained_crossings','unscored_episodes'):
            if metric in after['sequences']:
                parts.append(f"{metric} {before['sequences'][metric]} → {after['sequences'][metric]}")
    if 'physical' in after:
        parts.append(f"physical projection conflicts {after['physical']['projection_conflicts']}")
    if report.get('detection',{}).get('expected_people'):
        parts.append(f"missing poses {report['detection']['missing_poses']}/{report['detection']['expected_people']}")
    if after.get('visible',{}).get('total'):
        physical = after['visible']
        parts.append(f"visible people {physical['correct']}/{physical['total']} correct, {physical['partner']} partner, "
                     f"{physical['fragment']} fragments, {physical['withheld']} withheld, {physical['missing']} missing, "
                     f"{physical['unknown_reference']} without unique reference")
    fault = report.get('association_stress',{})
    if fault.get('status')=='tested':
        parts.append(f"association swap: {fault['later_cues']} later cues changed, {len(fault['other_scenes'])} other scenes")
    if 'verdict' in report:
        parts.append('independent holdout '+report['verdict']['status'])
    if 'contact_interior' in after.get('sequences',{}):
        parts.append('contact interior '+str(after['sequences']['contact_interior']))
    stress = report['omissions']
    omissions = ', '.join(f"{w['scene']}:{w['first_cue']}-{w['last_cue']} lost={w['lost_pairs']}"
                          f" joined={w['joined_pairs']} {w.get('mode','rejected')}" for w in stress) or 'no series identified'
    return 'Holdout baseline → winner · ' + ' | '.join(parts) + '\n  omitted windows: ' + omissions
