"""Bounded adaptive counterfactual search; goals never alter recognition rules."""
from collections import deque
from itertools import combinations
import math

from .config import digest
from .merge_report_policy import patched, proposal
from .merge_goals import assess, objective
from .replay_simulation import replay, impact


def measured_options(db, config, goals, proofs=None):
    from .clustering import _groups
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    import numpy as np
    ns = goals['namespace']
    groups = _groups(db, goals['scope'], ns, config.data['clustering'],
        scan_policy=scan_identity(config, ns), compatibility=fallback_sql(db, config, ns))
    wanted = {m['identity_id'] for g in goals['groups'] for m in g['members']}
    selected = [g for g in groups if g.identifier in wanted]
    scores = set()
    for a, b in combinations(selected, 2):
        matrix = np.asarray(a.samples, dtype='float64') @ np.asarray(b.samples, dtype='float64').T
        if proofs is not None:
            i,j = np.unravel_index(matrix.argmin(), matrix.shape)
            proofs.append(dict(left=a.identifier, right=b.identifier,
                minimum=float(matrix.min()), mean=float(matrix.mean()), maximum=float(matrix.max()),
                weakest_observations=[a.sample_ids[i], b.sample_ids[j]]))
        for value in (matrix.min(), matrix.mean(), matrix.max()):
            if 0 <= value <= 1:
                floor = math.floor(float(value)*1e6)/1e6
                scores.update((floor, min(1., floor+0.000002)))
    options = {key: sorted(scores, reverse=True) for key in (
        'recognition.match_threshold', 'clustering.cohesion_threshold',
        'clustering.view_match_threshold', 'clustering.view_cohesion_threshold',
        'recognition.cooccurrence_merge_threshold')}
    for section, key in (('recognition','ambiguity_margin'), ('clustering','view_ambiguity_margin')):
        current = config.data[section][key]
        options[section+'.'+key] = sorted({0., current/2, current/4})
    for section, key in (('recognition','min_identity_support'), ('recognition','min_merge_support'),
                         ('clustering','view_min_support'), ('clustering','view_attach_min_support'),
                         ('clustering','same_scene_min_support'), ('clustering','auxiliary_min_support')):
        current = config.data[section][key]
        options[section+'.'+key] = sorted({1, 2, max(1,current-1)})
    options['clustering.same_scene_match_bonus'] = [.03,.05,.1,.2]
    from .pose_policy import restricted
    if restricted(config.data['clustering']):
        options['clustering.anchor_view_matching'] = [True]
    # Only retained quality gates; changed measurement contracts are rejected by replay.
    for key in ('max_anchor_pose_deviation','min_anchor_visible_face_fraction',
                'min_auxiliary_visible_face_fraction','min_anchor_face_pixels','min_anchor_sharpness',
                'min_auxiliary_face_pixels','min_auxiliary_sharpness'):
        current = config.data['clustering'][key]
        if current is not None:
            options['clustering.'+key] = [current/2, current] if 'pose' not in key else [current, min(1.,current+.1)]
    return options


def neighbours(config, changes, options):
    # Joint candidates avoid spending the entire budget on isolated threshold changes.
    for value in options['recognition.match_threshold']:
        candidate = dict(changes)
        candidate.update({'recognition.match_threshold': value,
            'clustering.cohesion_threshold': min(value, config.data['clustering']['cohesion_threshold']),
            'clustering.view_match_threshold': value,
            'clustering.view_cohesion_threshold': min(value, config.data['clustering']['view_cohesion_threshold'])})
        yield candidate
    for key, values in options.items():
        section, field = key.split('.')
        current = changes.get(key, config.data[section][field])
        values = sorted(values, key=lambda v: (abs(v-current), v))[:3]
        for value in values:
            if value == current:
                continue
            candidate = dict(changes, **{key:value})
            if key == 'recognition.match_threshold':
                candidate['clustering.cohesion_threshold'] = min(value,
                    changes.get('clustering.cohesion_threshold', config.data['clustering']['cohesion_threshold']))
            yield candidate


def solve(config, goals, db, *, maximum=64, progress=None):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('max_trials must be a positive integer.')
    from .scanning import scan_identity
    proofs = []
    options = measured_options(db, config, goals, proofs)
    seen, patches, summaries = set(), {}, {}
    best, best_rank, baseline = None, None, None
    def consume(name, observations, retained, predicted):
        nonlocal best, best_rank, baseline
        status = assess(goals, observations, retained, predicted)
        summaries[name] = status
        if baseline is None:
            baseline = dict(predicted)
        target_ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
        outside = impact({k:v for k,v in baseline.items() if k not in target_ids},
                         {k:v for k,v in predicted.items() if k not in target_ids})
        patch = patches.get(name, {'changes': []})
        magnitude = sum(abs(float(c['proposed'])-float(c['current'])) for c in patch['changes'])
        rank = objective(status) + (-sum(g['outsider_observations'] for g in status['groups']), -outside['coverage_lost'], -outside['split_pairs'],
            -outside['newly_joined_pairs'], -len(patch['changes']), -magnitude)
        if status['admissible'] and (best_rank is None or rank > best_rank):
            best, best_rank = name, rank
        return dict(goal_result=status, outside_impact=outside)
    def inspect(target, settings):
        import json
        ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
        result = []
        for row in target.execute('SELECT scan,status,payload FROM candidates'):
            payload = json.loads(row['payload'])
            if payload.get('observation') in ids:
                result.append(dict(observation=payload['observation'], scan=row['scan'], status=row['status'],
                    reason=payload.get('reason'), measurements={k:v for k,v in payload.items()
                        if type(v) in (int,float,bool) and any(t in k for t in ('score','similarity','threshold','support','margin'))}))
        current_ids = {r[0] for r in target.execute('SELECT person FROM members WHERE observation IN (SELECT value FROM json_each(?))', (json.dumps(sorted(ids)),))}
        current_goals = dict(goals, groups=[dict(members=[dict(identity_id=p) for p in sorted(current_ids)])])
        updated = measured_options(target, settings, current_goals)
        for key, values in updated.items():
            options[key] = sorted(set(options.get(key, ())) | set(values), reverse=True)
        return result
    def stream():
        trials, attempts = 1, 0
        pending = deque([iter(neighbours(config, {}, options))])
        while pending and trials < maximum and attempts < maximum * 100:
            iterator = pending.popleft()
            try:
                changes = next(iterator)
            except StopIteration:
                continue
            pending.append(iterator)
            attempts += 1
            patch = proposal(config, changes)
            fingerprint = digest(patch['yaml_patch'])
            if not patch['changes'] or fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                variant = patched(config, changes)
            except ValueError:
                continue
            if scan_identity(variant, goals['namespace']) != scan_identity(config, goals['namespace']):
                continue
            name = f'trial-{trials}'
            patches[name] = patch
            yield name, variant
            trials += 1
            child = iter(neighbours(config, changes, options))
            if best == name:
                pending.appendleft(child)
            else:
                pending.append(child)
    result = replay(config, {}, source=db, scope=goals['scope'], consume=consume, variant_stream=stream, inspect_trial=inspect, progress=progress)
    for name, patch in patches.items():
        if name in result['trials']:
            result['trials'][name].update(patch)
    return dict(measured_reference_pairs=proofs, simulation=result, status=('fully_solved' if best and summaries[best]['fully_solved'] else
                'baseline_retained' if best == 'baseline' else 'partial_solution' if best else 'no_admissible_solution'),
        selected_trial=best,
        recommendation=dict(trial=best, **patches[best]) if best and best != 'baseline' else None,
        interrupted=result.get('interrupted', False),
        search=dict(max_trials=maximum, executed=len(result['trials']),
                    budget_exhausted=len(result['trials']) >= maximum, proves_infeasibility=False))
