"""Bounded adaptive counterfactual search; goals never alter recognition rules."""
from collections import deque
from itertools import combinations
import math

from .config import digest
from .merge_report_policy import patched, proposal
from .merge_goals import assess, objective
from .replay_simulation import replay, impact


def measured_options(db, config, goals):
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
    for key, values in options.items():
        section, field = key.split('.')
        current = changes.get(key, config.data[section][field])
        for value in values:
            if value == current:
                continue
            candidate = dict(changes, **{key:value})
            if key == 'recognition.match_threshold':
                candidate['clustering.cohesion_threshold'] = min(value,
                    changes.get('clustering.cohesion_threshold', config.data['clustering']['cohesion_threshold']))
            yield candidate


def solve(config, goals, db, *, maximum=64):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('max_trials must be a positive integer.')
    from .scanning import scan_identity
    options = measured_options(db, config, goals)
    queue, seen, patches, summaries = deque([{}]), set(), {}, {}
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
        rank = objective(status) + (-outside['coverage_lost'], -outside['split_pairs'],
            -outside['newly_joined_pairs'], -len(patch['changes']), -magnitude)
        if status['admissible'] and (best_rank is None or rank > best_rank):
            best, best_rank = name, rank
        return dict(goal_result=status, outside_impact=outside)
    def stream():
        trials = 1
        while queue and trials < maximum:
            parent = queue.popleft()
            for changes in neighbours(config, parent, options):
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
                # Prioritize admissible progress; keep alternatives for nonmonotonic policies.
                if summaries[name]['admissible']:
                    queue.appendleft(changes)
                else:
                    queue.append(changes)
                if trials >= maximum:
                    break
    result = replay(config, {}, source=db, scope=goals['scope'], consume=consume, variant_stream=stream)
    for name, patch in patches.items():
        result['trials'][name].update(patch)
    return dict(simulation=result, status=('fully_solved' if best and summaries[best]['fully_solved'] else
                'baseline_retained' if best == 'baseline' else 'partial_solution' if best else 'no_admissible_solution'),
        selected_trial=best,
        recommendation=dict(trial=best, **patches[best]) if best and best != 'baseline' else None,
        search=dict(max_trials=maximum, executed=len(result['trials']),
                    budget_exhausted=len(result['trials']) >= maximum, proves_infeasibility=False))
