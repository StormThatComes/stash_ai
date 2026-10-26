"""Bounded adaptive counterfactual search; goals never alter recognition rules."""
from collections import deque
from itertools import combinations
import math

from .config import digest
from .merge_report_policy import patched, proposal, parameter
from .merge_goals import assess, objective, unmet_conditions
from .replay_simulation import replay, impact


def measured_options(db, config, goals, proofs=None, *, hybrid=None):
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
    from .merge_solver_hybrid import analyse
    analysis = analyse(db, config, goals)
    options.update(analysis['options'])
    if hybrid is not None:
        hybrid.update(analysis)
    return options


def _face_neighbours(config, changes, options):
    # Joint candidates avoid spending the entire budget on isolated threshold changes.
    for value in options['recognition.match_threshold']:
        candidate = dict(changes)
        candidate.update({'recognition.match_threshold': value,
            'clustering.cohesion_threshold': min(value, config.data['clustering']['cohesion_threshold']),
            'clustering.view_match_threshold': value,
            'clustering.view_cohesion_threshold': min(value, config.data['clustering']['view_cohesion_threshold'])})
        yield candidate
    for key, values in options.items():
        if key.startswith('sapiensid.'):
            continue
        current = changes.get(key, parameter(config.data,key))
        values = sorted(values, key=lambda v: (abs(v-current), v))[:3]
        for value in values:
            if value == current:
                continue
            candidate = dict(changes, **{key:value})
            if key == 'recognition.match_threshold':
                candidate['clustering.cohesion_threshold'] = min(value,
                    changes.get('clustering.cohesion_threshold', config.data['clustering']['cohesion_threshold']))
            yield candidate


def _hybrid_neighbours(config, changes, options, measured):
    for patch in tuple(measured):
        yield dict(changes, **patch)
    # Round-robin parameters as well as search families: a long threshold list
    # must not starve the independent-support and competitor-margin gates.
    streams = []
    for key, values in options.items():
        if not key.startswith('sapiensid.'):
            continue
        current = changes.get(key, parameter(config.data, key))
        values = sorted((v for v in values if v != current), key=lambda v: (abs(v-current), v))[:3]
        streams.append(iter([dict(changes, **{key: value}) for value in values]))
    yield from _interleave(streams)


def _interleave(streams):
    pending = deque(streams)
    while pending:
        stream = pending.popleft()
        try:
            yield next(stream)
        except StopIteration:
            continue
        pending.append(stream)


def neighbours(config, changes, options, *, hybrid_changes=()):
    yield from _interleave([_hybrid_neighbours(config, changes, options, hybrid_changes),
                            _face_neighbours(config, changes, options)])


def rebuilt_goals(db, goals):
    """Follow observations through merges/splits without collapsing goal sets."""
    import json
    ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
    owners = dict(db.execute('SELECT observation,person FROM members WHERE observation IN '
                            '(SELECT value FROM json_each(?))', (json.dumps(sorted(ids)),)))
    groups = []
    for index, group in enumerate(goals['groups']):
        people = {owners[r['id']] for m in group['members'] for r in m['observations'] if r['id'] in owners}
        groups.append(dict(index=group.get('index', index), members=[dict(identity_id=p) for p in sorted(people)]))
    return dict(goals, groups=groups)


def solve(config, goals, db, *, maximum=64, progress=None):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('max_trials must be a positive integer.')
    from .scanning import scan_identity
    from .person_signal import same_contract
    from .merge_solver_hybrid import public
    proofs = []
    hybrid = {}
    options = measured_options(db, config, goals, proofs, hybrid=hybrid)
    initial_hybrid = public(hybrid)
    hybrid_changes = list(hybrid['proposals'])
    initial_changes = tuple(hybrid_changes)
    seen, patches, summaries, ranks = set(), {}, {}, {}
    families = {}
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
        ranks[name] = rank
        if best_rank is None or rank > best_rank:
            best, best_rank = name, rank
        return dict(goal_result=status, outside_impact=outside, hybrid_solver_evidence=public(hybrid))
    def inspect(target, settings):
        import json
        ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
        result = []
        for row in target.execute('SELECT scan,status,payload FROM candidates'):
            payload = json.loads(row['payload'])
            if payload.get('observation') in ids:
                result.append(dict(observation=payload['observation'], scan=row['scan'], status=row['status'],
                    reason=payload.get('reason'), hybrid_merge=payload.get('hybrid_merge'),
                    hybrid_query=payload.get('hybrid_query'), measurements={k:v for k,v in payload.items()
                        if type(v) in (int,float,bool) and any(t in k for t in ('score','similarity','threshold','support','margin'))}))
        current_goals = rebuilt_goals(target, goals)
        hybrid.clear()
        updated = measured_options(target, settings, current_goals, hybrid=hybrid)
        hybrid_changes[:] = list({digest(p): p for p in hybrid['proposals'] + list(initial_changes)}.values())
        for key, values in updated.items():
            options[key] = sorted(set(options.get(key, ())) | set(values), reverse=True)
        return result
    def stream():
        trials, attempts = 1, 0
        factories = {'hybrid': lambda changes: _hybrid_neighbours(config, changes, options, hybrid_changes),
                     'face': lambda changes: _face_neighbours(config, changes, options)}
        if not config.data['sapiensid']['enabled']:
            del factories['hybrid']
        pending = {family: deque([factory({})]) for family, factory in factories.items()}
        rotation = deque(pending)
        while pending and trials < maximum and attempts < maximum * 100:
            family = rotation[0]
            queue = pending[family]
            if not queue:
                del pending[family]
                rotation.popleft()
                continue
            iterator = queue.popleft()
            try:
                changes = next(iterator)
            except StopIteration:
                continue
            queue.append(iterator)
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
            if (scan_identity(variant, goals['namespace']) != scan_identity(config, goals['namespace']) or
                    not same_contract(config, variant)):
                continue
            name = f'trial-{trials}'
            patches[name] = patch
            families[name] = family
            yield name, variant
            trials += 1
            rotation.rotate(-1)
            for next_family, factory in factories.items():
                if next_family not in pending:
                    pending[next_family] = deque()
                    rotation.append(next_family)
                child = factory(changes)
                if best == name:
                    pending[next_family].appendleft(child)
                else:
                    pending[next_family].append(child)
    result = replay(config, {}, source=db, scope=goals['scope'], consume=consume, variant_stream=stream, inspect_trial=inspect, progress=progress)
    # Ctrl-C can land after consume ranks a trial but before replay records it.
    best = max(result['trials'], key=ranks.__getitem__, default=None)
    for name, patch in patches.items():
        if name in result['trials']:
            result['trials'][name].update(patch)
            result['trials'][name]['search_family'] = families[name]
    selection = None
    if best is not None:
        status = summaries[best]
        quality = ('fully_solved' if status['fully_solved'] else
                   'partial_solution' if status['admissible'] else 'best_effort')
        selection = dict(trial=best, quality=quality, changed=best != 'baseline',
            admissible=status['admissible'], fully_solved=status['fully_solved'],
            unmet_conditions=unmet_conditions(status),
            **patches.get(best, proposal(config, {})))
    return dict(measured_reference_pairs=proofs, measured_hybrid_evidence=initial_hybrid,
        simulation=result, status=('no_completed_trials' if selection is None else
            'baseline_retained' if best == 'baseline' and selection['quality'] == 'partial_solution' else selection['quality']),
        selected_trial=best, best_result=selection,
        recommendation=selection if selection is not None and selection['changed'] else None,
        interrupted=result.get('interrupted', False),
        search=dict(max_trials=maximum, executed=len(result['trials']),
                    scheduling='round_robin_hybrid_face' if config.data['sapiensid']['enabled'] else 'face',
                    executed_by_family={family: sum(families.get(name) == family for name in result['trials'])
                                        for family in ('hybrid', 'face')},
                    budget_exhausted=len(result['trials']) >= maximum,
                    selection_scope='completed_trials_including_baseline',
                    proves_optimality=False, proves_infeasibility=False))
