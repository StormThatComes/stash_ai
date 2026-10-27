"""Bounded adaptive counterfactual search; goals never alter recognition rules."""
from itertools import combinations
import math

from .config import digest
from .merge_report_policy import patched, proposal, parameter
from .merge_goals import assess, objective, unmet_conditions
from .replay_simulation import replay, impact
from .merge_solver_search import (hybrid_neighbours as _hybrid_neighbours,
    interleave as _interleave, spread, BeamSearch, refinement, BASELINE_WARMUP, PREFIX, GATES)


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
    def joint():
        for value in spread(options['recognition.match_threshold'],
                            changes.get('recognition.match_threshold', config.data['recognition']['match_threshold'])):
            candidate = dict(changes)
            candidate.update({'recognition.match_threshold': value,
                'clustering.cohesion_threshold': min(value, config.data['clustering']['cohesion_threshold']),
                'clustering.view_match_threshold': value,
                'clustering.view_cohesion_threshold': min(value, config.data['clustering']['view_cohesion_threshold'])})
            yield candidate
    def coordinate(key, values):
        current = changes.get(key, parameter(config.data,key))
        for value in spread(values, current):
            candidate = dict(changes, **{key:value})
            if key == 'recognition.match_threshold':
                candidate['clustering.cohesion_threshold'] = min(value,
                    changes.get('clustering.cohesion_threshold', config.data['clustering']['cohesion_threshold']))
            yield candidate
    yield from _interleave([joint(), _interleave(coordinate(k,v) for k,v in options.items()
                                                if not k.startswith('sapiensid.'))])


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


def solve(config, goals, db, *, maximum=500, progress=None):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('max_trials must be a positive integer.')
    from .scanning import scan_identity
    from .person_signal import same_contract
    from .merge_solver_hybrid import public
    from .merge_solver_feedback import QueryFeedback, pair_separation, quality_feedback
    from .merge_solver_diagnostics import diagnostic_record
    from .merge_solver_constraints import component_proposals, crossovers, coupled_refinement
    from .hybrid_query_observer import observe_queries
    proofs = []
    hybrid = {}
    options = measured_options(db, config, goals, proofs, hybrid=hybrid)
    initial_hybrid = public(hybrid)
    hybrid_changes = list(hybrid['proposals'])
    initial_changes = tuple(hybrid_changes)
    seen, patches, summaries, ranks = set(), {}, {}, {}
    families, origins = {}, {}
    transitions = {'baseline': dict(previous_trial=None, changes_from_previous=[])}
    feedback, outcomes = {}, {}
    query_feedback = QueryFeedback(goals)
    latest_feedback = {}
    search = None
    stop_reason = None
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
        # Canonicalize destination UUIDs so identical partitions share a beam slot.
        labels = {}
        outcomes[name] = digest([(oid, None if predicted.get(oid) is None else
            labels.setdefault(predicted[oid], len(labels))) for oid in sorted(target_ids)])
        feedback[name] = latest_feedback.copy()
        latest_feedback['provenance']['trial'] = name
        return dict(goal_result=status, outside_impact=outside, hybrid_solver_evidence=public(hybrid),
            solver_feedback={k:v for k,v in latest_feedback.items() if k not in ('proposals','options','hybrid_proposals')},
            best_trial=best, best_trial_number=0 if best == 'baseline' else int(best.split('-')[1]),
            **transitions[name])
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
        queries = query_feedback.finish(settings)
        separation = pair_separation(settings, hybrid)
        quality = quality_feedback(target, settings, goals)
        separated = [dict(hybrid_changes[0] if hybrid_changes else {}, **p) for p in separation['proposals']]
        directed = list(queries['proposals']) + separated + component_proposals(settings, hybrid) + hybrid_changes
        # Clear simultaneous query and reference blockers under one configuration.
        if queries['proposals'] and hybrid_changes and any(p['relation'] == 'within_set' for p in hybrid.get('pairs', ())):
            joint = dict(hybrid_changes[0])
            for key,value in queries['proposals'][0].items():
                joint[key] = min(joint.get(key,value), value)
            directed.insert(0, joint)
        if directed and quality['proposals']:
            directed.append(dict(directed[0], **quality['proposals'][0]))
        latest_feedback.clear()
        latest_feedback.update(query={k:v for k,v in queries.items() if k not in ('proposals','options')},
            separation_intervals=separation['intervals'], quality=quality['observations'],
            proposals=dict(hybrid=directed, face=quality['proposals']),
            options=updated, hybrid_proposals=list(hybrid['proposals']),
            provenance=diagnostic_record(settings, result, queries, public(hybrid), quality['observations']))
        for extra in (queries['options'], separation['options']):
            for key, values in extra.items():
                updated[key] = list(updated.get(key, ())) + list(values)
        for key, values in updated.items():
            options[key] = sorted(set(options.get(key, ())) | set(values), reverse=True)
        return result
    def stream():
        nonlocal search, stop_reason
        trials, attempts = 1, 0
        previous_name, previous_changes = 'baseline', {}
        factories = {'hybrid': lambda changes: _hybrid_neighbours(config, changes, options, hybrid_changes),
                     'face': lambda changes: _face_neighbours(config, changes, options)}
        if not config.data['sapiensid']['enabled']:
            del factories['hybrid']
        def factory(parent, changes, family, refine):
            if refine:
                for candidate in _interleave((refinement(config, changes, options),
                                              coupled_refinement(config, changes, options))):
                    family = 'hybrid' if any(k.startswith(PREFIX) and v != changes.get(k)
                        for k,v in candidate.items()) else 'face'
                    yield family, 'local_refinement', candidate
                return
            hints = feedback[parent]['proposals'][family]
            directed = (('measured_blockers', dict(changes, **patch)) for patch in hints)
            source = feedback[parent]
            scoped_options = dict(options, **source['options'])
            active = source['provenance']['active_parameters']
            scoped_options = dict(sorted(scoped_options.items(), key=lambda kv: (-active.get(kv[0],0),kv[0])))
            candidates = (_hybrid_neighbours(config, changes, scoped_options, source['hybrid_proposals'])
                          if family == 'hybrid' else _face_neighbours(config, changes, scoped_options))
            probes = (('coordinate_probe', p) for p in candidates)
            mixing = (('crossover',p) for p in crossovers(changes,
                [c['changes'] for n,c in search.beam.items() if n!=parent])) if parent!='baseline' else iter(())
            yield from _interleave((directed, probes, mixing))
        search = BeamSearch(factories, factory, maximum)
        search.observe('baseline', {}, ranks['baseline'], outcomes['baseline'])
        while trials < maximum and attempts < maximum * 100:
            try:
                family, phase, parent, strategy, changes = search.next()
            except StopIteration:
                stop_reason = search.stop_reason
                break
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
            normalized = {c['parameter']:c['proposed'] for c in patch['changes']}
            delta = []
            for key in sorted(previous_changes.keys() | normalized.keys()):
                before = previous_changes.get(key, parameter(config.data, key))
                after = normalized.get(key, parameter(config.data, key))
                if before != after:
                    delta.append(dict(parameter=key, previous=before, current=after))
            transitions[name] = dict(previous_trial=previous_name, changes_from_previous=delta)
            patches[name] = patch
            families[name] = family
            origins[name] = dict(search_phase=phase, search_parent=parent, search_strategy=strategy)
            yield name, variant
            previous_name, previous_changes = name, normalized
            trials += 1
            search.completed(phase)
            search.observe(name, normalized, ranks[name], outcomes[name], family=family)
            # Keep detailed hints only for parents that can still be expanded.
            for old in set(feedback) - search.beam.keys() - {'baseline'}:
                del feedback[old]
        if stop_reason is None:
            stop_reason = 'max_trials' if trials >= maximum else 'proposal_limit'
    def trial_progress(name, stage, item):
        if stage == 'start':
            item = dict(transitions[name], search_parent=origins.get(name, {}).get('search_parent'))
        progress(name, stage, item)
    with observe_queries(query_feedback):
        result = replay(config, {}, source=db, scope=goals['scope'], consume=consume, variant_stream=stream,
            inspect_trial=inspect, progress=trial_progress if progress else None, allow_partial_selection=True)
    # Ctrl-C can land after consume ranks a trial but before replay records it.
    best = max(result['trials'], key=ranks.__getitem__, default=None)
    for name, patch in patches.items():
        if name in result['trials']:
            result['trials'][name].update(patch)
            result['trials'][name]['search_family'] = families[name]
            result['trials'][name].update(origins[name])
    selection = None
    if best is not None:
        status = summaries[best]
        quality = ('fully_solved' if status['fully_solved'] else
                   'partial_solution' if status['admissible'] else 'best_effort')
        selection = dict(trial=best, quality=quality, changed=best != 'baseline',
            admissible=status['admissible'], fully_solved=status['fully_solved'],
            unmet_conditions=unmet_conditions(status),
            **patches.get(best, proposal(config, {})))
    tested_hybrid = {}
    if config.data['sapiensid']['enabled'] and result['trials']:
        for gate in GATES:
            values = {config.data['sapiensid']['matching'][gate]}
            values.update(change['proposed'] for name in result['trials']
                for change in patches.get(name, {}).get('changes', []) if change['parameter'] == PREFIX + gate)
            tested_hybrid[gate] = sorted(values)
    return dict(measured_reference_pairs=proofs, measured_hybrid_evidence=initial_hybrid,
        simulation=result, status=(result.get('stop_reason', 'no_completed_trials') if selection is None else
            'baseline_retained' if best == 'baseline' and selection['quality'] == 'partial_solution' else selection['quality']),
        selected_trial=best, best_result=selection,
        recommendation=selection if selection is not None and selection['changed'] else None,
        interrupted=result.get('interrupted', False),
        search=dict(max_trials=maximum, executed=len(result['trials']),
                    scheduling='blocker_guided_diverse_beam',
                    stop_reason='interrupted' if result.get('interrupted') else result.get('stop_reason', stop_reason),
                    beam_width=BeamSearch.width, peak_beam=search.peak_beam if search else 0,
                    beam_trials=list(search.beam) if search else [],
                    pruned_trials=search.pruned if search else 0,
                    stagnation_patience=search.patience if search else None,
                    refinement_budget=BeamSearch.refinement_budget,
                    executed_by_family={family: sum(families.get(name) == family for name in result['trials'])
                                        for family in ('hybrid', 'face')},
                    baseline_exploration_warmup=BASELINE_WARMUP,
                    executed_by_phase={phase: sum(origins.get(name, {}).get('search_phase') == phase
                        for name in result['trials']) for phase in ('baseline_exploration', 'adaptive', 'refinement')},
                    tested_hybrid_values=tested_hybrid,
                    budget_exhausted=len(result['trials']) >= maximum,
                    selection_scope='completed_trials_including_baseline',
                    proves_optimality=False, proves_infeasibility=False))
