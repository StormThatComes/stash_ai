"""Bounded adaptive counterfactual search; goals never alter recognition rules."""
from itertools import combinations

from .config import digest
from .merge_report_policy import patched, proposal, parameter
from .merge_goals import assess, objective, unmet_conditions, goal_diagnostics
from .replay_simulation import replay, impact
from .merge_solver_search import (hybrid_neighbours as _hybrid_neighbours,
    interleave as _interleave, spread, BeamSearch, refinement, restart_probes, cadence, BASELINE_WARMUP, PREFIX, PERSON_PREFIX, GATES)
from .merge_goal_index import solver_scope
from .parallel_numeric import diagnostic_scope


@diagnostic_scope
def measured_options(db, config, goals, proofs=None, *, hybrid=None, frozen_goals=None):
    from .clustering import _groups
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    from .merge_solver_measurements import measured_pairs
    from .resource_plan import plan
    ns = goals['namespace']
    wanted = {m['identity_id'] for g in goals['groups'] for m in g['members']}
    groups = _groups(db, goals['scope'], ns, config.data['clustering'],
        scan_policy=scan_identity(config, ns), compatibility=fallback_sql(db, config, ns), people=wanted)
    selected = [g for g in groups if g.identifier in wanted]
    scores = set()
    owners = {}
    for g in goals['groups']:
        for m in g['members']:owners.setdefault(m['identity_id'], set()).add(g['index'])
    face_pairs = {}
    from .merge_solver_measurements import diagnostic_pairs
    pairs, coverage = diagnostic_pairs(selected, config)
    for a, b, values in measured_pairs(pairs,
            max_bytes=min(4194304, plan(config).working)):
        minimum, mean, maximum, (i, j) = values
        if proofs is not None:
            proofs.append(dict(left=a.identifier, right=b.identifier,
                minimum=minimum, mean=mean, maximum=maximum,
                weakest_observations=[a.sample_ids[i], b.sample_ids[j]]))
        if frozen_goals is not None and owners[a.identifier] == owners[b.identifier] and len(owners[a.identifier]) == 1:
            index = next(iter(owners[a.identifier]))
            entry = dict(left=a.identifier, right=b.identifier, minimum=minimum, mean=mean,
                         observations=[a.sample_ids[i], b.sample_ids[j]], set=index)
            for endpoint in (a.identifier,b.identifier):
                bucket = face_pairs.setdefault((index,endpoint), [])
                bucket.append(entry)
                bucket.sort(key=lambda p: (-p['minimum'], -p['mean'], p['left'], p['right']))
                del bucket[2:]  # Keep bridges incident to weak fragments as well as dense cores.
        for value in (minimum, mean, maximum):
            from .threshold_boundaries import sides
            scores.update(sides(value))
        if len(scores)>384:
            ordered=sorted(scores)
            scores={ordered[i*(len(ordered)-1)//127] for i in range(128)}
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
    from .solver_measurement_budget import settings as diagnostic_settings, measure
    diagnostic = diagnostic_settings(config)
    diagnostic_coverage = {'hybrid': {}, 'body': {}}
    analysis = measure(lambda: analyse(db, diagnostic, goals),
        dict(enabled=False, options={}, proposals=[], pairs=[]), diagnostic_coverage['hybrid'])
    analysis['face_measurement_coverage'] = coverage
    from .body_solver import analyse as body_analyse
    bodies=measure(lambda: body_analyse(db,diagnostic,goals,frozen_goals=frozen_goals),
        dict(evidence=dict(enabled=False,pairs=[]), options={},proposals=[],quality=[]), diagnostic_coverage['body'])
    analysis['measurement_coverage'] = diagnostic_coverage
    analysis['independent_person']=bodies['evidence']
    analysis['options'].update(bodies['options'])
    analysis['proposals']=bodies['proposals']+analysis['proposals']
    if frozen_goals is not None:
        analysis['face_pairs'] = list({(p['left'],p['right']):p for bucket in face_pairs.values() for p in bucket}.values())
        analysis['body_quality'] = bodies['quality']
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
    from .merge_goal_identity import destination_owners
    ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
    owners = dict(db.execute('SELECT observation,person FROM members WHERE observation IN '
                            '(SELECT value FROM json_each(?))', (json.dumps(sorted(ids)),)))
    from .body_reports import members as body_members
    owners.update((k,v) for k,v in body_members(db).items() if k in ids)
    destinations = destination_owners(goals, owners)
    groups = []
    for index, group in enumerate(goals['groups']):
        people = {p for p, sets in destinations.items() if group.get('index', index) in sets}
        groups.append(dict(index=group.get('index', index), members=[dict(identity_id=p) for p in sorted(people)]))
    return dict(goals, groups=groups)


@solver_scope
def solve(config, goals, db, *, maximum=500, progress=None, checkpoint=None, resume=False, winner_writer=None, sampler="heuristic", seed=0, yaml_mode='full', outside_limits=None, feasibility_trials=6, targeted_trials=8, execution=None):
    from .solver_execution import current as execution_plan
    runtime = execution_plan()
    from .merge_solver_outside import validate as validate_outside, apply as apply_outside, outsiders
    outside_limits = validate_outside(outside_limits)
    if type(maximum) is not int or maximum < 1:
        raise ValueError('max_trials must be a positive integer.')
    if type(feasibility_trials) is not int or feasibility_trials < 0:
        raise ValueError('feasibility_trials must be a nonnegative integer.')
    if type(targeted_trials) is not int or targeted_trials < 0:
        raise ValueError('targeted_trials must be a nonnegative integer.')
    if yaml_mode not in ('full', 'diff'):
        raise ValueError('YAML mode must be full or diff.')
    from .scanning import scan_identity
    from .person_signal import same_contract
    from .merge_solver_sampler import create
    learned=create(sampler,seed)
    sampler_metadata=learned.metadata if learned else dict(name='heuristic',seed=seed)
    from .merge_solver_checkpoint import Journal, fingerprint
    from .merge_solver_robustness import confidence
    from .merge_solver_polish import Polisher, allowance
    from .merge_goal_output import configuration_document
    from copy import deepcopy
    if resume and checkpoint is None:raise ValueError('Resume requires a solver checkpoint.')
    from .solver_progress import heartbeat
    with heartbeat('initialization', progress, 'Validate solver checkpoint'):
        journal = Journal(checkpoint, fingerprint(config,goals,db,dict(maximum=maximum,sampler=sampler_metadata,yaml_mode=yaml_mode,outside_limits=outside_limits,feasibility_trials=feasibility_trials,targeted_trials=targeted_trials,execution=runtime.document() if runtime else None)),
                          resume=resume,winner=winner_writer) if checkpoint else None
    from .merge_solver_hybrid import public
    from .merge_solver_feedback import QueryFeedback, pair_separation, quality_feedback, face_fallbacks
    from .merge_solver_diagnostics import diagnostic_record, latest_decisions
    from .merge_solver_targets import deficits
    from .merge_solver_metrics import magnitude as mutation_magnitude
    from .merge_solver_constraints import component_proposals, crossovers, coupled_refinement, interaction_proposals
    from .merge_solver_preflight import boundary_probes, ordered
    from .merge_solver_bidirectional import BidirectionalSearch
    from .hybrid_query_observer import observe_queries
    from .offline_evaluation import _selection
    from .merge_solver_options import compact
    from .merge_solver_domains import normalize
    from .merge_goal_index import current as goal_index
    inputs = {}
    replay_scope, scans = _selection(db, config, None, goals['scope'], allow_partial=True, diagnostics=inputs)
    replay_selection = replay_scope, scans, inputs
    from .merge_solver_feasibility import Feasibility, resource_limit
    from .merge_solver_profiles import profiles
    from collections import deque
    feasibility = Feasibility(db, config, goals, scans)
    # Keep room for ordinary tuning; a two-trial run can still try one probe.
    probe_budget = min(feasibility_trials, maximum-1, max(1, (maximum-1)//2))
    probes = deque(profiles(config)) if probe_budget else deque()
    failures = {}
    from .merge_solver_targeted import plan as targeted_plan, TargetedSearch
    targeted = TargetedSearch(min(targeted_trials, maximum-1, max(1, (maximum-1)//3)))
    selected_scans = {s['id'] for s in scans}
    measurement_goals = dict(goals, groups=[dict(g, members=[m for m in g['members']
        if any(r['scan'] in selected_scans for r in m['observations'])]) for g in goals['groups']])
    target_ids = goal_index().observation_ids
    outcome_ids = goal_index().outcome_ids | feasibility.addresses
    proofs = []
    hybrid = {}
    from .solver_progress import heartbeat
    with heartbeat('initialization', progress, 'Measure retained evidence'):
        options = (measured_options(db, config, measurement_goals, proofs, hybrid=hybrid)
                   if scans else {})
    if not scans:
        hybrid.update(enabled=False, options={}, proposals=[])
    from .merge_exclusion_feedback import analyse as exclusion_feedback
    excluded = exclusion_feedback(db, config, goals) if goals.get('scene_exclusions') else None
    if excluded:
        for key, values in excluded['options'].items():
            options.setdefault(key, []).extend(values)
    options = compact(config, {}, options)
    initial_hybrid = public(hybrid)
    hybrid_changes = list(hybrid['proposals'])
    initial_changes = tuple(hybrid_changes)
    seen, patches, summaries, ranks = set(), {}, {}, {}
    families, origins, durations = {}, {}, {}
    transitions = {'baseline': dict(previous_trial=None, changes_from_previous=[])}
    feedback, outcomes = {}, {}
    query_feedback = QueryFeedback(goals)
    latest_feedback = {}
    search = None
    bidirectional = None
    stop_reason = None
    best, best_rank, baseline = None, None, None
    baseline_outsiders = None
    robustness = {}
    restored_count = 0
    written_winner = None
    probe_state = None
    def ranking(name):
        return ranks[name][:-2]+(confidence(robustness.get(name,[])),)+ranks[name][-2:]
    def selected(name):
        status=summaries[name]
        return dict(trial=name,quality=('fully_solved' if status['fully_solved'] else
            'partial_solution' if status['admissible'] else 'best_effort'),
            fully_solved=status['fully_solved'],unmet_conditions=unmet_conditions(status),
            goal_diagnostics=goal_diagnostics(status),
            **patches.get(name,proposal(config,{})))
    def consume(name, observations, retained, predicted):
        nonlocal best, best_rank, baseline, baseline_outsiders
        status = assess(goals, observations, retained, predicted)
        summaries[name] = status
        if baseline is None:
            baseline = dict(predicted)
        outside = impact({k:v for k,v in baseline.items() if k not in target_ids},
                         {k:v for k,v in predicted.items() if k not in target_ids})
        if 'new_outsider_observations' in outside_limits:
            attached=outsiders(status,predicted,target_ids)
            if baseline_outsiders is None:baseline_outsiders=attached
            outside['new_outsider_observations']=len(attached-baseline_outsiders)
        apply_outside(status, outside, summaries['baseline'], outside_limits)
        status['construction'] = feasibility.assess(status, observations, predicted,
                                                    latest_feedback['provenance']['blocker_trace'])
        patch = patches.get(name, {'changes': []})
        magnitude = mutation_magnitude(patch['changes'])
        rank = objective(status) + (-sum(g['outsider_observations'] for g in status['groups']), -outside['coverage_lost'], -outside['split_pairs'],
            -outside['newly_joined_pairs'], -len(patch['changes']), -magnitude)
        ranks[name] = rank
        if origins.get(name,{}).get('search_strategy')=='robustness_probe':
            parent=origins[name]['search_parent']
            robustness.setdefault(parent,[]).append(rank[:-2]>=ranks[parent][:-2])
        best=max(ranks,key=ranking)
        best_rank=ranking(best)
        # Canonicalize destination UUIDs so identical partitions share a beam slot.
        labels = {}
        outcomes[name] = digest([(oid, None if predicted.get(oid) is None else
            labels.setdefault(predicted[oid], len(labels))) for oid in sorted(outcome_ids)])
        feedback[name] = latest_feedback.copy()
        latest_feedback['provenance']['trial'] = name
        return dict(goal_result=status, outside_impact=outside, hybrid_solver_evidence=public(hybrid),
            solver_feedback={k:v for k,v in latest_feedback.items() if k not in ('proposals','options','hybrid_proposals')},
            best_trial=best, best_trial_number=0 if best == 'baseline' else int(best.split('-')[1]),
            **transitions[name])
    def adopt(settings, collected):
        result, fresh, measured, changes, updated = collected
        latest_feedback.clear(); latest_feedback.update(fresh)
        hybrid.clear(); hybrid.update(measured)
        hybrid_changes[:] = changes
        parents = [{k: parameter(settings.data, k) for k in updated if parameter(settings.data, k) is not None}]
        if search:
            parents.extend(c['changes'] for c in search.beam.values())
        if bidirectional:
            parents.extend(c['changes'] for c in bidirectional.frontier.values())
        bounded = compact(config, options, updated, parents=parents)
        options.clear(); options.update(bounded)
        latest_feedback['options'] = compact(settings, {}, updated, parents=parents)
        return result
    def inspect(target, settings):
        from .solver_feedback import collect
        return adopt(settings, collect(target, settings, goals, feasibility,
            query_feedback, initial_changes, excluded, bool(targeted.budget)))
    def request(name, settings, context):
        from .solver_trial import Request
        from dataclasses import replace
        from .config import plain
        return Request(name, replace(settings, data=plain(settings.data)),
            context['seed_path'], goals['scope'], goals['namespace'], context['installation'],
            context['scans'], context['observations'], goals, feasibility, initial_changes, excluded,
            bool(targeted.budget), dict(query_feedback.visits), dict(query_feedback.priorities), runtime,
            baseline_predictions=context.get('baseline_predictions'))
    def accept(settings, evaluated):
        adopt(settings, evaluated['feedback'])
        for key, amount in evaluated['sampling_visits_delta'].items():
            query_feedback.visits[key] = query_feedback.visits.get(key, 0) + amount
        latest_feedback['query']['sampling_visits'] = dict(query_feedback.visits)
        query_feedback.priorities = dict(latest_feedback.get('targets', {}).get('priorities', {}))
    retire = lambda name: None
    pending_search = {}
    def stream():
        nonlocal search, bidirectional, stop_reason, retire
        trials, attempts, probes_attempted = 1, 0, 0
        previous_name, previous_changes = 'baseline', {}
        factories = {'hybrid': lambda changes: _hybrid_neighbours(config, changes, options, hybrid_changes),
                     'face': lambda changes: _face_neighbours(config, changes, options)}
        if not config.data['sapiensid']['enabled']:
            del factories['hybrid']
        def factory(parent, changes, family, refine):
            if refine == 'restart':
                for candidate in restart_probes(config, changes, options):
                    family='hybrid' if any(k.startswith(PERSON_PREFIX) for k in candidate) else 'face'
                    yield family,'restart_probe',candidate
                return
            if refine:
                for candidate in _interleave((refinement(config, changes, options),
                                              coupled_refinement(config, changes, options, feedback[parent]))):
                    family = 'hybrid' if any(k.startswith(PERSON_PREFIX) and v != changes.get(k)
                        for k,v in candidate.items()) else 'face'
                    yield family, 'local_refinement', candidate
                return
            hints = feedback[parent]['proposals'][family]
            directed = (('measured_blockers', p) for p in ordered(config,
                (dict(changes, **patch) for patch in hints), feedback[parent]))
            source = feedback[parent]
            scoped_options = dict(options, **source['options'])
            active = source['provenance']['active_parameters']
            scoped_options = dict(sorted(scoped_options.items(), key=lambda kv: (-active.get(kv[0],0),kv[0])))
            candidates = (_hybrid_neighbours(config, changes, scoped_options, source['hybrid_proposals'])
                          if family == 'hybrid' else _face_neighbours(config, changes, scoped_options))
            probes = (('coordinate_probe', p) for p in candidates)
            mixing = (('crossover',p) for p in crossovers(changes,
                [c['changes'] for n,c in search.beam.items() if n!=parent])) if parent!='baseline' else iter(())
            boundaries = (('boundary_probe',p) for p in ordered(config,
                boundary_probes(changes,scoped_options),source))
            interactions = (('learned_interaction', p) for p in interaction_proposals(config, changes, scoped_options, source))
            yield from search.operations({'measured_blockers':directed,'coordinate_probe':probes,
                                          'crossover':mixing,'boundary_probe':boundaries,'learned_interaction':interactions})
        search = BeamSearch(factories, factory, maximum, config)
        search.observe('baseline', {}, ranking('baseline'), outcomes['baseline'])
        bidirectional = BidirectionalSearch(config, goals, maximum, feedback['baseline'])
        bidirectional.observe('baseline', {}, summaries['baseline'], ranking('baseline'), outcomes['baseline'])
        polishing = None
        sampled_for=None
        normal_exhausted = False
        alternate_streak = 0
        def changes_for(name):
            return {c['parameter']:c['proposed'] for c in patches.get(name,{}).get('changes',())}
        def completed(name):
            nonlocal normal_exhausted, alternate_streak
            if name not in pending_search:
                return
            normalized, family, phase, parent, strategy = pending_search.pop(name)
            if name in failures:
                search.rejected(family, strategy)
                return
            alternate_streak = alternate_streak+1 if phase == 'bidirectional' else 0
            search.completed(phase, parent=parent)
            improved = search.observe(name, normalized, ranking(name), outcomes[name], family=family, operator=strategy,
                elapsed=durations.get(name,0.), blockers=sorted({r['reason'] for r in feedback[name]['provenance']['blocker_trace']}), phase=phase)
            if improved:
                normal_exhausted = False
                search.stop_reason = None
            bidirectional.observe(name, normalized, summaries[name], ranking(name), outcomes[name], phase=phase)
            # Keep detailed hints only for parents that can still be expanded.
            for old in (set(feedback) - search.beam.keys() - bidirectional.frontier.keys() -
                        {'baseline', best} - set(robustness) -
                        set(sorted(ranks,key=ranks.get,reverse=True)[:2]) - {bidirectional.safe['name'] if bidirectional.safe else 'baseline'} -
                        set(sorted(ranks,key=ranking,reverse=True)[:2]) - set(pending_search) -
                        {v[3] for v in pending_search.values()}):
                del feedback[old]
        retire = completed
        while trials < maximum and attempts < maximum * 100:
            reserve = allowance(maximum, changes_for(best),
                feedback[best]['provenance']['active_parameters'])
            target = (targeted.next(best, feedback[best], changes_for(best))
                      if polishing is None and not summaries[best]['fully_solved'] else None)
            if target is not None:
                changes, strategy = target['patch'], target['route']
                family = 'hybrid' if any(k.startswith(PERSON_PREFIX) for k in changes) else 'face'
                phase, parent = 'targeted', best
            elif (probes and probes_attempted < probe_budget
                    and not any(s['construction']['merged'] for s in summaries.values())):
                strategy, changes = probes.popleft()
                family = 'hybrid' if any(k.startswith(PERSON_PREFIX) for k in changes) else 'face'
                phase, parent = 'feasibility', 'baseline'
            elif polishing is None and trials < maximum-reserve:
                period = cadence(search.operators.get('tpe', {}), 5)
                tpe_due = (learned and not normal_exhausted and trials >= BASELINE_WARMUP and
                    (trials % period == 0 if sampled_for is None else trials-sampled_for >= period))
                alternate = (bidirectional.next(trials, feedback, solved=summaries[best]['fully_solved'],
                    force=normal_exhausted, interval=cadence(search.lanes.get('bidirectional', {}), 2))
                    if not tpe_due and (alternate_streak < 2 or normal_exhausted) else None)
                try:
                    if tpe_due:
                        sampled_for=trials
                        history=[({c['parameter']:c['proposed'] for c in patches.get(n,{}).get('changes',())},
                                  ranking(n)) for n in ranks]
                        active=feedback[best]['provenance']['active_parameters']
                        changes=learned.suggest(config,history,options,active,trials)
                        family='hybrid' if any(k.startswith(PERSON_PREFIX) for k in changes) else 'face'
                        phase,parent,strategy='adaptive',best,'tpe'
                    elif alternate is not None:
                        family, phase, parent, strategy, changes = alternate
                    elif normal_exhausted:
                        raise StopIteration
                    else:
                        family, phase, parent, strategy, changes = search.next()
                except StopIteration:
                    if not normal_exhausted:
                        normal_exhausted = True
                        continue  # Finish the bounded alternate frontier before stopping.
                    stop_reason=search.stop_reason
                    polishing = Polisher(config, min(reserve, maximum-trials))
            else:
                if polishing is None:
                    polishing = Polisher(config, min(reserve, maximum-trials))
            if polishing is not None:
                try:
                    phase, parent, strategy, changes = polishing.next(best,
                        sorted(ranks,key=ranking,reverse=True), changes_for, feedback)
                except StopIteration:
                    stop_reason=stop_reason or 'polishing_complete'
                    break
                family='hybrid' if any(k.startswith(PERSON_PREFIX) for k in changes) else 'face'
            attempts += 1
            try:
                changes = normalize(config, changes)
            except ValueError:
                search.rejected(family,strategy)
                continue
            patch = proposal(config, changes)
            fingerprint = digest(patch['yaml_patch'])
            if not patch['changes'] or fingerprint in seen:
                search.rejected(family,strategy)
                continue
            seen.add(fingerprint)
            try:
                variant = patched(config, changes)
            except ValueError:
                search.rejected(family,strategy)
                continue
            if (scan_identity(variant, goals['namespace']) != scan_identity(config, goals['namespace']) or
                    not same_contract(config, variant)):
                search.rejected(family,strategy)
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
            if phase == 'targeted':
                origins[name]['search_target'] = {k:v for k,v in target.items() if k != 'patch'}
            pending_search[name] = normalized, family, phase, parent, strategy
            if polishing is not None:
                polishing.completed(phase)
            yield name, variant
            previous_name, previous_changes = name, normalized
            trials += 1
            probes_attempted += int(phase == 'feasibility')
            targeted.executed += int(phase == 'targeted')
        if stop_reason is None:
            stop_reason = 'max_trials' if trials >= maximum else 'proposal_limit'
    def restore(name,settings):
        nonlocal baseline,baseline_outsiders,best,best_rank,restored_count
        if journal is None or name not in journal.records:return None
        record=journal.get(name)
        if record['configuration']!=digest(settings.data):
            raise ValueError('Checkpoint search reconstruction diverged; completed trials were not reused.')
        state=deepcopy(record['state'])
        if state.get('failed'):
            failures[name] = deepcopy(record['item'])
            query_feedback.reset()
            query_feedback.visits = state['visits']
            restored_count += 1
            return deepcopy(record['item']), baseline
        summaries[name]=state['status'];ranks[name]=tuple(state['rank']);outcomes[name]=state['outcome']
        latest_feedback.clear();latest_feedback.update(state['feedback']);feedback[name]=latest_feedback.copy()
        query_feedback.visits = dict(latest_feedback['query'].get('sampling_visits', {}))
        query_feedback.priorities = dict(latest_feedback.get('targets', {}).get('priorities', {}))
        options.clear();options.update(state['options'])
        hybrid.clear();hybrid.update(state['hybrid']);hybrid_changes[:]=state['hybrid_changes']
        robustness.clear();robustness.update(state['robustness'])
        if name=='baseline':
            baseline=state['baseline']
            if 'new_outsider_observations' in outside_limits:
                baseline_outsiders=outsiders(summaries[name],baseline,target_ids)
        best=max(ranks,key=ranking);best_rank=ranking(best);restored_count+=1
        return deepcopy(record['item']),baseline
    def record(name,settings,item,inputs):
        nonlocal written_winner
        if item.get('failed'):
            if journal is not None:
                journal.record(name, settings, item, dict(failed=True, visits=dict(query_feedback.visits)), None)
            return
        winner_text=(configuration_document(config,selected(best),inputs,mode=yaml_mode)
                     if winner_writer and written_winner!=best else None)
        if journal is not None:
            state=deepcopy(dict(status=summaries[name],rank=ranks[name],outcome=outcomes[name],
                feedback=latest_feedback,options=options,hybrid=hybrid,hybrid_changes=hybrid_changes,
                robustness=robustness,**({'baseline':baseline} if name=='baseline' else {})))
            journal.record(name,settings,item,state,winner_text)
        elif winner_text is not None:winner_writer.write(winner_text)
        if winner_text is not None:written_winner=best
    last_stage_progress = 0.
    def trial_progress(name, stage, item):
        nonlocal probe_state
        nonlocal last_stage_progress
        if stage.startswith('stage: '):
            import time
            now = time.monotonic()
            if now - last_stage_progress < 1.:
                return
            last_stage_progress = now
        if stage == 'complete':
            durations[name]=item['elapsed_seconds']
            retire(name)
        elif stage == 'failed':
            retire(name)
        if stage == 'start':
            if name == 'baseline' and progress:
                detail = (f'cpus={runtime.cpus} trials={runtime.trials} inner_workers={runtime.inner_workers} '
                          f'scheduling={runtime.scheduling}' if runtime else 'trials=1 scheduling=serial')
                progress('execution', 'plan', dict(text=f'{detail} max_trials={maximum}; baseline runs first'))
            probe_state = (deepcopy(dict(options=options, hybrid=hybrid, hybrid_changes=hybrid_changes,
                feedback=latest_feedback, visits=query_feedback.visits, priorities=query_feedback.priorities))
                if origins.get(name, {}).get('search_phase') == 'feasibility' else None)
            item = dict(transitions[name], search_parent=origins.get(name, {}).get('search_parent'),
                        **{k:v for k,v in origins.get(name, {}).items() if k != 'search_parent'})
        if progress:progress(name, stage, item)
    trial_progress.telemetry_enabled = progress is not None
    def trial_error(name, settings, exc, stage):
        if name == 'baseline' or not resource_limit(exc):
            return None
        query_feedback.reset()
        if probe_state is not None and not (runtime and runtime.trials > 1):
            options.clear(); options.update(probe_state['options'])
            hybrid.clear(); hybrid.update(probe_state['hybrid'])
            hybrid_changes[:] = probe_state['hybrid_changes']
            latest_feedback.clear(); latest_feedback.update(probe_state['feedback'])
            query_feedback.visits = probe_state['visits']
            query_feedback.priorities = probe_state['priorities']
        item = dict(reason='resource_limit', error=str(exc), diagnostics=list(getattr(exc, '__notes__', ())),
                    stage=stage, **origins[name], **patches[name])
        failures[name] = item
        return item
    with observe_queries(query_feedback):
        result = replay(config, {}, source=db, scope=goals['scope'], consume=consume, variant_stream=stream,
            inspect_trial=inspect, progress=trial_progress, allow_partial_selection=True,
            restore_trial=restore,record_trial=record,prepared_selection=replay_selection,trial_error=trial_error,
            parallel_request=request if runtime and runtime.trials > 1 else None,
            parallel_accept=accept, journal=journal)
    # Ctrl-C can land after consume ranks a trial but before replay records it.
    robustness.clear()
    for name in result['trials']:
        if origins.get(name,{}).get('search_strategy')=='robustness_probe':
            parent=origins[name]['search_parent']
            robustness.setdefault(parent,[]).append(ranks[name][:-2]>=ranks[parent][:-2])
    best = max(result['trials'], key=ranking, default=None)
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
            goal_diagnostics=goal_diagnostics(status),
            **patches.get(best, proposal(config, {})))
    tested_hybrid = {}
    if config.data['sapiensid']['enabled'] and result['trials']:
        for gate in GATES:
            values = {config.data['sapiensid']['matching'][gate]}
            values.update(change['proposed'] for name in result['trials']
                for change in patches.get(name, {}).get('changes', []) if change['parameter'] == PREFIX + gate)
            tested_hybrid[gate] = sorted(values)
    return dict(measured_reference_pairs=proofs, measured_hybrid_evidence=initial_hybrid,
        feasibility=feasibility.report(result['trials'], result.get('failed_trials', {}), probe_budget),
        simulation=result, status=(result.get('stop_reason', 'no_completed_trials') if selection is None else
            'baseline_retained' if best == 'baseline' and selection['quality'] == 'partial_solution' else selection['quality']),
        selected_trial=best, best_result=selection,
        recommendation=selection if selection is not None and selection['changed'] else None,
        interrupted=result.get('interrupted', False),
        robustness=dict(selected=best,checks=robustness.get(best,[]),
            validated=bool(confidence(robustness.get(best,[]))),
            candidates=robustness,interpretation='Local completed replays only; no global stability guarantee.'),
        search=dict(max_trials=maximum,sampler=sampler_metadata, executed=len(result['trials']),restored_trials=restored_count,
                    attempted=len(result['trials'])+len(failures), failed_trials=len(failures),
                    scheduling='targeted_then_feasibility_and_bidirectional_beam',
                    execution=runtime.document() if runtime else None,
                    targeted=dict(requested=targeted_trials,budget=targeted.budget,
                        executed=sum(origins.get(n,{}).get('search_phase')=='targeted' for n in result['trials'])),
                    bidirectional=dict(enabled=bidirectional.enabled if bidirectional else False,
                        budget=bidirectional.budget if bidirectional else 0,
                        frontier=list(bidirectional.frontier) if bidirectional else [],
                        executed=sum(origins.get(n,{}).get('search_phase')=='bidirectional' for n in result['trials'])),
                    stop_reason='interrupted' if result.get('interrupted') else result.get('stop_reason', stop_reason),
                    beam_width=search.width if search else BeamSearch.width, peak_beam=search.peak_beam if search else 0,
                    beam_trials=list(search.beam) if search else [],
                    pruned_trials=search.pruned if search else 0,
                    operators=search.operators if search else {}, restarts=search.restarts if search else 0,
                    stagnation_patience=search.patience if search else None,
                    refinement_budget=BeamSearch.refinement_budget,
                    executed_by_family={family: sum(families.get(name) == family for name in result['trials'])
                                        for family in ('hybrid', 'face')},
                    baseline_exploration_warmup=BASELINE_WARMUP,
                    executed_by_phase={phase: sum(origins.get(name, {}).get('search_phase') == phase
                        for name in result['trials']) for phase in ('targeted', 'feasibility', 'baseline_exploration', 'adaptive', 'bidirectional', 'refinement', 'restart', 'minimization', 'robustness')},
                    tested_hybrid_values=tested_hybrid,
                    budget_exhausted=len(result['trials'])+len(failures) >= maximum,
                    selection_scope='completed_trials_including_baseline',
                    proves_optimality=False, proves_infeasibility=False))
