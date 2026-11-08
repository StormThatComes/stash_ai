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
    from .physical_goal import addresses
    ids = addresses(goals)
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
def solve(config, goals, db, *, maximum=500, progress=None, checkpoint=None, resume=False, winner_writer=None, sampler="heuristic", seed=0, yaml_mode='full', outside_limits=None, feasibility_trials=6, targeted_trials=8, execution=None, compare_anchor_rules=False, compact_report=False):
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
    from .merge_solver_tighten import Tightener
    from .merge_solver_domains import normalize, support_baseline_changes
    input_config = config
    baseline_changes = support_baseline_changes(config)
    if baseline_changes:
        config = patched(config, baseline_changes)
    from .merge_solver_profiles import anchor_rule_profiles, profiles
    from .merge_solver_report import observation_locations, trim_completed
    locations = observation_locations(goals) if compact_report or getattr(progress, 'show_winner', False) else {}
    rule_profiles = anchor_rule_profiles(config)[:maximum-1] if compare_anchor_rules else ()
    search_limit = maximum - len(rule_profiles)
    tightening = Tightener(config)
    from .merge_goal_output import configuration_document
    from copy import deepcopy
    if resume and checkpoint is None:raise ValueError('Resume requires a solver checkpoint.')
    from .solver_progress import heartbeat
    with heartbeat('initialization', progress, 'Validate solver checkpoint'):
        journal = Journal(checkpoint, fingerprint(input_config,goals,db,dict(maximum=maximum,sampler=sampler_metadata,yaml_mode=yaml_mode,outside_limits=outside_limits,feasibility_trials=feasibility_trials,targeted_trials=targeted_trials,compare_anchor_rules=compare_anchor_rules,execution=runtime.signature() if runtime else None)),
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
    from .merge_goal_index import current as goal_index
    inputs = {}
    replay_scope, scans = _selection(db, config, None, goals['scope'], allow_partial=True, diagnostics=inputs)
    replay_selection = replay_scope, scans, inputs
    from .merge_solver_feasibility import Feasibility, resource_limit
    from collections import deque
    feasibility = Feasibility(db, config, goals, scans)
    # Keep room for ordinary tuning; a two-trial run can still try one probe.
    probe_budget = min(feasibility_trials, search_limit-1, max(1, (search_limit-1)//2))
    probes = deque(profiles(config)) if probe_budget else deque()
    failures = {}
    from .merge_solver_targeted import plan as targeted_plan, TargetedSearch
    targeted = TargetedSearch(min(targeted_trials, search_limit-1, max(1, (search_limit-1)//3)))
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
    partial_diagnostics = set()
    def report_coverage(measurements):
        if not getattr(progress, 'show_diagnostics', False):
            return
        for kind, data in measurements.get('measurement_coverage', {}).items():
            if data.get('complete') is False and kind not in partial_diagnostics:
                partial_diagnostics.add(kind)
                progress('diagnostics', 'partial', dict(kind=kind, **data))
    report_coverage(hybrid)
    if not scans:
        hybrid.update(enabled=False, options={}, proposals=[])
    from .merge_exclusion_feedback import analyse as exclusion_feedback
    excluded = exclusion_feedback(db, config, goals) if (goals.get('scene_exclusions') or goals.get('observation_exclusions')) else None
    if excluded:
        for key, values in excluded['options'].items():
            options.setdefault(key, []).extend(values)
    options = compact(config, {}, options)
    initial_hybrid = public(hybrid)
    hybrid_changes = list(hybrid['proposals'])
    initial_changes = tuple(hybrid_changes)
    seen, patches, summaries, ranks = set(), {}, {}, {}
    families, origins, durations = {}, {}, {}
    transitions = {'baseline': dict(previous_trial=None, changes_from_previous=[
        dict(parameter=key, previous=parameter(input_config.data, key), current=value)
        for key, value in sorted(baseline_changes.items())])}
    feedback, outcomes = {}, {}
    fallback_references = {}
    query_feedback = QueryFeedback(goals)
    latest_feedback = {}
    search = None
    bidirectional = None
    stop_reason = None
    best, best_rank, baseline = None, None, None
    baseline_outsiders = None
    robustness = {}
    restored_count = 0
    restored_name = None
    restored_names = set()
    written_winner = None
    probe_state = None
    def changes_for(name):
        return {c['parameter']:c['proposed'] for c in patches.get(name,{}).get('changes',())}
    def transition(previous_name, previous, changes):
        from .config import plain
        delta = []
        for key in sorted(previous.keys() | changes.keys()):
            original = plain(parameter(config.data, key))
            before, after = previous.get(key, original), changes.get(key, original)
            if before != after:
                delta.append(dict(parameter=key, previous=before, current=after))
        return dict(previous_trial=previous_name, changes_from_previous=delta)
    def input_patch(name):
        # Replays search from the bounded baseline. Export diffs against the user's YAML.
        if baseline_changes:
            return proposal(input_config, baseline_changes | changes_for(name))
        return patches[name] if name in patches else proposal(input_config, {})
    def ranking(name):
        solved = summaries[name]['fully_solved']
        # Hard goals/exclus