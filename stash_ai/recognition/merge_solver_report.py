"""Compact JSON for diagnosing a solver run without replaying its observation history."""
from collections import Counter, defaultdict

from .merge_report_policy import parameter


def fields(row, names):
    return {key: row[key] for key in names.split() if key in row}


def observation_locations(goals):
    return {row['id']: (group['index'], member.get('performer_id'), row['scene'], row.get('cue'))
            for group in goals['groups'] for member in group['members'] for row in member['observations']}


def feedback_summary(feedback, hybrid, config, changes, locations):
    """Keep a measured query and a few distinct proposed routes per set."""
    query = feedback.get('query', {})
    examples = {}
    settings = config.data
    for pair in hybrid.get('pairs', ()):
        if pair['relation'] != 'within_set' or not pair['blockers']:
            continue
        for index in pair['sets'][0]:
            key = index, 'hybrid_pair'
            if key not in examples:
                examples[key] = dict(fields(pair, 'left right face person support blockers thresholds cooccurring'),
                                     source='measured_hybrid_pair', set=index)
    destinations = feedback.get('targets', {}).get('destinations', {})
    for pair in hybrid.get('independent_person', {}).get('pairs', ()):
        if pair.get('relation') != 'within_set' or not pair.get('blockers'):
            continue
        index = destinations.get(pair['people'][0], {}).get('set')
        if index is not None and (index, 'body_pair') not in examples:
            prefix = 'sapiensid.global_matching.'
            examples[index, 'body_pair'] = dict(fields(pair, 'people score minimum support blockers'),
                source='measured_body_pair', set=index, thresholds={k: changes.get(prefix+k, parameter(settings, prefix+k))
                    for k in ('match_threshold', 'cohesion_threshold', 'min_support', 'ambiguity_margin')})
    for row in query.get('queries', ()):
        key = row.get('set'), 'query'
        if key in examples or not row.get('blockers') or row.get('relation') != 'within_set':
            continue
        example = fields(row, 'set observation identity face person support query_support gap rival_score blockers')
        example['thresholds'] = {k: changes.get('sapiensid.matching.' + k, v)
                                 for k, v in settings['sapiensid']['matching'].items()
                                 if k in ('face_floor', 'person_threshold', 'min_support', 'ambiguity_margin')}
        examples[key] = dict(example, source='measured_query')
    targeted = feedback.get('targeted', {})
    for row in targeted.get('candidates', ()):
        route = row['route'].split('_')[0]
        key = row['set'], 'proposal_' + route
        if key in examples:
            continue
        patch = row['patch']
        examples[key] = dict(fields(row, 'set route blockers'), source='proposed_route', performers=row['performers'][:4],
            observations=row.get('observations', [])[:2], fragments=row.get('fragments', [])[:2],
            parameters={k: dict(current=changes.get(k, parameter(settings, k)), proposed=v) for k, v in patch.items()})
    for row in feedback.get('provenance', {}).get('blocker_trace', ()):
        location = locations.get(row.get('observation'))
        if location is None or row['stage'] not in ('assignment', 'quality', 'body_track'):
            continue
        index, performer, scene = location[:3]
        key = index, row['stage']
        if key in examples:
            continue
        example = fields(row, 'stage reason observation parameters requires_evidence cue rival peer observation_cue measured limit')
        rules = row.get('keypoint_rules', {})
        if rules:
            example['failed_rules'] = [fields(rule, 'parameter label measured minimum') for mode in ('all', 'any')
                for rule in rules.get(mode, ()) if rule['parameter'] in rules.get('failed_parameters', ())][:3]
        examples[key] = dict(example, source='observed_blocker', set=index, performer=performer, scene=scene)
    samples = sorted(feedback.get('label_probes', {}).items())
    return dict(**({'label_probe_count':len(samples),
                    'label_probes':[dict(observation=oid,**row) for oid,row in samples[:16]]} if samples else {}),
                query=fields(query, 'truncated truncation_reasons skipped_query_visits comparisons'),
                targeted_truncated=targeted.get('truncated', False), examples=list(examples.values()),
                fallback_references=feedback.get('fallback_references', 0))


def trim_completed(name, item, config, changes, locations):
    # Called after journaling and console progress. The live search owns its separate feedback.
    item['solver_feedback'] = feedback_summary(item.get('solver_feedback', {}), item.get('hybrid_solver_evidence', {}),
                                              config, changes, locations)
    for key in ('decision_evidence', 'hybrid_solver_evidence', 'hybrid'):
        item.pop(key, None)
    if name != 'baseline':  # Replay uses the baseline body identities to decode worker deltas.
        item.pop('independent_person', None)


def goal_input(group):
    def member(row):
        return dict(fields(row, 'identity_id performer_id requested_ids'),
                    observations=len(row.get('observations', ())),
                    excluded_observations=len(row.get('excluded_observations', ())))
    result = dict(index=group['index'], members=[member(row) for row in group['members']])
    if group.get('excluded_members'):
        result['excluded_members'] = [member(row) for row in group['excluded_members']]
    return result


def unmet_summary(status):
    from .merge_goals import unmet_conditions
    counts = Counter((row.get('set'), tuple(row.get('sets', ())), row['reason']) for row in unmet_conditions(status))
    return [dict(dict(sets=indices) if indices else dict(set=index), reason=reason, count=n)
            for (index, indices, reason), n in counts.items()]


def condition_examples(conditions):
    """Keep searchable examples and exact counts without whole scene/destination maps."""
    counts, examples = Counter(), []
    for row in conditions:
        key = row.get('set'), tuple(row.get('sets', ())), row['reason']
        counts[key] += 1
        if counts[key] > 2:
            continue
        example = {k:v for k,v in row.items() if not isinstance(v, (list, tuple, dict))}
        for k in ('sets', 'scenes', 'performer_ids', 'requested_observation_ids', 'observations', 'identities'):
            if k in row:
                example[k] = row[k][:4]
                if len(row[k]) > 4:
                    example[k + '_count'] = len(row[k])
        if 'scene_support' in row:
            example['scene_count'] = len(row['scene_support'])
        if isinstance(row.get('core_support'), dict):
            example['strongest_core_support'] = max(row['core_support'].values(), default=0)
        examples.append(example)
    return examples


class SetHistory:
    """Store identical per-set outcomes once, even across different search phases."""

    def __init__(self, groups):
        self.states = []
        self.state_ids = {}
        self.frequency = defaultdict(dict)
        self.first_blocker_trials = set()
        self.sets = {g['index']: dict(index=g['index'], merged_trials=0, unmerged_trials=0,
            performers=[m.get('performer_id') or m['identity_id'] for m in g['members']]) for g in groups}

    def add(self, name, group, construction):
        from .config import digest
        index = group['index']
        row = fields(group, 'index total_performers consolidated_performers represented_performers '
                     'largest_complete_performers complete_components fragments assigned observations missing withheld '
                     'required_scenes covered_scenes set_shared_coverage fully_merged identity_fully_merged')
        row['construction'] = fields(construction, 'merged core_support core_required')
        row['observed_blockers'] = construction.get('observed_blockers', [])
        signature = digest(row)
        if signature not in self.state_ids:
            self.state_ids[signature] = len(self.states)
            self.states.append(row)
        summary = self.sets[index]
        if group.get('identity_fully_merged', group['fully_merged']):
            summary['merged_trials'] += 1
            summary.setdefault('first_merge_trial', name)
        else:
            summary['unmerged_trials'] += 1
            for blocker in row['observed_blockers']:
                key = blocker['stage'], blocker['reason']
                stats = self.frequency[index].setdefault(key, dict(stage=key[0], reason=key[1],
                    trials=0, occurrences=0, first_trial=name))
                stats['trials'] += 1
                stats['occurrences'] += blocker['count']
                self.first_blocker_trials.add(stats['first_trial'])
        return self.state_ids[signature]

    def report(self, result, history):
        for index, summary in self.sets.items():
            summary['observed_blockers'] = sorted(self.frequency[index].values(),
                key=lambda v: (-v['trials'], -v['occurrences'], v['stage'], v['reason']))
        for label, name in (('baseline_state', 'baseline'), ('selected_state', result.get('selected_trial'))):
            for ref in history.get(name, {}).get('goal_result', {}).get('set_states', ()):
                self.sets[self.states[ref]['index']][label] = ref
        owners = {m['identity_id']: g['index'] for g in result['groups'] for m in g['members']}
        for pair in result.get('measured_reference_pairs', ()):
            index = owners.get(pair['left'])
            if index is None or index != owners.get(pair['right']):
                continue
            previous = self.sets[index].get('baseline_face_pair')
            if previous is None or pair['minimum'] > previous['minimum']:
                self.sets[index]['baseline_face_pair'] = fields(pair, 'left right minimum mean maximum weakest_observations')
        return dict(sets=list(self.sets.values()), set_states=self.states,
            interpretation='Trial set_states are zero-based indices into this table. Blocker frequencies cover '
                           'unmerged sets and up to 16 observed stage/reason counts per set and replay. '
                           'Repeated observations across trials count again. These are not proofs of causation.')


def compact_document(result, config):
    from .merge_goals import unmet_conditions
    simulation = result['simulation']
    output = {k:v for k,v in result.items() if k not in (
        'simulation', 'explanation', 'measured_hybrid_evidence', 'measured_reference_pairs', 'physical_alternatives', 'positive_reference_ids')}
    output['solver_report_format'] = 'compact-v1'
    output['groups'] = [goal_input(group) for group in result['groups']]
    for kind in ('scene_exclusions', 'observation_exclusions'):
        if kind in output:
            output[kind] = [dict(fields(row, 'set scene performer_ids resolved_identity_ids requested_observation_ids'),
                                 evidence_count=len(row.get('observation_ids', ()))) for row in output[kind]]

    states = SetHistory(result['groups'])
    highlights = {result.get('selected_trial'), 'baseline'}
    highlights.update(g.get('best_trial') for g in result.get('feasibility', {}).get('groups', ()))
    history = {}
    locations = None
    for name, trial in simulation['trials'].items():
        status = trial['goal_result']
        construction = status.get('construction', {})
        evidence = {g['index']: g for g in construction.get('groups', ())}
        refs = [states.add(name, group, evidence.get(group['index'], {})) for group in status['groups']]
        item = fields(trial, 'previous_trial changes_from_previous search_family search_phase search_parent search_strategy '
                      'best_trial best_trial_number elapsed_seconds yaml_patch outside_impact')
        item['goal_result'] = dict(fields(status, 'fully_solved admissible disjoint strict_disjoint strict_fully_solved'),
                                   set_states=refs, unmet=unmet_summary(status), construction_status=construction.get('status'))
        if 'search_target' in trial:
            item['search_target'] = fields(trial['search_target'], 'set route performers blockers')
        if name in highlights or name in states.first_blocker_trials:
            item['goal_result']['unmet_examples'] = condition_examples(unmet_conditions(status))
            item['sample_assignment'] = [dict(index=g['index'], **fields(g.get('sample_assignment', {}), 'face sapiensid'))
                                         for g in status['groups']]
            feedback = trial.get('solver_feedback', {})
            if 'examples' not in feedback:
                # An interrupt can land after the replay was recorded but before progress ran.
                changes = {r['parameter']: r['proposed'] for r in trial.get('changes', ())}
                if locations is None:
                    locations = observation_locations(result)
                feedback = feedback_summary(feedback, trial.get('hybrid_solver_evidence', {}), config, changes, locations)
            item['solver_feedback'] = feedback
        history[name] = item
    output['merge_diagnostics'] = states.report(result, history)
    output['simulation'] = {k:v for k,v in simulation.items() if k not in ('trials', 'trial_report_encoding', 'failed_trials')}
    output['simulation']['trials'] = history
    if 'failed_trials' in simulation:
        failures = {name: fields(item, 'failed reason error diagnostics resource stage elapsed_seconds search_phase '
                                 'search_strategy search_parent yaml_patch') for name,item in simulation['failed_trials'].items()}
        output['simulation']['failed_trials'] = failures
        if 'feasibility' in output:
            output['feasibility'] = dict(output['feasibility'], failed_probes={
                n: failures[n] for n in output['feasibility'].get('failed_probes', {}) if n in failures})
    output['explanation'] = fields(result['explanation'], 'selected_trial unresolved conclusion search_stop_reason '
                                   'feasibility_status input_blockers validation_scope')
    if result.get('best_result'):
        best = dict(result['best_result'])
        conditions = best['unmet_conditions']
        best['unmet_conditions'] = condition_examples(conditions)
        best['unmet_condition_count'] = len(conditions)
        output['best_result'] = best
        if output.get('recommendation'):
            output['recommendation'] = dict(trial=best['trial'], details='best_result')
        output['explanation']['unresolved'] = best['unmet_conditions']
    # Put the cross-trial diagnosis first so inspecting a large report starts with the reasons.
    return {**fields(output, 'solver_report_format status selected_trial merge_diagnostics feasibility explanation'), **output}
