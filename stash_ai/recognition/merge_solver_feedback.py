"""Blocker-directed hints from frozen query evidence, including pruned queries.

These measurements steer experiments; they never assign an observation. Every
patch still needs the production replay, including its guards and competitors.
"""
from collections import defaultdict
import math

from .config import digest
from .database import encoded
from .hybrid_evidence import scores
from .independent_evidence import Votes, body_vote
from .merge_report_policy import parameter, _floor
from .merge_solver_hybrid import boundaries, PREFIX


def separating_proposals(config, positives, negatives, *, routes=()):
    """Try common acceptance bounds and single-gate separation intervals.

    A gate accepts <= the weakest desired score/support; rejecting every measured
    unwanted edge requires > the strongest unwanted one. An empty interval only
    rules out this scalar explanation on the frozen witnesses, not all replays.
    """
    joint, intervals, options = {}, [], defaultdict(set)
    for key in sorted({k for p in positives for k in p}):
        upper = min(p[key] for p in positives if key in p)
        support = key.endswith('min_support')
        if upper < (1 if support else 0):
            continue
        upper = int(upper) if support else _floor(upper)
        joint[key] = min(parameter(config.data, key), upper)
        options[key].update((upper, upper+1) if support else boundaries(upper))
        unwanted = [p[key] for p in negatives if key in p]
        if not unwanted:
            continue
        lower = max(unwanted)
        from .threshold_boundaries import strictly_above
        first = strictly_above(lower, integer=support)
        feasible = first <= upper
        intervals.append(dict(parameter=key, lower_exclusive=lower, upper_inclusive=upper,
            feasible=feasible, resolution=1 if support else .000001, interpretation='frozen_witnesses_only',
            reason='separating_interval' if feasible else 'overlapping_within_and_between_scores'))
        if support or 0 <= first <= 1:
            options[key].add(first)
    from .merge_solver_constraints import joint_separation, alternative_separation
    combined = joint_separation(config, positives, negatives)
    proposals = alternative_separation(config, routes, negatives) if routes else []
    proposals.extend(combined['proposals'])
    # Prefer separating settings over a blanket relaxation, but test both.
    for interval in intervals:
        if interval['feasible']:
            key = interval['parameter']
            lower = interval['lower_exclusive']
            first = strictly_above(lower, integer=key.endswith('min_support'))
            value = min(interval['upper_inclusive'], max(first, parameter(config.data, key)))
            proposals.append(dict(joint, **{key: value}))
    if joint:
        proposals.append(joint)
    proposals.extend(positives)
    return dict(proposals=list({digest(p): p for p in proposals}.values()),
                intervals=intervals, joint_constraints={k:v for k,v in combined.items() if k!='proposals'},
                options={k: sorted(v) for k,v in options.items()})


def pair_separation(config, analysis):
    positive, negative = [], []
    for entry in analysis.get('pairs', ()):
        proof = entry.get('mode_boundary_evidence') or entry
        if proof.get('missing_original_cores') and any(proof['missing_original_cores']):
            continue
        values = {PREFIX+gate: proof[signal]['minimum'] for gate,signal in
                  (('face_floor','face'), ('person_threshold','person')) if proof.get(signal) is not None}
        if proof.get('support'):
            values[PREFIX+'min_support'] = min(proof['support'])
        if values:
            (positive if entry['relation'] == 'within_set' else negative).append(values)
    return separating_proposals(config, positive, negative)


class QueryFeedback:
    """Observe only goal queries; production still considers all identities.

    All reference cores are measured before face-floor nomination. Retain desired
    routes and the strongest cross-set/outsider witnesses, not a corpus matrix.
    Budget counters include comparisons that are subsequently discarded.
    """
    def __init__(self, goals):
        from .merge_goal_identity import diagnostic_goals
        goals = diagnostic_goals(goals)
        self.goals = goals
        self.selected = ()
        self.sets = {r['id']: g['index'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
        self.visits, self.priorities = {}, {}
        self.reset()

    def reset(self):
        self.entries, self.negatives, self.votes = [], [], {}
        self.face_votes = {}
        self.decisions = []
        self.negative_keys = set()
        self.sampled = set()
        self.sampling = []
        self.comparisons = self.used = self.passes = 0
        self.truncation_reasons = set()
        self.skipped_query_visits = 0

    def decision(self, observation, stage, reason, details):
        if observation in self.sampled:
            self.decisions.append(dict(observation=observation, stage=stage, reason=reason, details=details))

    def begin(self, db, banks, config, births):
        self.passes += 1
        pass_id = self.passes
        settings = config.data['sapiensid']['matching']
        limits = config.data['sapiensid']['execution']
        from .merge_solver_diagnostics import fair_queries
        bank_count = max(1, sum(i.person not in births for i,_,_,_ in banks))
        quota = max(0, (limits['max_candidate_pairs']-self.comparisons)//bank_count)
        selected, eligible = fair_queries(db, config, self.goals, self.selected, births, quota,
                                         visits=self.visits, priorities=self.priorities)
        for oid in selected:
            self.visits[oid] = self.visits.get(oid, 0)+1
        self.sampled.update(selected)
        self.sampling.append(dict(pass_id=pass_id, eligible=eligible, selected=len(selected)))
        if len(selected) < eligible:
            self.truncation_reasons.add('max_candidate_pairs')
        from .merge_goal_identity import destination_owners
        membership = dict(db.execute('SELECT observation,person FROM members WHERE observation IN '
                '(SELECT value FROM json_each(?))', (encoded(sorted(self.sets)),)))
        owners = destination_owners(self.goals, membership)

        def measure(row, face, body):
            if row['id'] not in self.sets:
                return
            if row['id'] not in selected:
                self.skipped_query_visits += 1
                return
            def limited(reason):
                self.truncation_reasons.add(reason)
                self.skipped_query_visits += 1
            group = self.sets[row['id']]
            choices = []
            scratch_bytes = 0
            for identity, _, bodies, faces in banks:
                if identity.person in births:
                    continue
                if self.comparisons >= limits['max_candidate_pairs']:
                    limited('max_candidate_pairs')
                    return
                self.comparisons += 1
                f = scores(face, faces)
                modes = identity.body_modes if bodies is identity.persons else ()
                measures = [scores(body, mode) for mode in (modes or (bodies,))
                            if body is not None and len(mode)]
                b = max(measures, key=lambda m: (m['minimum'], m['mean']), default=None)
                admitted = [m for m in measures if m['minimum'] >= settings['person_threshold']]
                current = max(admitted or measures, key=lambda m: m['mean'], default=None)
                choices.append(dict(identity=identity.person, face=f, person=b,
                    score=current['mean'] if current else None, support=len(identity.independent),
                    missing_original_core=bool(identity.guard_missing),
                    relation='within_set' if owners[identity.person] == {group} else
                             'between_sets' if owners[identity.person] else 'outside_goals'))
                scratch_bytes += len(encoded(choices[-1]).encode())*2 + 256
                if self.used + scratch_bytes > limits['plan_bytes']:
                    limited('plan_bytes')
                    return
            desired = [c for c in choices if c['relation'] == 'within_set']
            # One strongest route per owner; alternative original cores are not
            # simultaneous obligations of a query attachment.
            routes = {}
            for c in desired:
                rank = (not c['missing_original_core'], c['person'] is not None,
                        c['person']['minimum'] if c['person'] else -1, c['face']['minimum'])
                if c['identity'] not in routes or rank > routes[c['identity']][0]:
                    routes[c['identity']] = rank, c
            # Reserve retained proofs and vote bookkeeping before publishing this
            # query. An incomplete competitor scan must never imply a safe gap.
            retained_cost = (scratch_bytes + 4096*max(1, len(routes)) +
                             sum(len(encoded(c).encode())*2 for _,c in routes.values()))
            if self.used + scratch_bytes + retained_cost > limits['plan_bytes']:
                limited('plan_bytes')
                return
            # Keep real per-core rival vectors; mixed coordinate maxima can
            # incorrectly make multidimensional separation look impossible.
            for c in choices:
                if c['relation'] == 'within_set':continue
                negative = {PREFIX+'face_floor': c['face']['minimum'], PREFIX+'min_support':c['support']}
                if c['person'] is not None:negative[PREFIX+'person_threshold']=c['person']['minimum']
                key=digest(negative)
                if key not in self.negative_keys:
                    self.negative_keys.add(key)
                    self.negatives.append(negative)
            if not routes:
                self.entries.append(dict(observation=row['id'], set=group, pass_id=pass_id,
                    blockers=['no_within_set_query_reference'], proposals=[]))
            for _, c in routes.values():
                floor = min(settings['face_floor'], c['face']['minimum'])
                rivals = [v for v in choices if v['identity'] != c['identity'] and v['face']['minimum'] >= floor]
                unknown = any(v['score'] is None for v in rivals)
                rival = max((v['score'] for v in rivals if v['score'] is not None), default=None)
                gap = c['person']['mean'] - rival if c['person'] is not None and rival is not None else None
                blockers = []
                if c['face']['minimum'] < settings['face_floor']:
                    blockers.append('face_floor')
                if c['person'] is None:
                    blockers.append('missing_person_evidence')
                elif c['person']['minimum'] < settings['person_threshold']:
                    blockers.append('person_threshold')
                if c['missing_original_core']:
                    blockers.append('missing_original_core_evidence')
                if c['support'] < settings['min_support']:
                    blockers.append('reference_support')
                if unknown:
                    blockers.append('unknown_competitor')
                if gap is not None and gap < settings['ambiguity_margin']:
                    blockers.append('ambiguity_margin' if gap >= 0 else 'stronger_competitor')
                entry = dict(observation=row['id'], set=group, pass_id=pass_id, **c,
                    rival_score=rival, gap=gap, unknown_competitor=unknown, blockers=blockers,
                    nearest_competitor=max((dict(identity=v['identity'], score=v['score'], relation=v['relation'])
                        for v in rivals if v['score'] is not None), key=lambda v:v['score'], default=None),
                    unknown_competitor_identity=next((v['identity'] for v in rivals if v['score'] is None), None))
                face_rivals = [v['face']['mean'] for v in choices if v['identity'] != c['identity']]
                entry['face_gap'] = c['face']['minimum']-max(face_rivals) if face_rivals else None
                self.entries.append(entry)
                self.face_votes.setdefault((pass_id, c['identity']), Votes()).add(row)
                if c['person'] is not None and not c['missing_original_core']:
                    self.votes.setdefault((pass_id, c['identity']), Votes()).add(body_vote(row))
            self.used += retained_cost
        return measure

    def finish(self, config):
        positives, face_proposals = [], []
        alternatives = defaultdict(list)
        for entry in self.entries:
            if 'identity' not in entry:
                continue
            entry['query_support'] = len(self.votes.get((entry['pass_id'], entry['identity']), ()))
            entry['face_query_support'] = len(self.face_votes.get((entry['pass_id'], entry['identity']), ()))
            # Missing body evidence blocks the hybrid route, not face matching.
            # Only measured faces are proposed; production still checks pose,
            # independent support, original cores and competing identities.
            if entry['person'] is None and entry['face']['minimum'] >= 0:
                face_support = min(entry['support'], len(self.face_votes.get((entry['pass_id'], entry['identity']), ())))
                if face_support:
                    value = _floor(entry['face']['minimum'])
                    face_proposals.append({
                        'recognition.match_threshold': min(parameter(config.data, 'recognition.match_threshold'), value),
                        'clustering.cohesion_threshold': min(parameter(config.data, 'clustering.cohesion_threshold'), value),
                        'recognition.min_identity_support': min(parameter(config.data, 'recognition.min_identity_support'), face_support)})
            support = min(entry['support'], entry['query_support'])
            if support < config.data['sapiensid']['matching']['min_support']:
                entry['blockers'].append('query_support')
            if (not support or entry['person'] is None or entry['missing_original_core'] or
                    min(entry['face']['minimum'], entry['person']['minimum']) < 0):
                continue
            patch = {PREFIX+'face_floor': _floor(entry['face']['minimum']),
                     PREFIX+'person_threshold': _floor(entry['person']['minimum']),
                     PREFIX+'min_support': support}
            if entry['gap'] is not None and entry['gap'] >= 0:
                patch[PREFIX+'ambiguity_margin'] = _floor(entry['gap'])
            positives.append(patch)
            alternatives[entry['observation']].append(patch)
        positive_routes = [dict(observation=oid, routes=list({digest(p):p for p in routes}.values()),
                                priority=self.priorities.get(oid, 0)) for oid, routes in sorted(alternatives.items())]
        focused = [r for r in positive_routes if r['priority'] > 0] or positive_routes
        hints = separating_proposals(config, positives, self.negatives, routes=[r['routes'] for r in focused])
        result = dict(queries=self.entries, decisions=self.decisions, sampling=self.sampling, comparisons=self.comparisons, passes=self.passes,
            truncated=bool(self.truncation_reasons), truncation_reasons=sorted(self.truncation_reasons),
            skipped_query_visits=self.skipped_query_visits,
            separation_intervals=hints['intervals'], joint_constraints=hints['joint_constraints'],
            positive_witnesses=positives, negative_witnesses=self.negatives,
            positive_routes=positive_routes, sampling_visits=dict(self.visits),
            face_proposals=list({digest(p):p for p in face_proposals}.values()),
            proposals=hints['proposals'], options=hints['options'],
            interpretation='Frozen query-stage hints before face-floor pruning; full replay decides acceptance.')
        self.reset()
        return result


def quality_feedback(db, config, goals, *, priorities=None):
    """Retained admission boundaries for target observations, never new measurements."""
    import json
    from .visibility import cached
    ids = [r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']
           if priorities is None or priorities.get(r['id'], 0) > 0]
    entries, joint = [], {}
    for row in db.execute('SELECT id,quality FROM observations WHERE id IN '
                          '(SELECT value FROM json_each(?))', (encoded(ids),)):
        quality = json.loads(row['quality'])
        if quality.get('alignment_valid') is False:
            entries.append(dict(observation=row['id'], blockers=['invalid_alignment'], patch={}))
            continue
        patch, missing = {}, []
        for role in ('anchor', 'auxiliary'):
            for suffix, field in (('face_pixels','pixels'), ('sharpness','sharpness'),
                                  ('visible_face_fraction','visible_face_fraction')):
                key = 'clustering.min_'+role+'_'+suffix
                current = parameter(config.data, key)
                if not current:
                    continue
                value = quality.get(field)
                if (type(value) not in (int,float) or not math.isfinite(value) or
                        field == 'visible_face_fraction' and not cached(quality)):
                    missing.append(field)
                elif value < current:
                    patch[key] = max(0., value)
        key = 'clustering.max_anchor_pose_deviation'
        current, value = parameter(config.data, key), quality.get('pose_deviation')
        if current is not None and type(value) in (int,float) and current < value <= 1:
            patch[key] = value
        elif current is not None and (type(value) not in (int,float) or not math.isfinite(value)):
            missing.append('pose_deviation')
        if patch or missing:
            entries.append(dict(observation=row['id'], patch=patch, missing_measurements=sorted(set(missing))))
        for key,value in patch.items():
            choose = max if key.endswith('pose_deviation') else min
            joint[key] = choose(joint.get(key,value), value)
    return dict(observations=entries, proposals=([joint] if joint else []) + [e['patch'] for e in entries if e['patch']])


def face_fallbacks(db, config, goals, targets, limit=32, *, evidence=None):
    """Measure retained goal queries even when the body matcher cannot run.

    Query observers may never be invoked when every body bank is unavailable.
    These independent face hints cannot create person features or bypass guards.
    """
    import numpy as np
    from .clustering import _groups, vector
    from .reference_compatibility import fallback_sql
    from .scanning import scan_identity
    from .merge_solver_diagnostics import round_robin
    profiles = targets.get('destinations', {})
    priorities = targets.get('priorities', {})
    if not profiles or not priorities:
        return []
    membership = {r['id']: g['index'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
    banks = [g for g in _groups(db, goals['scope'], goals['namespace'], config.data['clustering'],
        scan_policy=scan_identity(config, goals['namespace']), compatibility=fallback_sql(db, config, goals['namespace']))
        if g.identifier in profiles]
    buckets = defaultdict(list)
    for row in db.execute('SELECT o.*,s.scene FROM observations o JOIN scans s ON s.id=o.scan '
            'WHERE o.rejected IS NULL AND o.id IN (SELECT value FROM json_each(?)) ORDER BY o.id',
            (encoded(sorted(priorities)),)):
        buckets[membership[row['id']], row['scene']].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda r: (-priorities[r['id']], r['id']))
    proposals, comparisons = {}, 0
    budget = config.data['sapiensid']['execution']['max_candidate_pairs']
    for row in round_robin(buckets):
        query = vector(row).astype('float64')
        for bank in banks:
            if profiles[bank.identifier]['set'] != membership[row['id']] or row['id'] in bank.sample_ids:
                continue
            if comparisons >= budget:
                return list(proposals.values())
            comparisons += 1
            value = float((np.asarray(bank.samples, dtype='float64') @ query).min())
            if not 0 <= value <= 1:
                continue
            patch = {k: min(parameter(config.data, k), _floor(value)) for k in (
                'recognition.match_threshold', 'clustering.cohesion_threshold',
                'clustering.view_match_threshold', 'clustering.view_cohesion_threshold')}
            if evidence is not None and len(evidence)<limit:
                evidence.append(dict(observation=row['id'],identity=bank.identifier,patch=patch))
            proposals[digest(patch)] = patch
            if len(proposals) >= limit:
                return list(proposals.values())
    return list(proposals.values())
