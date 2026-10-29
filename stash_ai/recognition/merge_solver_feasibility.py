"""Constructive reachability checks on frozen inputs, separate from goal rank.

A completed replay is a witness. Failure of permissive probes is not an
infeasibility proof. Certificates below cover immutable goal-input availability
only; observed matching blockers are explicitly not certificates.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from .database import encoded


class Feasibility:
    def __init__(self, db, config, goals, scans):
        from .body_identity import available, enabled
        from .body_observations import contract
        rows = [r for g in goals['groups'] for m in g['members'] for r in m['observations']]
        ids = {r['id'] for r in rows}
        selected = {s['id'] for s in scans}
        eligible = set()
        alternatives = defaultdict(list)
        for row in db.execute('SELECT id,scan,rejected,embedding IS NOT NULL AS ready FROM observations '
                'WHERE id IN (SELECT value FROM json_each(?))', (encoded(sorted(ids)),)):
            if row['scan'] in selected and row['rejected'] is None and row['ready']:
                eligible.add(row['id'])
        body_ids = set()
        if enabled(config) and available(db):
            for row in db.execute('SELECT b.id,b.face,b.scan,b.rejected,b.embedding IS NOT NULL AS ready,'
                "m.status,m.person,f.person AS face_person FROM body_observations b "
                'LEFT JOIN body_members m ON m.observation=b.id LEFT JOIN members f ON f.observation=b.face '
                'LEFT JOIN observations o ON o.id=b.face WHERE b.contract=? AND '
                '(b.id IN (SELECT value FROM json_each(?)) OR '
                '(b.face IN (SELECT value FROM json_each(?)) AND b.scan=o.scan))',
                (contract(config), encoded(sorted(ids)), encoded(sorted(ids)))):
                linked = (row['face'] in ids and row['status'] == 'accepted' and row['person'] == row['face_person']
                          and row['rejected'] is None and row['ready'])
                if row['id'] not in ids and not linked:
                    continue
                body_ids.add(row['id'])
                if linked:
                    alternatives[row['face']].append(row['id'])
                if row['scan'] in selected and row['rejected'] is None and row['ready']:
                    eligible.add(row['id'])
        self.addresses = ids | body_ids
        self.groups = []
        for group in goals['groups']:
            members = [r for m in group['members'] for r in m['observations']]
            evidence = deepcopy(group.get('identity_evidence', group.get('construction_evidence')))
            if evidence is None:
                # Construction for strict goals may use a linked body as the
                # same physical witness; strict goal acceptance still cannot.
                units = [dict(id=r['id'], observations=[dict(id=r['id'], scene=r['scene'])]) for r in members]
                scenes = sorted({r['scene'] for r in members})
                evidence = dict(core_units=units, core_required=len(units), soft_units=[], soft_loss_allowance=0,
                    scenes={s: dict(membership_units=[u for u in units if u['observations'][0]['scene'] == s],
                                    scoring_units=[u for u in units if u['observations'][0]['scene'] == s]) for s in scenes})
            collections = [evidence['core_units'], evidence['soft_units']]
            collections += [units for scene in evidence['scenes'].values() for units in scene.values()]
            seen = set()
            for units in collections:
                for unit in units:
                    if id(unit) in seen:
                        continue
                    seen.add(id(unit))
                    additions = [dict(id=oid, scene=r['scene']) for r in unit['observations']
                                 for oid in alternatives[r['id']]]
                    unit['observations'].extend(additions)
            addresses = {r['id'] for u in evidence['core_units']+evidence['soft_units'] for r in u['observations']}
            addresses.update(r['id'] for scene in evidence['scenes'].values()
                             for u in scene['membership_units'] for r in u['observations'])
            def available_unit(unit):
                return any(r['id'] in eligible for r in unit['observations'])
            certificates = []
            if not evidence['scenes']:
                certificates.append(dict(reason='empty_goal_set'))
            for scene, frozen in evidence['scenes'].items():
                if not any(available_unit(u) for u in frozen['membership_units']):
                    certificates.append(dict(reason='required_scene_has_no_available_goal_witness', scene=scene))
            core = sum(available_unit(u) for u in evidence['core_units'])
            if core < evidence['core_required']:
                certificates.append(dict(reason='insufficient_available_core_units', available=core,
                                         required=evidence['core_required']))
            self.groups.append(dict(index=group['index'], evidence=evidence, addresses=addresses,
                certificates=certificates, inputs=dict(goal_observations=len({r['id'] for r in members}),
                    available_addresses=len(addresses & eligible), body_addresses=len(addresses & body_ids),
                    linked_body_alternatives=len(addresses & body_ids - ids))))

    def assess(self, status, observations, predicted, trace):
        from .merge_goal_sets import assess_set
        groups = []
        for source, goal in zip(self.groups, status['groups']):
            destinations = {predicted[oid] for oid in source['addresses']
                            if oid in observations and predicted.get(oid) is not None}
            measured = assess_set(source['evidence'], observations, predicted, destinations)
            blockers = Counter((r['stage'], r['reason']) for r in trace
                if r.get('observation') in source['addresses'] or
                any(r.get(k) in destinations for k in ('left', 'right', 'target')))
            groups.append(dict(index=source['index'], merged=measured['identity_fully_merged'],
                goal_merged=goal.get('identity_fully_merged', goal['fully_merged']),
                target=measured['set_target'], fragments=len(destinations),
                required_scenes=measured['required_scenes'], covered_scenes=measured['covered_scenes'],
                scene_coverage_fraction=measured['scene_coverage_fraction'],
                shared_coverage=measured['set_shared_coverage'], pair_fraction=measured['set_pair_fraction'],
                core_required=measured['set_core_required'], core_support=max(measured['set_core_destinations'].values(), default=0),
                observed_blockers=[dict(stage=stage, reason=reason, count=count)
                                   for (stage, reason), count in blockers.most_common(16)]))
        merged = all(g['merged'] for g in groups)
        conflicts = (not status['disjoint'] or not status.get('scene_exclusions_satisfied', True)
                     or not status.get('outside_preserved', True))
        outcome = ('possible' if merged and status['fully_solved'] else
                   'possible_with_conflicts' if merged and conflicts else
                   'possible_goal_mismatch' if merged else 'not_demonstrated')
        return dict(status=outcome, merged=merged, goals_satisfied=status['fully_solved'], groups=groups)

    def report(self, trials, failures, budget):
        completed = {n: t['goal_result']['construction'] for n, t in trials.items()}
        joint = next((n for n, c in completed.items() if c['status'] == 'possible'), None)
        joint = joint or next((n for n, c in completed.items() if c['merged']), None)
        groups = []
        for source in self.groups:
            candidates = [(name, next(g for g in c['groups'] if g['index'] == source['index']))
                          for name, c in completed.items()]
            best = max(candidates, key=lambda v: (v[1]['merged'], v[1]['shared_coverage'],
                       v[1]['covered_scenes'], v[1]['core_support']), default=None)
            proven = bool(best and best[1]['merged'])
            groups.append(dict(index=source['index'], status=('possible' if proven else
                'blocked_inputs' if source['certificates'] else 'not_demonstrated'),
                witness_trial=best[0] if proven else None, best_trial=best[0] if best else None,
                inputs=source['inputs'], input_certificates=source['certificates'],
                best_observation=best[1] if best else None))
        status = (completed[joint]['status'] if joint else 'blocked_inputs'
                  if any(g['input_certificates'] for g in groups) else 'not_demonstrated')
        return dict(format=1, status=status, joint_witness_trial=joint, groups=groups,
            probe_budget=budget, probes_completed=sum(t.get('search_phase') == 'feasibility' for t in trials.values()),
            probes_failed=len(failures), failed_probes=failures,
            certificate_scope='Frozen goal witnesses in selected retained scans and enabled evidence routes only.',
            interpretation='Completed production replays prove reachability. Unsuccessful or truncated probes do not prove impossibility. '
                           'Per-set witnesses from different trials do not prove one jointly feasible configuration. '
                           'Observed matching blockers are trial-specific, not input certificates.')


def resource_limit(exc):
    """Only known resource guards may turn an early probe into an unknown result."""
    return isinstance(exc, ValueError) and 'exceed' in str(exc).lower() and any(key in str(exc) for key in (
        'reference_bytes', 'plan_bytes', 'max_candidate_pairs', 'max_witness_comparisons',
        'rebuild_memory_bytes', 'configured person budget', 'configured budget'))


def lines(report):
    result = [f"Merge feasibility: {report['status']}; witness={report['joint_witness_trial'] or 'none'}; "
              f"probes={report['probes_completed']}/{report['probe_budget']}; failed={report['probes_failed']}."]
    for group in report['groups']:
        best = group['best_observation']
        detail = (f" scenes={best['covered_scenes']}/{best['required_scenes']} "
                  f"core={best['core_support']}/{best['core_required']}" if best else '')
        result.append(f"  Set {group['index']+1}: {group['status']} "
                      f"witness={group['witness_trial'] or 'none'}{detail}")
        for certificate in group['input_certificates']:
            result.append('    Input block: '+str(certificate))
        for blocker in (best or {}).get('observed_blockers', ())[:5]:
            result.append(f"    Observed blocker: {blocker['stage']}/{blocker['reason']} count={blocker['count']}")
    return result
