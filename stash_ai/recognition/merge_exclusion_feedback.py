"""Bounded rejection-boundary hints from explicit negative scenes.

Frozen vectors only steer proposals. Positive preservation and the full negative
scene, including observations outside this sample, are checked by every replay.
"""
from collections import defaultdict

from .config import digest
from .database import encoded
from .merge_report_policy import parameter
from .threshold_boundaries import strictly_above


def analyse(db, config, goals):
    from .body_identity import available, enabled, vector as body_vector
    from .body_observations import contract as body_contract
    from .clustering import vector as face_vector
    from .person_signal import available as person_available, contract as person_contract
    from .merge_solver_domains import registry
    from .merge_solver_measurements import reference_stats
    from .resource_plan import plan
    domains = registry(config)
    budget = config.data['sapiensid']['execution']['max_candidate_pairs']
    working = min(4194304, plan(config).working)
    options, proposals, evidence = defaultdict(set), {}, []
    comparisons = 0
    truncated = False
    modes = [('face', 'observations', '', '', (), face_vector,
        ('recognition.match_threshold', 'clustering.cohesion_threshold',
         'clustering.view_match_threshold', 'clustering.view_cohesion_threshold',
         'sapiensid.matching.face_floor'))]
    if config.data['sapiensid']['enabled'] and person_available(db):
        modes.append(('hybrid_person', 'person_evidence', 'observation',
            " AND o.contract=? AND o.status='ready'", (person_contract(config),), body_vector,
            ('sapiensid.matching.person_threshold',)))
    if enabled(config) and available(db):
        modes.append(('person', 'body_observations', '', ' AND o.contract=?',
            (body_contract(config),), body_vector,
            tuple('sapiensid.'+section+'.'+key for section in ('identities', 'global_matching')
                  for key in ('match_threshold', 'cohesion_threshold'))))
    # Keep at most 32 vectors on either side; repeated exclusions share no
    # unbounded corpus matrix. The same global pair budget bounds all modes.
    def bank(ids, table, oid, where, params, vector, limit):
        column = oid or 'id'
        guard = '' if table == 'person_evidence' else ' AND o.rejected IS NULL'
        rows = db.execute(f'SELECT o.* FROM {table} o WHERE o.{column} IN '
            f'(SELECT value FROM json_each(?)) AND o.embedding IS NOT NULL{guard}{where} '
            f'ORDER BY o.{column} LIMIT ?', (encoded(ids), *params, limit+1)).fetchall()
        return [vector(r) for r in rows[:limit]], len(rows) > limit
    for item in goals.get('scene_exclusions', ()):
        joint = {}
        for kind, table, oid, where, params, vector, keys in modes:
            keys = [k for k in keys if k in domains]
            if not keys:
                continue
            remaining = budget-comparisons
            if remaining <= 0:
                truncated = True
                break
            limit = min(32, max(1, int(remaining**.5)))
            left, a = bank(item['target_observation_ids'], table, oid, where, params, vector, limit)
            right, b = bank(item['observation_ids'], table, oid, where, params, vector, limit)
            truncated |= a or b
            if not left or not right:
                continue
            low, mean, high, _ = reference_stats(left, right, max_bytes=working)
            comparisons += len(left)*len(right)
            evidence.append(dict(set=item['set'], scene=item['scene'], evidence_kind=kind,
                minimum=low, mean=mean, maximum=high, positive_samples=len(left), negative_samples=len(right),
                sampled=a or b))
            for score in (high, mean, low):
                value = strictly_above(score)
                patch = {k: max(parameter(config.data, k), value) for k in keys
                         if domains[k].low <= value <= domains[k].high}
                if patch:
                    proposals[digest(patch)] = patch
                    for k, v in patch.items():
                        options[k].add(v)
                    if score == high:
                        joint.update(patch)
        if joint:
            # Close face, hybrid and independent body routes together too.
            proposals = {digest(joint): joint, **proposals}
    return dict(proposals=list(proposals.values()), options={k: sorted(v) for k, v in options.items()},
        evidence=evidence, comparisons=comparisons, truncated=truncated,
        interpretation='Sampled rejection boundaries; all exclusions require full production replay.')
