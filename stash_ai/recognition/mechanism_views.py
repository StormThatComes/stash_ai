"""Bounded diagnoses of missing strict View prerequisites, never assignments."""
from collections import defaultdict
from itertools import islice

from .config import digest
from .clustering import vector
from .database import encoded
from .independent_evidence import representatives
from .merge_report_policy import patched
from .pose_policy import is_anchor, restricted
from .threshold_boundaries import at_or_below
from .view_prototypes import add_cohort, witness_limit

MAX_ROWS = 128
MAX_COHORTS = 16


def load_rows(db, config, scope, namespace, addresses):
    """Callers choose bounded addresses; old scans and foreign models never vote."""
    from .reference_roles import strict_sql, verified_sql
    from .scan_heads import predicate
    from .scanning import scan_identity
    from .reference_compatibility import fallback_sql
    return [dict(r) for r in db.execute(
        'SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality,s.scene,m.person,'+
        strict_sql()+' AS strict_ok,'+verified_sql()+' AS verified_ok FROM observations o '
        'JOIN scans s ON s.id=o.scan LEFT JOIN members m ON m.observation=o.id '
        "WHERE o.id IN (SELECT value FROM json_each(?)) AND s.scope=? AND s.namespace=? "
        "AND s.status='complete' AND o.rejected IS NULL AND o.embedding IS NOT NULL AND "+predicate()+
        ' AND (s.policy_sha=? OR '+fallback_sql(db, config, namespace)+') '
        'ORDER BY s.scene,o.cue,o.seconds,o.id',
        (encoded(addresses), scope, namespace, scan_identity(config, namespace)))]


def diagnose(config, rows, *, quality=()):
    """Use production pose grouping and independent timestamp/pixel witnesses.

Different current owners are never certified as one reference. Unassigned
samples can suggest a construction experiment, but cannot claim authority.
    """
    import numpy as np
    from .run_cache import metadata
    grouping = config.data['clustering']
    quality = {r['observation']:r for r in quality}
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get('set', 0), row.get('source', row.get('person')), row['scan']].append(row)
    output = []
    for bucket in buckets.values():
        # A relaxed complete-link group offers measured boundaries, not a
        # transitive chain. Cap the evidence before quadratic comparisons.
        slots = []
        for row in bucket[:32]:
            add_cohort(slots, row, vector(row), 0., complete=True, maximum=32)
        for slot in slots:
            if slot.needs_final_selection:
                slot._resolve(slot.maximum)
            ids = set(slot.observations)
            witnesses = list(representatives(r for r in bucket if r['id'] in ids))
            if len(witnesses) < 2:
                continue
            witnesses = witnesses[:min(8, witness_limit(grouping))]
            samples = np.asarray([vector(r) for r in witnesses], dtype='float64')
            minimum = float((samples @ samples.T).min())
            if not np.isfinite(minimum) or minimum < 0:
                continue
            admission, missing = {}, set()
            for row in witnesses:
                hint = quality.get(row['id'], {})
                missing.update(hint.get('missing_measurements', ()))
                record = metadata(row['quality'])
                if record.get('alignment_valid') is False or record.get('query_only'):
                    missing.add('invalid_alignment_or_query_only')
                for key, value in hint.get('patch', {}).items():
                    choose = max if key.endswith('pose_deviation') else min
                    admission[key] = choose(admission.get(key, value), value)
            count = len(witnesses)
            patch = dict(admission, **{
                'clustering.anchor_view_matching': True,
                'clustering.view_cohesion_threshold': min(grouping['view_cohesion_threshold'], at_or_below(minimum)),
                'clustering.view_min_support': min(grouping['view_min_support'], count),
                'clustering.same_scene_min_support': min(grouping['same_scene_min_support'], count),
                'recognition.min_identity_support': min(config.data['recognition']['min_identity_support'], count)})
            blockers = sorted(missing)
            eligible = all(is_anchor(r['quality'], grouping) for r in witnesses)
            owners = {r.get('person') for r in witnesses}
            rooted = eligible and len(owners) == 1 and None not in owners and all(r['strict_ok'] for r in witnesses)
            verified = len(owners) == 1 and None not in owners and all(r.get('verified_ok') for r in witnesses)
            support_ok = count >= max(grouping['view_min_support'], grouping['same_scene_min_support'])
            coherent = minimum >= grouping['view_cohesion_threshold']
            ready = rooted and support_ok and coherent and grouping['anchor_view_matching']
            if not eligible: blockers.append('anchor_admission')
            if not rooted: blockers.append('strict_reference_authority')
            if not support_ok: blockers.append('independent_support')
            if not coherent: blockers.append('view_cohesion')
            if not grouping['anchor_view_matching']: blockers.append('anchor_view_matching_disabled')
            try:
                variant = patched(config, patch)
                if not restricted(variant.data['clustering']) or not all(
                        is_anchor(r['quality'], variant.data['clustering']) for r in witnesses):
                    missing.add('unavailable_anchor_evidence')
            except ValueError:
                missing.add('incompatible_formation_settings')
            keys = sorted(r['id'] for r in witnesses)
            output.append(dict(key=digest(keys), set=witnesses[0].get('set', 0),
                witnesses=keys, scene=witnesses[0]['scene'], pose=slot.pose,
                independent_support=count, required_support=max(grouping['view_min_support'],grouping['same_scene_min_support']),
                minimum_similarity=minimum, blockers=sorted(set(blockers)|missing),
                admitted=eligible, rooted=rooted, supported=ready,
                verified=verified,
                patch={} if missing or ready else patch,
                interpretation='Supported strict cohort on sampled retained witnesses; full replay decides emitted Views and ownership.'))
            if len(output) >= MAX_COHORTS:
                return output
    return output


def for_goals(db, config, goals, quality, *, priorities=None):
    from .merge_solver_diagnostics import round_robin
    buckets = defaultdict(list)
    eligible = 0
    for group in goals['groups']:
        for member in group['members']:
            for row in member['observations']:
                if row.get('evidence_kind', 'face') == 'face':
                    eligible += 1
                    bucket = buckets[group['index'], member['identity_id'], row['scene']]
                    if len(bucket) < 8:
                        bucket.append(row['id'])
    by_set = defaultdict(list)
    for key, ids in buckets.items():
        if len(ids) >= 2:
            by_set[key[0]].append(key)
    priorities = priorities or {}
    for keys in by_set.values():
        keys.sort(key=lambda key:(-max(priorities.get(oid,0) for oid in buckets[key]),key))
    selected_buckets = list(islice(round_robin(by_set), MAX_COHORTS))
    labels = {oid:(index,source) for index,source,scene in selected_buckets
              for oid in buckets[index,source,scene]}
    selected = list(labels)[:MAX_ROWS]
    rows = load_rows(db, config, goals['scope'], goals['namespace'], selected)
    for row in rows:
        row['set'], row['source'] = labels[row['id']]
    cohorts = diagnose(config, rows, quality=quality)
    return dict(cohorts=cohorts, sampled=len(rows),
                truncated=eligible>len(selected) or len(cohorts)>=MAX_COHORTS,
                missing_retained=len(selected)-len(rows))
