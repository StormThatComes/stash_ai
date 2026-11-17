"""Pair-only counterfactuals on the inspector's frozen production reference banks."""

from dataclasses import replace

from .database import encoded
from .global_views import cohort_evidence
from .independent_evidence import unit
from .inspection_advice import _path
from .inspection_pair_evidence import witness
from .pose_policy import restricted
from .inspection_runtime import check_cancelled
from .merge_report_policy import _planner_config, check
from .reference_usage import ReferenceUsage
from .strict_partition import summary_views


def _without(view, omitted, config):
    keep = [(k, v) for k, v in zip(view.support, view.examples) if k not in omitted]
    if not keep:
        return None
    ids, samples = zip(*keep)
    group = replace(view.group, samples=samples, sample_ids=ids)
    result = summary_views([group], ReferenceUsage(config.data['clustering']['max_reference_bytes']))
    return next(iter(result), None)


def anchor_influence(db, config, ids, summaries):
    route = dict(
        title='Isolate an influential strict anchor',
        status='Frozen diagnostic',
        patches=[],
        notes=[],
        witnesses=[],
        strategy='anchor_influence',
    )
    owned = {v.group.identifier: v for v in summaries if v.group.identifier in ids}
    if len(owned) != 2:
        route['notes'].append('Both identities need retained strict face references.')
        return route
    effective = _planner_config(config, 'strict_core')
    left, right = (owned[p] for p in ids)
    original = cohort_evidence(db, left, right, effective)
    route['notes'].append(
        f'Original cross minimum {original["minimum_cross_similarity"]:.4f}, '
        f'mean {original["mean_cross_similarity"]:.4f}. Each trial removes reference authority only.'
    )
    best = None
    count = 0
    for source, target in ((left, right), (right, left)):
        rows = db.execute(
            'SELECT id,scan,seconds,quality FROM observations WHERE id IN (SELECT value FROM json_each(?))',
            (encoded(source.support),),
        )
        aliases = {}
        for row in rows:
            aliases.setdefault(unit(row), []).append(row['id'])
        groups = [(key,) for key in source.support]
        groups.extend(tuple(sorted(keys)) for keys in aliases.values() if len(keys) > 1)
        for omit in groups:
            check_cancelled()
            reduced = _without(source, omit, config)
            if reduced is None:
                continue
            count += 1
            proof = cohort_evidence(
                db, reduced, target, effective, cooccurs=lambda *_: original['cooccurring']
            )
            rank = (
                -len(proof['blockers']),
                proof['minimum_cross_similarity'],
                proof['mean_cross_similarity'],
            )
            if best is None or rank > best[0]:
                best = rank, omit, source, reduced, target, proof
    route['notes'].append(f'{count} single-anchor or repeated-crop exclusions tested on frozen references.')
    if best:
        _, omit, source, reduced, target, proof = best
        # Validate the best numerical counterfactual against the original competing banks.
        trial = tuple(reduced if v is source else v for v in summaries)
        checked = check(db, trial, config, ids, 'strict_core')
        route['notes'].extend(
            (
                f'Best exclusion ({len(omit)}): {", ".join(omit)}. '
                f'Cross minimum {proof["minimum_cross_similarity"]:.4f}, '
                f'mean {proof["mean_cross_similarity"]:.4f}.',
                'Frozen component: '
                + (
                    'accepted'
                    if checked['can_merge_in_frozen_catalogue']
                    else ', '.join(checked['blockers'])
                ),
                'Diagnostic only. Final ownership and unassigned counts require a rebuild.',
            )
        )
        patch = _path(config, 'Replay with the selected anchors restricted to queries')
        patch.update(
            omit_face_anchors=list(omit),
            watch=list(omit),
            evidence=['Private diagnostic replay. The omitted anchors can still attach as queries.'],
        )
        route['patches'].append(patch)
        for key in omit:
            vector = source.examples[source.support.index(key)]
            other = min(zip(target.support, target.examples), key=lambda item: float(vector @ item[1]))[0]
            route['witnesses'].append(
                witness('Omitted anchor / weakest opposite reference', 'face', key, other)
            )
    return route


def _rank_views(middle, bank):
    return sorted(
        (
            (float(middle.group.samples[0] @ v.group.samples[0]), v)
            for v in bank
            if v is not middle and v.pose != middle.pose and 'unknown' not in (v.pose, middle.pose)
        ),
        key=lambda pair: (-pair[0], pair[1].medoid),
    )


def view_bridges(db, config, ids, views, component, summaries):
    route = dict(
        title='Bridge appearances through an intermediate View',
        strategy='view_bridge',
        status='Frozen diagnostic',
        patches=[],
        witnesses=[],
        notes=[],
    )
    sides = [[v for v in views if v.group.identifier == p] for p in ids]
    controls = []
    nearest = None
    best = None
    # Focus only on A/B vs the catalogue. Keep a single candidate triple, never an N-view graph.
    for middle in views:
        check_cancelled()
        outside = middle.group.identifier not in ids
        if outside:
            for bank in sides:
                for v in bank:
                    entry = (float(middle.group.samples[0] @ v.group.samples[0]), middle.group.identifier)
                    if nearest is None or entry > nearest:
                        nearest = entry
        a, b = (_rank_views(middle, bank) for bank in sides)
        if a and b:
            rank = min(a[0][0], b[0][0]), a[0][0] + b[0][0], middle.medoid
            if best is None or rank > best[0]:
                best = rank, a[0][1], middle, b[0][1]
    if nearest:
        controls.append(nearest[1])
    if not best:
        route['notes'].append(
            'No independently supported intermediate appearance distinct from both endpoint Views.'
        )
        return route, controls
    _, left, middle, right = best
    route['notes'].append(
        f'Intermediate View {middle.medoid} ({middle.pose}), owner {middle.group.identifier}.'
    )
    support_ids = set(left.support) | set(middle.support) | set(right.support)
    units = {
        row['id']: unit(row)
        for row in db.execute(
            'SELECT id,scan,seconds,quality FROM observations WHERE id IN (SELECT value FROM json_each(?))',
            (encoded(sorted(support_ids)),),
        )
    }
    for a, b, title in (
        (left, middle, 'A / intermediate'),
        (middle, right, 'Intermediate / B'),
        (left, right, 'A / B'),
    ):
        same_owner = a.group.identifier == b.group.identifier
        proof = cohort_evidence(db, a, b, config, cooccurs=(lambda *_: False) if same_owner else None)
        overlap = bool(
            {units.get(k, k) for k in a.support}.intersection(units.get(k, k) for k in b.support)
        )
        blockers = list(proof['blockers'])
        if overlap:
            blockers.append('shared_reference')
        route['notes'].append(
            f'{title}: {a.pose} / {b.pose}, support {a.support_count}/{b.support_count}, '
            f'weakest {proof["minimum_cross_similarity"]:.4f}, '
            + (', '.join(blockers) or 'pair gates pass')
        )
        route['witnesses'].append(witness(title, 'face', a.medoid, b.medoid))
    strict = {v.group.identifier: v for v in summaries}
    for endpoint in ids:
        if endpoint in strict:
            weak = min(float(a @ b) for a in middle.examples for b in strict[endpoint].examples)
            route['notes'].append(f'Intermediate View vs original core {endpoint}: minimum {weak:.4f}.')
    if nearest:
        route['notes'].append(f'Nearest outside View by medoid: {nearest[1]} ({nearest[0]:.4f}).')
    if restricted(config.data['clustering']):
        option = _path(
            config,
            'Replay confirmed and corroborated Views',
            {
                'clustering.anchor_view_matching': True,
                'clustering.corroborated_view_enrichment': True,
                'clustering.auxiliary_matching': True,
                'clustering.view_prototypes_enabled': True,
            },
        )
        if not option['notes']:
            option['controls'] = [middle.group.identifier] if middle.group.identifier not in ids else []
            route['patches'].append(option)
    passed = component and component['outcome'] == 'merge'
    route['notes'].append(
        'Direct A/B component in the frozen production planner: '
        + ('accepted.' if passed else 'not accepted; a two-edge chain cannot authorize a merge.')
    )
    return route, controls


def face_catalogue(db, config, ids, summaries, views, view_current):
    routes = [anchor_influence(db, config, ids, summaries)]
    route, view_controls = view_bridges(
        db, config, ids, views, view_current.get('shared_component'), summaries
    )
    routes.append(route)
    focused = [v for v in summaries if v.group.identifier in ids]
    best = None
    for v in summaries:
        check_cancelled()
        if v.group.identifier not in ids:
            for endpoint in focused:
                candidate = (float(endpoint.group.samples[0] @ v.group.samples[0]), v.group.identifier)
                if best is None or candidate > best:
                    best = candidate
    return routes, dict(face=[best[1]] if best else [], view=view_controls)


def body_anchor_options(db, config, ids, scan, route, local):
    """Any exclusion that improves the cross minimum must remove a weakest endpoint."""
    from .appearance_index import canonical_buckets
    from .body_identity import policy, quality
    from .body_observations import contract

    refs = next(
        (w['refs'] for w in local.get('witnesses', ()) if w['label'] == 'Weakest cross-core similarity'), ()
    )
    if not refs:
        return
    settings = config.data['sapiensid']['identities']
    rows = list(
        db.execute(
            'SELECT b.id,b.quality,m.person FROM body_observations b JOIN body_members m ON m.observation=b.id '
            "WHERE b.scan=? AND b.contract=? AND m.policy=? AND m.person IN (?,?) AND m.status='accepted' "
            "AND m.role='anchor' AND json_extract(m.proof,'$.reason')='body_scene_core' ORDER BY b.cue,b.id",
            (scan['id'], contract(config), policy(config), *ids),
        )
    )
    for _, key in refs:
        source = next((r for r in rows if r['id'] == key), None)
        if source is None:
            continue
        bank = [r for r in rows if r['person'] == source['person']]
        fingerprints = {}
        hashes = {}
        for row in bank:
            q = quality(row)
            fingerprints[row['id']] = q.get('pixel_digest')
            if q.get('appearance_hash'):
                hashes[row['id']] = int(q['appearance_hash'], 16)
        distance = settings['min_appearance_distance']
        classes = canonical_buckets(set(hashes.values()), distance) if distance else {}
        cohort = [
            k
            for k, pixel in fingerprints.items()
            if k == key
            or (pixel and pixel == fingerprints[key])
            or (classes and k in hashes and key in hashes and classes[hashes[k]] == classes[hashes[key]])
        ]
        for omitted in ([key], cohort) if len(cohort) > 1 else ([key],):
            if len(omitted) == len(bank):
                continue
            patch = _path(
                config, f'Replay without body anchor authority for {key} ({len(omitted)} samples)'
            )
            patch.update(
                omit_body_anchors=omitted,
                watch=omitted,
                evidence=[
                    'The local body planner reconsiders these observations as queries on a private replay.'
                ],
            )
            route['patches'].append(patch)
    route['witnesses'].extend(local['witnesses'])
    route['notes'].append(
        'Body ablations test each endpoint of the weakest cross-core comparison and its repeated appearances.'
    )
