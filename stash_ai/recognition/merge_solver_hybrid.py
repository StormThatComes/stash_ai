"""Measured hybrid search hints; only a full production replay certifies them."""
from .resource_errors import ResourceLimit

from .database import encoded
from .hybrid_evidence import catalogue, pair, scores
from .hybrid_modes import competitor_score
from .merge_report_policy import _floor

PREFIX = 'sapiensid.matching.'


def boundaries(value):
    from .threshold_boundaries import sides
    return sides(value)


def mode_boundary(left, right):
    """Strongest threshold covering every supported mode on both sides.

    Keep one maximum per mode, not a full mode cross-product. Using the minimum
    of every cross-score would unnecessarily collapse different clothing looks.
    Original-core guards are checked separately by the production pair policy.
    """
    if not len(left) or not len(right):
        return None
    rows, columns = [-1.] * len(left), [-1.] * len(right)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            measurement = scores(a, b)
            if measurement is not None:
                rows[i] = max(rows[i], measurement['minimum'])
                columns[j] = max(columns[j], measurement['minimum'])
    return min(rows + columns)


def analyse(db, config, goals):
    """Inspect target pairs and every incident competitor, never outsider pairs.

    Goal sets only direct the search. They neither enter production matching nor
    turn existing memberships into truth. Missing features stay missing.
    """
    if not config.data['sapiensid']['enabled']:
        return dict(enabled=False, options={}, proposals=[])
    settings = config.data['sapiensid']['matching']
    limits = config.data['sapiensid']['execution']
    owners = {}
    for index, group in enumerate(goals['groups']):
        for member in group['members']:
            owners.setdefault(member['identity_id'], set()).add(group.get('index', index))
    bank = sorted(catalogue(db, config, goals['scope'], goals['namespace']), key=lambda i: i.person)
    by_id = {i.person: i for i in bank}
    wanted = set(owners)
    options = {PREFIX + key: set() for key in settings}
    result = dict(enabled=True, options=options, proposals=[], pairs=[], nearest_competitors=[],
        missing_identities=sorted(wanted - by_id.keys()), identities=[], comparisons=0, candidate_pairs=0,
        limitations=['Frozen hybrid measurements are search hints, not a merge guarantee.'])
    for identity in bank:
        if identity.person in wanted:
            result['identities'].append(dict(identity=identity.person, sets=sorted(owners[identity.person]),
                face_references=len(identity.faces), person_references=len(identity.persons),
                independent_support=len(identity.independent), body_modes=len(identity.body_modes),
                missing_person_observations=list(identity.missing),
                missing_original_core_observations=list(identity.guard_missing)))

    from .hybrid_lookup import Lookup
    from .global_views import _cooccurs
    from .progress import tracking
    from .memory_size import Reservation, size
    reservation = Reservation(limits['plan_bytes'])
    def reserve(required):
        if required > limits['plan_bytes']:
            raise ResourceLimit('SapiensID plan_bytes budget exceeded by solver evidence.')
        reservation.set(required)
    lookup = Lookup(bank, limits['reference_bytes'])
    nearest, positive = {}, []
    proof_bytes = 0
    with tracking(lookup.pairs(incident=wanted), 'Measure solver hybrid evidence', unit='pair') as pairs:
        for a, b, face_mean, body_mean in pairs:
            result['comparisons'] += 1
            left, right = by_id[a], by_id[b]
            target_pair = a in wanted and b in wanted
            same_set = target_pair and owners[a] == owners[b] and len(owners[a]) == 1
            nominated = ((face_mean is not None and face_mean + 1e-6 >= settings['face_floor']) or
                (body_mean is not None and body_mean + 1e-6 >= settings['person_threshold'] - settings['ambiguity_margin']))
            if target_pair or nominated:
                result['candidate_pairs'] += 1
                if result['candidate_pairs'] > limits['max_candidate_pairs']:
                    raise ResourceLimit('SapiensID max_candidate_pairs budget exceeded by solver evidence.')
            present = _cooccurs(db, a, b, config) if target_pair else False
            proof = (pair(left, right, settings, copresent=present,
                          reflection_threshold=config.data['recognition']['cooccurrence_merge_threshold'])
                     if target_pair or nominated else None)
            if target_pair:
                boundary = mode_boundary(left.body_modes, right.body_modes)
                alternate = None
                if boundaries(boundary):
                    alternate = pair(left, right, dict(settings, person_threshold=_floor(boundary)),
                        copresent=present, reflection_threshold=config.data['recognition']['cooccurrence_merge_threshold'])
                entry = dict(proof, relation='within_set' if same_set else 'between_sets',
                    sets=[sorted(owners[a]), sorted(owners[b])], mode_boundary=boundary,
                    mode_boundary_evidence=alternate)
                proof_bytes += len(encoded(entry).encode()) * 2 + 1024
                if proof_bytes > limits['plan_bytes']:
                    raise ResourceLimit('SapiensID plan_bytes budget exceeded by solver evidence.')
                result['pairs'].append(entry)
                for measured in (proof, alternate):
                    if measured is None:
                        continue
                    for field, signal in (('face_floor', 'face'), ('person_threshold', 'person')):
                        if measured[signal] is not None:
                            options[PREFIX + field].update(boundaries(measured[signal]['minimum']))
                    support = min(measured['support'])
                    if support:
                        options[PREFIX + 'min_support'].update((support, support + 1))
                if same_set:
                    positive.append(entry)
            # Match production competitor semantics: missing body evidence can
            # still compete through a strong face; known face contradictions do not.
            from .competitor_eligibility import hybrid as eligible_rival
            if same_set or not eligible_rival(proof, face_mean, settings['face_floor']):
                continue
            score = competitor_score(left, right, body_mean)
            if score is None and face_mean is not None and face_mean >= config.data['recognition']['match_threshold']:
                score = 1.
            if score is None:
                continue
            for person, rival in ((a, b), (b, a)):
                if person not in wanted:
                    continue
                previous = nearest.get(person)
                if previous is None or (-score, rival) < (-previous['score'], previous['rival']):
                    nearest[person] = dict(identity=person, rival=rival, score=score,
                        relation='other_set' if rival in wanted else 'outside_goals')
            reserve(proof_bytes + size(nearest) + size(options) + size(result['identities']))

    def changes(proof):
        support = min(proof['support'])
        if (not support or 'missing_original_core_evidence' in proof['blockers'] or
                any(proof[k] is None or not boundaries(proof[k]['minimum']) for k in ('face', 'person'))):
            return None
        patch = {PREFIX + field: min(settings[field], _floor(proof[signal]['minimum']))
                 for field, signal in (('face_floor', 'face'), ('person_threshold', 'person'))}
        patch[PREFIX + 'min_support'] = min(settings['min_support'], support)
        gaps = [proof['person']['mean'] - nearest[p]['score'] for p in (proof['left'], proof['right']) if p in nearest]
        if gaps:
            gap = min(gaps)
            options[PREFIX + 'ambiguity_margin'].update(boundaries(gap))
            if gap >= 0:
                patch[PREFIX + 'ambiguity_margin'] = min(settings['ambiguity_margin'], _floor(gap))
        if proof['cooccurring']:
            patch['recognition.cooccurrence_merge_threshold'] = min(
                config.data['recognition']['cooccurrence_merge_threshold'], _floor(proof['face']['minimum']))
        return patch

    individual, preferred = [], []
    for entry in positive:
        # Prefer a complete supported-mode route over the raw cross-look minimum.
        variants = [p for p in (entry['mode_boundary_evidence'], entry) if p is not None]
        variants.sort(key=lambda p: -(p['person']['minimum'] if p['person'] else -1.))
        patches = [p for proof in variants if (p := changes(proof)) is not None]
        if patches:
            preferred.append(patches[0])
            individual.extend(patches)
    if preferred:
        # A shared configuration must clear every within-set gate, across all
        # requested sets. Partial pair proposals follow if this is too permissive.
        joint = {}
        for patch in preferred:
            for key, value in patch.items():
                joint[key] = min(joint.get(key, value), value)
        result['proposals'].append(joint)
        options[PREFIX + 'ambiguity_margin'].update((0., settings['ambiguity_margin'] / 2,
                                                    settings['ambiguity_margin'] / 4))
        options[PREFIX + 'min_support'].update((1, max(1, settings['min_support'] - 1)))
    result['proposals'].extend(individual)
    result['nearest_competitors'] = [nearest[p] for p in sorted(nearest)]
    result['options'] = {k: sorted(v, reverse=True) for k, v in options.items()}
    reserve(size(result))
    return result


def public(analysis):
    return {key: value for key, value in analysis.items() if key not in ('options', 'proposals')}
