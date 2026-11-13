"""Conservative tightening hypotheses for accepted memberships, without replay."""
from .admission_report import check
from .inspection_advice import RESCAN, _name, _number, _path
from .merge_report_policy import parameter
from .threshold_boundaries import at_or_above, at_or_below, strictly_above


def _tighten(config, path, measured, *, bonus=0., operator='>='):
    """Reject an inclusive measured gate; never relax an already stricter YAML."""
    if not _number(measured):
        return {}
    current = parameter(config.data, path)
    enabled = current is not None and not (operator == '>=' and current == 0)
    effective = max(0., current-bonus) if current is not None and bonus else current
    if check(path, measured, effective, operator, enabled=enabled)['status'] == 'fail':
        return {}
    integer = path.endswith(('_pixels', '_keypoints', '_support'))
    if operator == '<=':
        value = at_or_below(measured)
        if value >= measured:
            value = round(value-.000001, 6)
    elif operator == '>':
        value = at_or_above(measured)
    else:
        value = strictly_above(measured+bonus, integer=integer)
    if value < 0:
        return {}
    changes = {path:value}
    # Keep schema dependencies valid by tightening the dependent gate too.
    dependent = None
    if path == 'clustering.cohesion_threshold':
        dependent = 'recognition.match_threshold'
    elif path.startswith('clustering.min_auxiliary_'):
        dependent = path.replace('min_auxiliary_', 'min_anchor_')
    if dependent:
        old = parameter(config.data, dependent)
        if old is not None and old < value:
            changes[dependent] = value
    if path.startswith('detection.min_'):
        suffix = path.removeprefix('detection.min_')
        for role in ('anchor', 'auxiliary'):
            dependent = f'clustering.min_{role}_{suffix}'
            old = config.data['clustering'].get(dependent.split('.')[-1])
            if old is not None and old < value:
                changes[dependent] = value
    if path == 'detection.max_pose_deviation':
        old = parameter(config.data, 'clustering.max_anchor_pose_deviation')
        if old is not None and old > value:
            changes['clustering.max_anchor_pose_deviation'] = value
    return changes


def _boundary(config, title, path, measured, *, bonus=0., evidence=(), notes=()):
    if path is None:
        return None
    changes = _tighten(config, path, measured, bonus=bonus)
    if not changes:
        return None
    result = _path(config, title, changes,
        evidence=[*evidence, f'Retained boundary: {measured}; current {_name(config,path)}: {parameter(config.data,path)}.'],
        notes=[*notes, 'This blocks the recorded gate with unchanged evidence, not every possible assignment route. Rebuild to test final ownership.'])
    # No impossible >1 similarity patches, or empty "solutions" after validation.
    return result if result['changes'] else None


def _matching(config, kind, trace):
    notes, paths = [], []
    if trace['evidence_status'] != 'recorded_current_policy':
        return [], ['No current-policy assignment proof is retained. Rebuild with the current YAML before selecting numeric matching changes.']
    steps = trace['decisions']
    if len(steps) != 1:
        return [], ['The retained assignment has multiple or missing decision paths. Inspect them separately; no single numeric matching boundary is assumed.']
    step = steps[0]
    reason, metrics = step['reason'], step['metrics']
    if step.get('merges') or step.get('merge_source') or step.get('final_reason', reason) != reason:
        notes.append('Membership was inherited through an identity merge. Inspect the merge chain and its witnesses; the original sample score is not a boundary for undoing that merge.')
        routes = {edge.get('route') for edge in step.get('merges', [])}
        if step.get('final_reason') == 'body_global_identity' or 'body_global' in routes:
            names = ['sapiensid.global_matching.'+key for key in ('match_threshold','cohesion_threshold','min_support')]
        elif step.get('final_reason') == 'hybrid_identity' or 'sapiensid_hybrid' in routes:
            names = ['sapiensid.matching.'+key for key in ('face_floor','person_threshold','ambiguity_margin')]
        else:
            names = []
        if names:
            notes.append('Parameters to review for this merge (no measured rejection boundary): '+
                         ', '.join(f'{_name(config,key)} = {parameter(config.data,key)}' for key in names)+'.')
        return [], notes
    if kind == 'body' and reason == 'measured_face_association':
        return [], ['This body follows its linked assigned face. Inspect that face and the face/person association first; body similarity thresholds do not control this route.']
    if kind == 'face' and reason == 'candidate_group' and step.get('stage') != 'match':
        return [], [f'This observation started a candidate group, or its comparison stage was not retained. Review {_name(config,"recognition.min_identity_support")} and the other group members; required support alone is not a measured support count.']
    if kind == 'body' and reason == 'body_scene_core':
        support = metrics.get('independent_support')
        path = 'sapiensid.identities.min_support'
        if type(support) is int and 1 <= support < 8:
            candidate = _boundary(config, 'Require more independent evidence for this body core', path, support,
                notes=['This can make the whole core provisional, including its other members; it is not a sample-specific exclusion.'])
            if candidate:
                paths.append(candidate)
        return paths, [f'Review {_name(config,path)}. Numeric support proposals are capped at 8; missing support is never guessed.']
    if kind == 'face' and (reason in ('known_identity', 'candidate_group', 'auxiliary_identity', 'same_scene_identity')
                          or reason == 'hybrid_query' and metrics.get('signal') == 'face'):
        score, minimum = metrics.get('score'), metrics.get('minimum')
        match, cohesion, margin = 'recognition.match_threshold', 'clustering.cohesion_threshold', 'recognition.ambiguity_margin'
        bonus = parameter(config.data, 'clustering.same_scene_match_bonus') if reason == 'same_scene_identity' else 0.
        if bonus:
            notes.append(f'The same-scene route subtracts its {bonus} bonus from match/cohesion thresholds. Proposed base thresholds include this bonus.')
    elif kind == 'face' and reason == 'anchor_view_identity':
        score, minimum = metrics.get('match_score'), metrics.get('minimum')
        match, cohesion, margin = ('clustering.view_match_threshold', 'clustering.view_cohesion_threshold',
                                   'clustering.view_ambiguity_margin')
        bonus = 0.
    elif kind == 'face' and reason == 'hybrid_query' and metrics.get('signal') == 'face_and_body':
        score, minimum = metrics.get('score'), metrics.get('minimum')
        match, cohesion, margin = None, 'sapiensid.matching.person_threshold', 'sapiensid.matching.ambiguity_margin'
        bonus = 0.
        notes.append('This face attachment used a hybrid body comparison. Its minimum is a body score; it is not a measured face-floor boundary.')
    elif kind == 'body' and reason == 'local_body_track':
        score, minimum = metrics.get('track_floor'), None
        match, cohesion, margin = 'sapiensid.identities.track_similarity_threshold', None, 'sapiensid.identities.ambiguity_margin'
        bonus = 0.
        notes.append('The track floor is the weakest link on the recorded path to a root. Tightening it may break continuity for other members of this track.')
    elif kind == 'body' and reason in ('frozen_face_body_bank', 'body_core_attachment', 'body_confirmed_core_attachment'):
        score, minimum = metrics.get('score'), metrics.get('minimum')
        prefix = 'sapiensid.identities.'
        match, cohesion, margin = prefix+'match_threshold', prefix+'cohesion_threshold', prefix+'ambiguity_margin'
        bonus = 0.
        if 'certified_floor' in metrics and not _number(score):
            notes.append('The certified floor is only a lower bound, not the actual similarity. It cannot establish a rejection threshold.')
    else:
        return [], [f'Recorded route: {reason}. Its effective gates are not mapped to a numeric detachment patch. Inspect its references and retained evidence before tightening YAML.']
    evidence = [f'Recorded route: {reason}.']
    runner = metrics.get('rival_score') if reason == 'hybrid_query' else metrics.get('runner')
    for title, path, value, offset in (
            ('Require a stronger similarity match', match, score, bonus),
            ('Require stronger agreement with the weakest witness', cohesion, minimum, bonus),
            ('Require a larger lead over the competing identity', margin,
             score-runner if _number(score) and _number(runner) else None, 0.)):
        candidate = _boundary(config, title, path, value, bonus=offset, evidence=evidence)
        if candidate:
            paths.append(candidate)
    if not paths:
        notes.append('No stricter valid boundary is established by the retained measurements and current YAML. No matching value is guessed.')
    return paths, notes


def _quality(config, title, groups, *, role=None, next_action=None, effect):
    """Offer one independent quality alternative; missing measurements yield none."""
    priority = ('sharpness', 'face_pixels', 'person_pixels', 'visible_face_fraction', 'pose_deviation', 'score_threshold', 'max_overlap')
    rows = [(g, row) for g in groups for row in g['checks']
            if role is None or row['parameter'].startswith('clustering.min_'+role+'_')
            or (role == 'anchor' and row['parameter'] == 'clustering.max_anchor_pose_deviation')]
    rows.sort(key=lambda pair: next((i for i, suffix in enumerate(priority) if pair[1]['parameter'].endswith(suffix)), len(priority)))
    for group, row in rows:
        path, measured = row['parameter'], row['measured']
        if row['operator'] == '==' or not _number(measured):
            continue
        try:
            changes = _tighten(config, path, measured, operator=row['operator'])
        except KeyError:
            continue
        if not changes:
            continue
        notes = [effect, 'Use this alternative only if the measured quality explains the bad sample; it also filters other samples with similar quality.']
        if group['configuration_source'] not in ('current_configuration', 'current_configuration_hash_verified'):
            notes.append('Scan settings differ or cannot be verified. This compares retained measurements with the current YAML; a new scan can change them.')
        kwargs = dict(next_action=next_action) if next_action else {}
        result = _path(config, title, changes, notes=notes,
            evidence=[f'{_name(config,path)}: retained measurement {measured}; acceptance requires {row["operator"]} the configured threshold.'], **kwargs)
        if result['changes']:
            return result
    return None


def suggest(config, kind, trace, admission):
    if trace['status'] != 'accepted':
        return None
    paths, notes = _matching(config, kind, trace)
    groups = {g['stage']:g for g in admission['groups']}
    alternatives = []
    if kind == 'face':
        alternatives.append(_quality(config, 'If quality is the problem: exclude the face from matching',
            [groups['face_auxiliary_quality']], role='auxiliary',
            effect='This removes quality eligibility for both anchor and auxiliary matching globally, not just this performer.'))
        if trace['role'] == 'anchor':
            alternatives.append(_quality(config, 'Reduce the face to auxiliary eligibility',
                [groups['face_anchor_quality']], role='anchor',
                effect='This removes anchor eligibility only. The face may remain assigned as auxiliary; this is not a detachment guarantee.'))
        alternatives.append(_quality(config, 'If quality is the problem: exclude the detection on a fresh scan',
            [groups['face_detection']], next_action=RESCAN,
            effect='This rejects the detection globally before matching, rather than excluding only this performer.'))
    else:
        if trace['role'] == 'anchor':
            alternatives.append(_quality(config, 'Reduce the body to auxiliary eligibility',
                [groups['person_anchor_quality']],
                effect='This removes body anchor eligibility only. Face association or auxiliary matching can still assign the body. Regional all/any keypoint rules are unchanged.'))
        alternatives.append(_quality(config, 'If quality is the problem: exclude the independent body on a fresh scan',
            [groups['independent_person_collection']], next_action=RESCAN,
            effect='This rejects independent body collection globally. It does not remove the linked face or its separate face-associated body evidence.'))
    paths.extend(p for p in alternatives if p)
    return dict(title='If this assignment is incorrect', paths=paths, notes=[
        'These are alternative first steps, not a diagnosis that this sample is wrong. Test one path at a time.',
        'Suggestions use retained evidence, not a verified detachment replay. A blocked route may still attach through another route or to another identity.',
        'YAML settings affect other observations too and can split valid identities. Compare the result after rebuilding.',
        *notes])
