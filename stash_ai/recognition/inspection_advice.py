"""Bounded first-step YAML advice from retained evidence, never a matching replay."""
import math

from .admission_report import check
from .config_layout import public_path
from .merge_report_policy import parameter, patched, proposal
from .threshold_boundaries import at_or_above, at_or_below


RESCAN = 'Run recognition again for this scene; detection/collection changes require fresh scan evidence. Then refresh the inspector.'
REBUILD = 'Run recognition again with the relevant scenes selected to rebuild assignments, then refresh the inspector.'


def _name(config, path):
    try:
        return public_path(path) if config.document_version == 2 else path
    except KeyError:
        return path


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _path(config, title, changes=None, *, evidence=(), notes=(), next_action=REBUILD):
    changes = changes or {}
    notes = list(dict.fromkeys(notes))
    try:
        patched(config, changes)
    except ValueError:
        # A measured boundary may be outside the schema or conflict with another gate.
        notes.append('These boundaries do not form a valid YAML patch. Review dependent gates; no patch is suggested.')
        changes = {}
    return dict(title=title, **proposal(config, changes), evidence=list(evidence), notes=notes,
                next_action=next_action, verification='retained_evidence_only')


def _quality(config, title, groups, *, next_action=REBUILD):
    changes, evidence, notes = {}, [], []
    failures = 0
    for group in groups:
        source = group['configuration_source']
        if source not in ('current_configuration', 'current_configuration_hash_verified'):
            notes.append('Scan settings differ or cannot be verified. Boundaries use retained measurements against the current YAML; a new scan can change them.')
        regional = group['stage'].startswith('person_anchor_keypoints_')
        if regional:
            if group['status'] != 'pass':
                failures += 1
                notes.append('Body anchor keypoint rules do not pass. Keep the configured all/any rules; inspect pose geometry or use the auxiliary route.')
            continue  # A failed ANY alternative is not a failed gate if another alternative passed.
        for row in group['checks']:
            path, measured, operator = row['parameter'], row['measured'], row['operator']
            try:
                current = parameter(config.data, path)
            except KeyError:
                if path == 'landmarks_inside' and config.data['detection']['max_pose_deviation'] is None:
                    continue
                if row['status'] in ('fail', 'unknown'):
                    failures += row['status'] == 'fail'
                    notes.append(f'{path}: {row["status"]}; no YAML threshold can repair missing or invalid geometry.')
                continue
            enabled = current is not None and not (operator in ('>=', '==') and not current)
            result = check(path, measured, current, operator, enabled=enabled)
            if result['status'] == 'unknown':
                notes.append(f'{_name(config,path)}: measurement unavailable; no numeric value suggested.')
                continue
            if result['status'] != 'fail':
                continue
            failures += 1
            evidence.append(f'{_name(config,path)}: retained measurement {measured}, current gate {operator} {current}.')
            if operator == '==':
                value = False  # Only require_complete_face is a configurable Boolean admission gate.
            elif operator == '<=':
                value = at_or_above(measured)
            else:
                value = at_or_below(measured)
                if operator == '>' and value >= measured:
                    value = round(value - .000001, 6)
            if type(current) is int:
                value = math.ceil(value) if operator == '<=' else math.floor(value)
            changes[path] = value
    result = _path(config, title, changes, evidence=evidence, notes=notes, next_action=next_action)
    if result['changes']:
        result['notes'].append('The patch only clears the listed measured quality gates. Matching, independent support and competing identities still apply.')
    return result, failures


def _matching(config, kind, decisions):
    paths, notes = [], []
    current = [d for d in decisions if d['current'] and d['status'] != 'accepted']
    if not current:
        notes.append('No current-policy matching attempt is retained. Rebuild with the current YAML before choosing matching thresholds.')
        return paths, notes
    # Multiple retained policies are never combined into one hypothetical decision.
    decision = current[0]
    proof = decision['recorded']
    reason = proof.get('reason', 'unknown')
    if (any(word in reason for word in ('conflict', 'co_present', 'co_presence', 'copresence', 'different_face', 'reassignment'))
            or reason in ('unresolved_partner', 'unconfirmed_contact_face')):
        notes.append(f'Recorded blocker: {reason}. Inspect the competing identities and geometry; do not bypass this by lowering thresholds.')
        return paths, notes
    if kind == 'face' and reason in ('insufficient_match_or_cohesion', 'auxiliary_unmatched'):
        scores = proof.get('scores') or ()
        settings = config.data['recognition']
        # These two production paths retain scores only after the cohesion filter.
        # An empty ranking cannot supply a similarity or cohesion boundary.
        if scores and _number(scores[0]) and config.data['clustering']['cohesion_threshold'] <= scores[0] < settings['match_threshold']:
            if len(scores)>1 and _number(scores[1]) and scores[0]-scores[1] < settings['ambiguity_margin']:
                notes.append('The retained face candidate also lacks the required rival margin. Compare both identities before lowering the match threshold.')
            else:
                boundary = max(config.data['clustering']['cohesion_threshold'], at_or_below(scores[0]))
                paths.append(_path(config, 'Test the retained face similarity boundary', {'recognition.match_threshold':boundary},
                    evidence=[f'Retained best candidate score: {scores[0]}; current match threshold: {settings["match_threshold"]}.'],
                    notes=['The retained ranking passed the cohesion filter. Rebuild to check current references, rivals, co-presence and independent support.']))
        else:
            notes.append('No usable below-threshold face candidate score is retained. Inspect matching/cohesion evidence; no numeric value is guessed.')
    elif kind == 'body' and reason == 'body_scene_core':
        support = proof.get('independent_support')
        path = 'sapiensid.identities.min_support'
        minimum = parameter(config.data, path)
        if type(support) is int and 1 <= support <= 8 and support < minimum:
            paths.append(_path(config, 'Accept the measured body core with less support', {path:support},
                evidence=[f'Independent support: {support}; required by current YAML: {minimum}.'],
                notes=['Less independent support increases the chance of creating a false identity. The full rebuild is still required.']))
            paths.append(_path(config, 'Keep support requirements and collect stronger evidence',
                notes=['Include more distinct cues/scenes of this person. Repeated crops do not necessarily provide independent votes.']))
        else:
            notes.append('Body-core support is missing, already sufficient, or outside the supported proposal range; no support value is guessed.')
    elif kind == 'body' and reason == 'ambiguous_or_insufficient_face_body_bank':
        score, minimum = proof.get('score'), proof.get('minimum')
        settings = config.data['sapiensid']['identities']
        if _number(score) and _number(minimum):
            runner = proof.get('runner')
            changes = {}
            for key, value in (('match_threshold',score), ('cohesion_threshold',minimum)):
                if value < settings[key]:
                    changes['sapiensid.identities.'+key] = at_or_below(value)
            notes_for_path = ['This frozen-bank attempt does not prove final ownership; face guards and competing claims are re-evaluated on rebuild.']
            if _number(runner) and score-runner < settings['ambiguity_margin']:
                notes_for_path.append('The rival margin also fails. Review both identities; this patch does not resolve that ambiguity.')
            if changes:
                paths.append(_path(config, 'Test the retained body-bank similarity boundary', changes,
                    evidence=[f'Retained mean: {score}; weakest witness: {minimum}.'], notes=notes_for_path))
            else:
                notes.extend(notes_for_path)
        else:
            notes.append('The body-bank comparison is incomplete; no numeric matching threshold is suggested.')
    elif reason == 'auxiliary_disabled':
        paths.append(_path(config, 'Enable attachment to an existing face identity', {'clustering.auxiliary_matching':True},
            notes=['Requires an established identity and sufficient matching/support evidence. Does not promote the sample to an anchor.']))
    else:
        face = kind == 'face'
        prefix = 'recognition' if face else 'sapiensid.identities'
        if 'ambiguous' in reason or 'competing' in reason:
            notes.append(f'Recorded blocker: {reason}. Compare the best candidate and its rival before changing {_name(config,prefix+".ambiguity_margin")}. No margin reduction is suggested.')
        elif 'support' in reason or reason in ('candidate_group', 'anonymous_body_track'):
            path = ('clustering.auxiliary_min_support' if reason.startswith('auxiliary') else
                    'recognition.min_identity_support' if face else prefix+'.min_support')
            notes.append(f'Review {_name(config,path)} and add independent evidence. The retained decision does not establish the complete support count needed to propose a value.')
        else:
            notes.append(f'Recorded blocker: {reason}. Inspect references and {_name(config,prefix+".match_threshold")}; the retained attempt does not establish a complete numeric patch.')
    return paths, notes


def suggest(config, kind, admission, decisions, rejection):
    notes = ['Suggestions use retained evidence, not a verified assignment replay. YAML settings affect other observations too.']
    if kind == 'body':
        switches = ('sapiensid.enabled', 'sapiensid.collection.enabled', 'sapiensid.identities.enabled')
        changes = {path:True for path in switches if not parameter(config.data,path)}
        if changes:
            return dict(paths=[_path(config, 'Enable the body recognition route', changes,
                notes=['The current YAML disables part of this route. Quality gates and identity matching must still pass after enabling it.'],
                next_action=RESCAN if any(p in changes for p in switches[:2]) else REBUILD)], notes=notes)
    groups = {g['stage']:g for g in admission['groups']}
    stages = ('face_detection',) if kind == 'face' else ('person_detection','independent_person_collection')
    scan, failures = _quality(config, 'Admit the observation before matching',
                             [groups[s] for s in stages if s in groups], next_action=RESCAN)
    if rejection or failures:
        if rejection:
            scan['notes'].insert(0, f'Retained rejection: {rejection}.')
        if not scan['changes']:
            scan['notes'].append('No supported numeric edit is indicated by the current measured gates. Reprocess this scene and check missing measurements or geometry before relaxing further.')
        return dict(paths=[scan, _path(config, 'Keep quality gates and inspect a better cue',
                    notes=['A clearer or less occluded observation can provide stronger identity evidence without relaxing global quality floors.'])], notes=notes)
    paths = []
    notes.extend(scan['notes'])
    if kind == 'face':
        anchor, failures = _quality(config, 'Qualify as a face anchor', [groups['face_anchor_quality']])
        if failures:
            auxiliary, _ = _quality(config, 'Attach as an auxiliary face instead', [groups['face_auxiliary_quality']])
            if not config.data['clustering']['auxiliary_matching']:
                changes = {c['parameter']:c['proposed'] for c in auxiliary['changes']}
                changes['clustering.auxiliary_matching'] = True
                auxiliary = _path(config, auxiliary['title'], changes, evidence=auxiliary['evidence'], notes=auxiliary['notes'])
            auxiliary['notes'].append('Requires an existing identity and matching/support evidence. Anchor quality restrictions remain in place.')
            paths.extend((auxiliary, anchor))
        else:
            notes.extend(anchor['notes'])
    else:
        anchor, failures = _quality(config, 'Qualify as a body anchor',
            [g for name,g in groups.items() if name.startswith('person_anchor_')])
        gate = admission.get('anchor_gate', {})
        if gate.get('reason') == 'body_query_only':
            message = 'This evidence is query-only; quality changes alone cannot promote it to an anchor.'
            anchor['notes'].append(message)
            notes.append(message)
        elif gate.get('reason') and not failures:
            notes.append(f'Body anchor blocker: {gate["reason"]}. Inspect the retained geometry and association before changing anchor thresholds.')
        if failures:
            paths.extend((_path(config, 'Attach to an existing identity as auxiliary',
                notes=['Inspect linked faces and accepted body references. Anchor quality need not be relaxed for an attachment; matching and face/identity guards still apply.']), anchor))
    matching, matching_notes = _matching(config, kind, decisions)
    paths.extend(matching)
    notes.extend(matching_notes)
    if not paths:
        paths.append(_path(config, 'Recheck matching and independent support', notes=matching_notes))
    return dict(paths=paths[:4], notes=list(dict.fromkeys(notes)))


def describe(advice):
    if not advice:
        return ''
    lines = ['Suggested next steps'] + advice['notes']
    for index, path in enumerate(advice['paths'], 1):
        lines.extend(('', f'{index}. {path["title"]}'))
        for change in path['changes']:
            lines.append(f'{change.get("yaml_parameter",change["parameter"])}: {change["current"]} -> {change["proposed"]}')
        lines.extend(path['evidence'])
        lines.extend(path['notes'])
        if path['yaml']:
            lines.extend(('YAML patch to test:', path['yaml'].rstrip()))
        lines.append('Next: '+path['next_action'])
    return '\n'.join(lines)+'\n\nRecorded evidence\n'
