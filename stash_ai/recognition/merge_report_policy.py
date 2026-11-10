"""Measured parameter proposals, verified by the production frozen planner."""
from dataclasses import replace

from .config import plain, validate
from .config_layout import internal_path, public_path, v2_patch
from .global_views import cohort_evidence, consolidation_plan
from .pose_policy import restricted


def _floor(value):
    # A rounded-up threshold could reject the very sample used to propose it.
    from .threshold_boundaries import at_or_below
    return at_or_below(min(1., value))


def parameter(data, path):
    if path == 'sapiensid.identities.scene_match_bonus':
        from .body_identity import scene_bonus
        return scene_bonus(data)
    if path == 'sapiensid.identities.track_similarity_threshold':
        settings = data['sapiensid']['identities']
        threshold = settings.get('track_similarity_threshold')
        return settings['cohesion_threshold'] if threshold is None else threshold
    for key in path.split('.'):
        data = data[int(key)] if isinstance(data, (list, tuple)) else data[key]
    return data


def pins_inherited_value(data, path, value, changes):
    """An explicit equal value can decouple a child from a changing parent."""
    parents = {'sapiensid.identities.scene_match_bonus':'clustering.same_scene_match_bonus',
               'sapiensid.identities.track_similarity_threshold':'sapiensid.identities.cohesion_threshold'}
    parent = parents.get(path)
    return (parent in changes and data['sapiensid']['identities'].get(path.rsplit('.',1)[-1]) is None
            and value is not None and value != changes[parent])


def assign(data, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        data = data[int(key)] if isinstance(data, list) else data.setdefault(key, {})
    data[int(keys[-1]) if isinstance(data, list) else keys[-1]] = value


def patched(config, changes):
    data = plain(config.data)
    for path, value in changes.items():
        path = internal_path(path)
        parameter(data, path)  # Refuse unknown paths before assigning.
        assign(data, path, value)
    validate(data)
    return replace(config, data=data)


def proposal(config, changes):
    import yaml
    from .pose_anchor_rules import PARAMETER as regional
    patch, details = {}, []
    changes = {internal_path(path):value for path,value in changes.items()}
    for path, value in sorted(changes.items()):
        current = plain(parameter(config.data, path))
        pin = pins_inherited_value(config.data,path,value,changes)
        if current == value and not pin:
            continue
        if path.startswith(regional+'.') and not any(row['parameter'].startswith(regional+'.') for row in details):
            # A YAML list is replaced as a whole. Preserve every selector and
            # untouched sibling when emitting a patch for an indexed minimum.
            assign(patch, regional, plain(parameter(config.data, regional)))
        assign(patch, path, value)
        details.append(dict(parameter=path, current=current, proposed=value,
                            direction='replace' if pin or path == regional or path.endswith(('tracklet_matching','endpoint_matching','scene_bonus_mode')) else
                            'disable' if value is None else 'enable' if type(value) is bool or current is None else
                            'decrease' if value < current else 'increase'))
    if config.document_version == 2:
        for row in details:
            row['yaml_parameter'] = public_path(row['parameter'])
        patch = v2_patch(patch)
    return dict(changes=details, yaml_patch=patch,
                yaml=yaml.safe_dump(patch, sort_keys=False) if patch else '')


def _planner_config(config, route):
    if route != 'strict_core':
        return config
    # Exactly the overrides used by strict_partition.resolve.
    data = plain(config.data)
    data['recognition']['min_merge_support'] = 1
    data['clustering'].update(view_match_threshold=data['recognition']['match_threshold'],
        view_cohesion_threshold=data['clustering']['cohesion_threshold'],
        view_ambiguity_margin=data['recognition']['ambiguity_margin'])
    return replace(config, data=data)


def check(db, views, config, identities, route):
    owned = {person: [v for v in views if v.group.identifier == person] for person in identities}
    effective = _planner_config(config, route)
    evidence = [cohort_evidence(db, a, b, effective)
                for a in owned[identities[0]] for b in owned[identities[1]]]
    evidence.sort(key=lambda row: (len(row['blockers']), -row['match_score'], row['medoids']))
    plans = consolidation_plan(db, views, effective, focus=identities)
    shared = next((p for p in plans if set(identities).issubset(p['people'])), None)
    blockers = []
    if not config.data['clustering']['anchor_view_matching']:
        blockers.append('anchor_view_matching_disabled')
    if not restricted(config.data['clustering']):
        blockers.append('anchor_restriction_required')
    for person in identities:
        if not owned[person]:
            blockers.append('no_eligible_references:' + person)
    if shared is None:
        blockers.append('no_shared_strong_component')
    elif shared['blocker']:
        blockers.append(shared['blocker'])
    return dict(can_merge_in_frozen_catalogue=not blockers, blockers=blockers,
                pair_evidence=evidence[:8], pair_evidence_count=len(evidence),
                components=list(plans), shared_component=shared,
                effective_thresholds=dict(
                    match=effective.data['recognition']['match_threshold'] if route == 'strict_core' else
                          max(effective.data['recognition']['match_threshold'], effective.data['clustering']['view_match_threshold']),
                    cohesion=effective.data['clustering']['cohesion_threshold' if route == 'strict_core' else 'view_cohesion_threshold'],
                    ambiguity_margin=max(effective.data['recognition']['ambiguity_margin'], effective.data['clustering']['view_ambiguity_margin'])))


def _thresholds(config, proof, route, isolate):
    cap = min(proof['match_score'], proof['mean_cross_similarity'])
    if proof['cooccurring']:
        cap = min(cap, proof['minimum_cross_similarity'])
    low = proof['minimum_cross_similarity']
    if cap < 0 or low < 0:
        return None  # YAML thresholds are nonnegative; never clamp a failing score to zero.
    settings, grouping = config.data['recognition'], config.data['clustering']
    match = _floor(cap)
    cohesion = _floor(min(low, match))
    changes = {'recognition.match_threshold': match if isolate else min(settings['match_threshold'], match)}
    # The YAML constraint also applies when the view route reduces the global match threshold.
    changes['clustering.cohesion_threshold'] = min(grouping['cohesion_threshold'], changes['recognition.match_threshold'])
    if route == 'strict_core':
        changes['clustering.cohesion_threshold'] = cohesion if isolate else min(changes['clustering.cohesion_threshold'], cohesion)
    else:
        changes.update({
            'clustering.view_match_threshold': match if isolate else min(grouping['view_match_threshold'], match),
            'clustering.view_cohesion_threshold': cohesion if isolate else min(grouping['view_cohesion_threshold'], cohesion),
            'recognition.min_merge_support': min(settings['min_merge_support'], min(proof['support'])),
        })
    if proof['cooccurring']:
        changes['recognition.cooccurrence_merge_threshold'] = min(settings['cooccurrence_merge_threshold'], _floor(low))
    if restricted(grouping):
        changes['clustering.anchor_view_matching'] = True
    return changes


def analyse(db, config, identities, route, make_views):
    """Rebuild changed view catalogues; never reuse cohorts after changing their gates."""
    current = check(db, make_views(config), config, identities, route)
    result = dict(route=route, current=current, attempts=[], recommendations=[])
    if current['can_merge_in_frozen_catalogue']:
        result['recommendations'].append(dict(proposal(config, {}), verification=current))
        return result
    base_changes = {}
    available = current
    if route == 'confirmed_views' and not current['pair_evidence']:
        # A view needs at least two independent cues, even if merge support is one.
        base_changes['clustering.view_min_support'] = 2
        available = check(db, make_views(patched(config, base_changes)),
                          patched(config, base_changes), identities, route)
    # Each supported appearance pair can require a different joint patch. Keep
    # attempts bounded; final recommendations always recheck the whole component.
    proofs = available['pair_evidence'][:4]
    result['unresolved_reasons'] = ([] if proofs else ['no_supported_reference_pair'])
    if proofs and all(min(p['match_score'], p['mean_cross_similarity'], p['minimum_cross_similarity']) < 0 for p in proofs):
        result['unresolved_reasons'].append('pair_requires_a_negative_threshold_outside_the_yaml_range')
    for proof in proofs:
        for isolate in (False, True):
            changes = _thresholds(config, proof, route, isolate)
            if changes is None:
                continue
            changes.update(base_changes)
            candidate = patched(config, changes)
            verification = check(db, make_views(candidate), candidate, identities, route)
            shared = verification['shared_component']
            if shared and shared['blocker'] == 'competing_view':
                gap = shared['nearest_rival']['gap']
                if gap >= 0:
                    margin = _floor(gap)
                    changes['recognition.ambiguity_margin'] = min(config.data['recognition']['ambiguity_margin'], margin)
                    if route != 'strict_core':
                        changes['clustering.view_ambiguity_margin'] = min(config.data['clustering']['view_ambiguity_margin'], margin)
                    candidate = patched(config, changes)
                    verification = check(db, make_views(candidate), candidate, identities, route)
            item = dict(proposal(config, changes), strategy='isolate_pair' if isolate else 'relax_blocking_gates',
                        verification=verification)
            if any(a['yaml_patch'] == item['yaml_patch'] for a in result['attempts']):
                continue
            result['attempts'].append(item)
            if verification['can_merge_in_frozen_catalogue']:
                result['recommendations'].append(item)
    result['recommendations'].sort(key=lambda item: (
        item['verification']['shared_component']['component_size'], len(item['changes']), item['yaml']))
    return result
