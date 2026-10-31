"""Deterministic permissive experiments independent of rejection feedback."""
from .merge_report_policy import parameter, patched
from .merge_solver_domains import registry, normalize, UNIT, SUPPORT


def admission(config):
    domains = registry(config)
    keys = [k for k in domains if k.startswith('clustering.min_') or k == 'clustering.max_anchor_pose_deviation'
            or k.startswith(('sapiensid.identities.anchor_min_', 'sapiensid.identities.anchor_keypoint_rules.'))]
    values = {k: (max(parameter(config.data, k), domains[k].high) if domains[k].relax > 0
                  else min(parameter(config.data, k), domains[k].low))
              for k in keys if parameter(config.data, k) is not None}
    return normalize(config, values)


def profiles(config):
    """Stay inside enabled routes and retained contracts; never invent features."""
    from .body_parameters import active
    from .models import namespace
    from .scanning import scan_identity
    from .person_signal import same_contract
    domains = registry(config)
    keys = [k for k in UNIT+SUPPORT+active(config) if k in domains
            and '.anchor_min_' not in k and '.anchor_keypoint_rules.' not in k]
    matching = {k: domains[k].low for k in keys}
    matching['clustering.same_scene_match_bonus'] = max(config.data['clustering']['same_scene_match_bonus'],
                                                       domains['clustering.same_scene_match_bonus'].high)
    quality = admission(config)
    combined = dict(matching, **quality)
    from .pose_policy import restricted
    if restricted(config.data['clustering']):
        combined['clustering.anchor_view_matching'] = True
    candidates = [('matching_floor', matching), ('admission_floor', quality), ('all_floors', combined),
        ('face_floors', {k: v for k, v in combined.items() if not k.startswith('sapiensid.')}),
        ('hybrid_floors', {k: v for k, v in combined.items()
                           if k.startswith('sapiensid.matching.') or k in quality}),
        ('body_floors', {k: v for k, v in combined.items()
                         if k.startswith(('sapiensid.identities.', 'sapiensid.global_matching.'))})]
    ns, seen = namespace(config), set()
    for name, changes in candidates:
        changes = normalize(config, changes)
        signature = tuple(sorted(changes.items()))
        if not changes or signature in seen:
            continue
        variant = patched(config, changes)
        if scan_identity(variant, ns) != scan_identity(config, ns) or not same_contract(config, variant):
            continue
        seen.add(signature)
        yield name, changes


def relaxed_quality(config, feedback):
    """Only admission hints can supplement a measured liberal seed.

    Previously the first face proposal could overwrite similarity thresholds
    with an exclusion's rejection boundary, turning a liberal seed restrictive.
    """
    allowed = admission(config)
    values = {}
    for item in feedback.get('quality', ()):
        for key, value in item.get('patch', {}).items():
            if key not in allowed:
                continue
            current = parameter(config.data, key)
            choose = max if key.endswith('pose_deviation') else min
            values[key] = choose(values.get(key, current), value)
    return values
