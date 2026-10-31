"""Regional COCO17 admission, independent of pose backend and model inference."""
from collections.abc import Mapping

PARAMETER = 'sapiensid.identities.anchor_keypoint_rules'
POINTS = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
GROUPS = {
    'eyes': (1, 2), 'head_without_eyes': (0, 3, 4), 'head': tuple(range(5)),
    'shoulders': (5, 6), 'hips': (11, 12), 'torso': (5, 6, 11, 12),
    'left_torso': (5, 11), 'right_torso': (6, 12),
    'left_arm': (5, 7, 9), 'right_arm': (6, 8, 10), 'arms': tuple(range(5, 11)),
    'left_leg': (11, 13, 15), 'right_leg': (12, 14, 16), 'legs': tuple(range(11, 17)),
    'body': tuple(range(5, 17)),
}


def indices(rule):
    return GROUPS[rule['group']] if 'group' in rule else tuple(POINTS.index(p) for p in rule['points'])


def validate(rules):
    if not isinstance(rules, Mapping) or set(rules)-{'all', 'any'}:
        raise ValueError(PARAMETER+' must be a mapping containing only all/any lists.')
    for mode in ('all', 'any'):
        rows = rules.get(mode, ())
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f'{PARAMETER}.{mode} must be a list.')
        for index, rule in enumerate(rows):
            path = f'{PARAMETER}.{mode}.{index}'
            if not isinstance(rule, Mapping) or set(rule) not in ({'group', 'min'}, {'points', 'min'}):
                raise ValueError(path+' requires min and exactly one of group or points.')
            if 'group' in rule:
                if not isinstance(rule['group'], str) or rule['group'] not in GROUPS:
                    raise ValueError(path+'.group must be one of: '+', '.join(GROUPS))
            else:
                points = rule['points']
                if (not isinstance(points, (list, tuple)) or not points
                        or any(not isinstance(p, str) or p not in POINTS for p in points)
                        or len(set(points)) != len(points)):
                    raise ValueError(path+'.points requires unique COCO17 names: '+', '.join(POINTS))
            maximum = len(indices(rule))
            if type(rule['min']) is not int or not 0 <= rule['min'] <= maximum:
                raise ValueError(f'{path}.min must be an integer in [0,{maximum}].')


def parameters(settings):
    """Only numerical minima are tunable; preserve selections and boolean logic."""
    rules = settings.get('anchor_keypoint_rules', {})
    return {f'{PARAMETER}.{mode}.{index}.min': len(indices(rule))
            for mode in ('all', 'any') for index, rule in enumerate(rules.get(mode, ()))}


def evaluate(rules, points):
    """Points have anatomical labels after the encoder's exclusion filters.

    None means missing/invalid retained geometry, never zero observed points.
    A minimum of zero adds no requirement, including for legacy measurements.
    """
    result = {}
    for mode in ('all', 'any'):
        rows = []
        for index, rule in enumerate(rules.get(mode, ())):
            selected = indices(rule)
            present = ([POINTS[i] for i in selected if min(points[i]) >= 0]
                       if points is not None else None)
            measured = len(present) if present is not None else None
            minimum = rule['min']
            passed = minimum == 0 or measured is not None and measured >= minimum
            rows.append(dict(parameter=f'{PARAMETER}.{mode}.{index}.min',
                label=rule.get('group') or '+'.join(rule['points']),
                points=[POINTS[i] for i in selected], present=present,
                measured=measured, minimum=minimum, passed=passed))
        result[mode] = rows
    eligible = (all(row['passed'] for row in result['all']) and
                (not result['any'] or any(row['passed'] for row in result['any'])))
    failed = [row for row in result['all'] if not row['passed']]
    if result['any'] and not any(row['passed'] for row in result['any']):
        failed.extend(result['any'])
    missing = any(row['measured'] is None for row in failed)
    return dict(result, enabled=bool(result['all'] or result['any']), eligible=eligible,
        failed_parameters=[row['parameter'] for row in failed],
        reason=None if eligible else 'body_anchor_keypoint_rules_missing' if missing else 'body_anchor_keypoint_rules')
