"""Retained-evidence parameter domains shared by all proposal lanes."""
from dataclasses import dataclass
import math

from .merge_report_policy import parameter


@dataclass(frozen=True)
class Domain:
    kind: type = float
    low: float = 0.
    high: float = 1.
    relax: int = -1

    def project(self, value):
        if self.kind is bool:
            if type(value) is not bool:
                raise ValueError('Boolean solver parameters require boolean values.')
            return value
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Solver parameters must be finite numbers.')
        value = min(self.high, max(self.low, value))
        from .threshold_boundaries import at_or_below, at_or_above
        rounded = at_or_below(value) if self.relax < 0 else at_or_above(value)
        return int(round(value)) if self.kind is int else min(self.high, max(self.low, rounded))


UNIT = ('recognition.match_threshold', 'recognition.ambiguity_margin',
        'recognition.cooccurrence_merge_threshold', 'clustering.cohesion_threshold',
        'clustering.view_match_threshold', 'clustering.view_cohesion_threshold',
        'clustering.view_ambiguity_margin', 'sapiensid.matching.face_floor',
        'sapiensid.matching.person_threshold', 'sapiensid.matching.ambiguity_margin')
SUPPORT = ('recognition.min_identity_support', 'recognition.min_merge_support',
           'clustering.same_scene_min_support', 'clustering.auxiliary_min_support',
           'clustering.view_min_support', 'clustering.view_attach_min_support', 'sapiensid.matching.min_support')


def registry(config):
    result = {k: Domain() for k in UNIT}
    result.update({k: Domain(int, 2 if k == 'clustering.view_min_support' else 1, 1000) for k in SUPPORT})
    result['sapiensid.matching.min_support'] = Domain(int, 1, max(1000, config.data['sapiensid']['matching']['min_support']*2))
    result['clustering.same_scene_match_bonus'] = Domain(high=.1, relax=1)
    result['clustering.anchor_view_matching'] = Domain(bool, relax=1)
    detection = config.data['detection']
    pose = detection['max_pose_deviation']
    result['clustering.max_anchor_pose_deviation'] = Domain(high=1. if pose is None else pose, relax=1)
    for role in ('anchor', 'auxiliary'):
        for suffix, source, high in (('face_pixels', 'min_face_pixels', 4096),
                                     ('sharpness', 'min_sharpness', 100000),
                                     ('visible_face_fraction', 'min_visible_face_fraction', 1.)):
            result[f'clustering.min_{role}_{suffix}'] = Domain(low=detection[source], high=high)
    if not config.data['sapiensid']['enabled']:
        result = {k: v for k, v in result.items() if not k.startswith('sapiensid.')}
    from .body_identity import enabled
    if enabled(config):
        for section in ('identities','global_matching'):
            for key in ('match_threshold','cohesion_threshold','ambiguity_margin'):
                result['sapiensid.'+section+'.'+key]=Domain()
            result['sapiensid.'+section+'.min_support']=Domain(int,2 if section=='identities' else 1,
                1000 if section=='identities' else config.data['sapiensid']['global_matching']['max_references'])
        result['sapiensid.identities.attach_min_support']=Domain(int,1,1000)
    return result


def normalize(config, changes):
    domains = registry(config)
    if set(changes)-domains.keys():
        raise ValueError('Solver proposals may only change registered retained-evidence parameters.')
    patch = {k: (v if v == parameter(config.data, k) else domains[k].project(v)) for k, v in changes.items()}
    def value(key):
        return patch.get(key, parameter(config.data, key))
    def ordered(lower, upper):
        a, b = value(lower), value(upper)
        if a is not None and b is not None and a > b:
            if lower in patch and upper not in patch:
                patch[upper] = a
            else:
                patch[lower] = b
    ordered('clustering.cohesion_threshold', 'recognition.match_threshold')
    from .body_identity import enabled
    if enabled(config):
        for section in ('identities','global_matching'):
            ordered('sapiensid.'+section+'.cohesion_threshold','sapiensid.'+section+'.match_threshold')
    for suffix in ('face_pixels', 'sharpness', 'visible_face_fraction'):
        ordered('clustering.min_auxiliary_'+suffix, 'clustering.min_anchor_'+suffix)
    key = 'clustering.anchor_view_matching'
    if key in patch:
        from .pose_policy import restricted
        grouping = dict(config.data['clustering'], **{k.split('.')[1]: v for k, v in patch.items() if k.startswith('clustering.')})
        if patch[key] and not restricted(grouping):
            patch[key] = False
        elif not patch[key] and any(grouping[k] for k in (
                'anchor_outlier_quarantine', 'strict_appearance_matching', 'context_view_enrichment')):
            patch[key] = True
    return {k: v for k, v in patch.items() if v != parameter(config.data, k)}


def dimensions(config, options, active, sequence, width=8):
    domains = registry(config)
    keys = sorted((k for k in options if k in domains and parameter(config.data, k) is not None),
                  key=lambda k: (-active.get(k, 0), not k.startswith('sapiensid.matching.'), k))
    if len(keys) <= width:
        return keys
    fixed = width//2
    head, tail = keys[:fixed], keys[fixed:]
    offset = ((sequence//5)*(width-fixed)) % len(tail)
    return head+(tail[offset:]+tail[:offset])[:width-fixed]
