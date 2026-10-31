"""Explain retained admission measurements without rerunning models or matching."""
import json
import math
from types import SimpleNamespace

from .person_anchor import pixels
from .pose_policy import measurement_required
from .visibility import cached


def check(parameter, measured, threshold, operator='>=', *, enabled=True, note=None):
    """A boundary applies to this gate only, not to the whole detection pipeline."""
    numeric = type(measured) in (int, float) and math.isfinite(measured) and measured >= 0
    valid = type(measured) is bool if operator == '==' else numeric
    result = dict(parameter=parameter, measured=measured if valid else None,
                  threshold=threshold, operator=operator, status='unknown',
                  margin=None, parameter_requirement=None)
    if not enabled:
        result['status'] = 'disabled'
    elif valid:
        passed = (measured >= threshold if operator == '>=' else
                  measured > threshold if operator == '>' else
                  measured <= threshold if operator == '<=' else measured == threshold)
        result['status'] = 'pass' if passed else 'fail'
        if operator != '==':
            result['margin'] = measured-threshold if operator in ('>', '>=') else threshold-measured
            result['parameter_requirement'] = dict(
                operator={'<=': '>=', '>=': '<=', '>': '<'}[operator], boundary=measured)
    if note:
        result['note'] = note
    return result


def group(name, checks, *, source='current_configuration', note=None, context=None):
    states = {row['status'] for row in checks}
    status = ('fail' if 'fail' in states else 'unknown' if 'unknown' in states else
              'disabled' if states == {'disabled'} else 'pass')
    return dict(stage=name, configuration_source=source, status=status, checks=checks,
                **({'note': note} if note else {}), **({'context': context} if context else {}))


class Policies:
    """Resolve hashes against saved effective configurations, once per report.

    Matching a hash verifies the settings, not which run created an observation.
    Missing/old snapshots never get silently filled with today's defaults.
    """
    def __init__(self, db, config, scope, namespace):
        self.db, self.config, self.scope, self.namespace = db, config, scope, namespace
        self.saved = None
        self.cache = {}

    def fingerprint(self, config, kind):
        if kind == 'face':
            from .scanning import scan_identity
            return scan_identity(config, self.namespace)
        if kind == 'body':
            from .body_observations import contract
        else:
            from .person_signal import contract
        return contract(config)

    def resolve(self, kind, fingerprint):
        key = kind, fingerprint
        if key in self.cache:
            return self.cache[key]
        if self.fingerprint(self.config, kind) == fingerprint:
            result = self.config, 'current_configuration_hash_verified'
        else:
            if self.saved is None:
                self.saved = []
                for row in self.db.execute('SELECT config FROM runs WHERE scope=? ORDER BY started DESC,id', (self.scope,)):
                    try:
                        data = json.loads(row['config'])['effective']
                        if isinstance(data, dict):
                            self.saved.append(SimpleNamespace(data=data))
                    except (KeyError, TypeError, ValueError):
                        continue
            result = self.config, 'current_configuration_only_historical_unavailable'
            for candidate in self.saved:
                try:
                    if self.fingerprint(candidate, kind) == fingerprint:
                        result = candidate, 'retained_configuration_hash_verified'
                        break
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
        self.cache[key] = result
        return result


def face(quality, geometry, config, policies, fingerprint):
    scan, source = policies.resolve('face', fingerprint)
    d, c = scan.data['detection'], config.data['clustering']
    visible = quality.get('visible_face_fraction') if cached(quality) else None
    score = geometry[14] if isinstance(geometry, list) and len(geometry) == 15 else None
    checks = [
        check('detection.score_threshold', score, d['score_threshold'], '>'),
        check('detection.min_face_pixels', quality.get('pixels'), d['min_face_pixels']),
        check('detection.min_sharpness', quality.get('sharpness'), d['min_sharpness']),
        check('detection.require_complete_face', quality.get('complete'), d['require_complete_face'], '==', enabled=d['require_complete_face']),
        check('detection.max_pose_deviation', quality.get('pose_deviation'), d['max_pose_deviation'], '<=',
              enabled=d['max_pose_deviation'] is not None),
        check('detection.min_visible_face_fraction', visible, d['min_visible_face_fraction'],
              enabled=measurement_required(scan), note='Requires a retained valid visibility measurement when enabled.'),
        check('alignment_valid', quality.get('alignment_valid'), True, '==', note='Mandatory geometry gate; no YAML switch.'),
        check('landmarks_inside', quality.get('landmarks_inside'), True, '==',
              enabled=d['max_pose_deviation'] is not None, note='Required when the detection pose limit is enabled.'),
    ]
    groups = [group('face_detection', checks, source=source,
        context={'detection.'+key: d[key] for key in ('nms_threshold', 'max_faces_per_frame')},
        note='NMS and result limits depend on other detections; their boundary cannot be recovered from one retained crop.')]
    for role in ('anchor', 'auxiliary'):
        checks = []
        for suffix, field in (('face_pixels', 'pixels'), ('sharpness', 'sharpness')):
            keys = ['min_auxiliary_'+suffix] + (['min_anchor_'+suffix] if role == 'anchor' else [])
            for key in keys:
                floor = c[key]
                checks.append(check('clustering.'+key, quality.get(field), floor, enabled=bool(floor)))
        keys = ['min_auxiliary_visible_face_fraction'] + (['min_anchor_visible_face_fraction'] if role == 'anchor' else [])
        for key in keys:
            checks.append(check('clustering.'+key, visible, c[key], enabled=bool(c[key])))
        if role == 'anchor':
            maximum = c['max_anchor_pose_deviation']
            checks.append(check('clustering.max_anchor_pose_deviation', quality.get('pose_deviation'), maximum, '<=',
                                enabled=maximum is not None))
        groups.append(group('face_'+role+'_quality', checks,
            note='Additional current quality gates after scan admission; passing does not establish identity membership.'))
    return dict(evidence_kind='face', groups=groups)


def person(metadata, config, policies, fingerprint, *, independent=False):
    collected, source = policies.resolve('body' if independent else 'person', fingerprint)
    settings = collected.data['sapiensid']
    person = metadata.get('person') or {}
    quality = metadata.get('quality') or {}
    pose = settings['pose']
    checks = [
        check('sapiensid.pose.score_threshold', person.get('score'), pose['score_threshold'], '>'),
        check('sapiensid.pose.keypoint_score_threshold', None, pose['keypoint_score_threshold'], '>',
              note='Individual raw keypoint confidences are not retained; accepted coordinates alone cannot recover them.'),
    ]
    groups = [group('person_detection', checks, source=source,
        context={'sapiensid.pose.'+key: pose[key] for key in ('nms_threshold', 'max_people_per_frame')},
        note='Keypoint confidence filters individual points. NMS/result limits depend on the entire frame.')]
    if independent:
        collection = settings['collection']
        checks = [check('sapiensid.collection.'+key, value, collection[key], op)
            for key, value, op in (
                ('min_person_pixels', pixels(metadata), '>='),
                ('min_sharpness', quality.get('sharpness'), '>='),
                ('min_keypoints', quality.get('keypoints'), '>='),
                ('max_overlap', quality.get('overlap'), '<='))]
        groups.append(group('independent_person_collection', checks, source=source,
            note='An ambiguous face/person association can also reject the observation.'))
    else:
        groups.append(group('face_person_association', [check('unique_face_person_association',
            bool(person) if 'person' in metadata else None, True, '==')], source=source,
            note='Face-associated evidence inherits face admission. Independent collection quality floors are not required here.'))
    identity = config.data['sapiensid']['identities']
    from .person_anchor import anchor_gate
    gate = anchor_gate(metadata, identity)
    floor = identity['anchor_min_person_pixels']
    checks = [check('sapiensid.identities.anchor_min_person_pixels', pixels(metadata), floor, enabled=floor is not None)]
    checks.extend(check('sapiensid.identities.'+key, gate.get(field), identity[key])
                  for key, field in (('anchor_min_keypoints', 'keypoints'), ('anchor_min_sharpness', 'sharpness')))
    groups.append(group('person_anchor_quality', checks,
        note='Current quality gates only. Face authority, complete evidence, support and matching are separate requirements.'))
    regional = gate['keypoint_rules']
    for mode in ('all', 'any'):
        rows = regional[mode]
        if not rows:
            continue
        checks = [dict(check(row['parameter'], row['measured'], row['minimum'], enabled=row['minimum'] > 0),
                       label=row['label'], points=row['points'], present=row['present']) for row in rows]
        summary = group('person_anchor_keypoints_'+mode, checks,
            note=('Every rule must pass.' if mode == 'all' else 'At least one rule must pass; other failed alternatives do not reject the anchor.')+
                 ' Counts use surviving pose points with anatomical left/right labels; min=0 adds no requirement.')
        passed = all(row['passed'] for row in rows) if mode == 'all' else any(row['passed'] for row in rows)
        if passed:
            summary['status'] = 'pass'
        elif mode == 'any':
            summary['status'] = 'unknown' if any(row['measured'] is None for row in rows) else 'fail'
        groups.append(summary)
    return dict(evidence_kind='person', route='independent' if independent else 'face_associated',
                groups=groups, anchor_gate=gate)
