"""Read-only facial evidence for regional body rules, never for pose ownership."""
from collections.abc import Mapping
import json
import math

PARAMETER = 'sapiensid.identities.anchor_face_landmark_fallback'
LANDMARKS = {'nose': (0, 2), 'left_eye': (1, 1), 'right_eye': (2, 0)}


def defaults():
    return dict(enabled=False, points=list(LANDMARKS), min_face_score=.9,
                max_eye_pose_deviation=.25)


def validate(settings):
    from .config import _number
    if not isinstance(settings, Mapping) or set(settings) != set(defaults()):
        raise ValueError(PARAMETER+' requires enabled, points, min_face_score and max_eye_pose_deviation.')
    if type(settings['enabled']) is not bool:
        raise ValueError(PARAMETER+'.enabled must be boolean.')
    points = settings['points']
    if (not isinstance(points, (list, tuple)) or not points
            or any(not isinstance(p, str) or p not in LANDMARKS for p in points)
            or len(set(points)) != len(points)):
        raise ValueError(PARAMETER+'.points requires unique names: '+', '.join(LANDMARKS))
    _number(settings['min_face_score'], PARAMETER+'.min_face_score', 0, 1)
    _number(settings['max_eye_pose_deviation'], PARAMETER+'.max_eye_pose_deviation', 0, 1)


def face_record(row):
    """Face observation already paired with this person by the collector."""
    row = dict(row)
    return dict(observation=row.get('id'), geometry=row.get('geometry'),
                quality=row.get('quality'), rejected=row.get('rejected'))


def needed(settings):
    rules = settings.get('anchor_keypoint_rules', {})
    return settings.get('anchor_face_landmark_fallback', {}).get('enabled', False) and any(rules.values())


def body_face_sql(alias='b', *, settings=None, respect_corrections=True):
    """Read retained geometry without per-row Python queries or loading pixels."""
    if settings is not None and not needed(settings):
        return 'NULL AS anchor_face'
    correction = (f" AND NOT EXISTS (SELECT 1 FROM body_members bm WHERE bm.observation={alias}.id "
                  "AND json_extract(bm.proof,'$.contact_correction') IS NOT NULL)"
                  if respect_corrections else '')
    return ("(SELECT json_object('observation',af.id,'geometry',json(af.geometry),"
            "'quality',json(af.quality),'rejected',af.rejected) FROM observations af "
            f'WHERE af.id={alias}.face AND af.scan={alias}.scan AND af.cue={alias}.cue'
            + correction + ') AS anchor_face')


def retained_body_face(db, identifier, settings):
    if not needed(settings):
        return None
    row = db.execute('SELECT '+body_face_sql()+' FROM body_observations b WHERE b.id=?',
                     (identifier,)).fetchone()
    return row[0] if row else None


def _decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def supplement(points, metadata, settings, evidence):
    """Return separate rule coordinates and provenance; raw points stay untouched.

    The retained face link and original head points must establish ownership
    before borrowing anything. An inferred landmark is not a visibility score.
    """
    result = dict(enabled=True, recovered=[], skipped={}, reason=None)
    wanted = [name for name in settings['points'] if min(points[LANDMARKS[name][0]]) < 0]
    if not wanted:
        result['reason'] = 'no_missing_selected_points'
        return points, result
    try:
        face_data = _decoded(evidence)
        if not isinstance(face_data, Mapping) or face_data.get('rejected') is not None:
            result['reason'] = 'linked_face_unavailable'
            return points, result
        from .models import checked_detection
        face = checked_detection(_decoded(face_data['geometry']))
        quality = _decoded(face_data.get('quality')) or {}
        person = metadata['person']
        if face.score < settings['min_face_score'] or quality.get('alignment_valid') is False:
            result['reason'] = 'linked_face_quality'
            return points, result
        from .person_association import association_rejection
        rejection = association_rejection(face, person)
        if rejection is not None:
            result['reason'] = rejection
            return points, result
        from .quality import pose_measurements
        pose = pose_measurements(face.landmarks, face.box[2:])
        eye_span = math.dist(*face.landmarks[:2])
        eye_allowed = (not quality.get('face_profile') and eye_span >= max(2., .16*min(face.box[2:]))
                       and pose['pose_deviation'] <= settings['max_eye_pose_deviation'])
        result.update(face_observation=face_data.get('observation'), face_score=face.score,
                      eye_pose_deviation=pose['pose_deviation'])
        left, top, right, bottom = person['bounds']
        fx, fy, fw, fh = face.box
        copied = list(points)
        refinement = person.get('pose_refinement', {})
        contested = refinement.get('point_exclusions', {}).get('near_peer', 0)
        for index in set(refinement.get('joint_ownership_withheld', ())) | set(
                refinement.get('withheld_rival_joints', ())) | set(refinement.get('mask_ownership_withheld', ())):
            contested |= 1 << index
        visibility = person.get('keypoint_visibility')
        states = person.get('keypoint_states', ())
        for name in wanted:
            index, landmark = LANDMARKS[name]
            xy = face.landmarks[landmark]
            reason = None
            if not (max(0, left, fx) <= xy[0] < min(right, fx+fw)
                    and max(0, top, fy) <= xy[1] < min(bottom, fy+fh)):
                reason = 'outside_crop'
            elif contested & (1 << index) or refinement.get('mask_conflict') or (
                    len(states) > index and states[index] == 'conflicted'):
                reason = 'point_ownership_conflict'
            elif (len(states) > index and states[index] == 'occluded') or (
                    person.get('keypoint_visibility_measured') and visibility is not None and (
                    not math.isfinite(visibility[index]) or visibility[index] <= .5)):
                reason = 'explicit_pose_occlusion'
            elif name != 'nose' and not eye_allowed:
                reason = 'eye_visibility_uncertain'
            elif any(min(p) >= 0 and math.dist(p, xy) < 1 for j, p in enumerate(copied[:3]) if j != index):
                reason = 'coincident_head_points'
            if reason:
                result['skipped'][name] = reason
                continue
            copied[index] = xy
            result['recovered'].append(dict(point=name, coordinates=list(xy), source='face_detector',
                                           face_landmark_index=landmark, visibility=None))
        return copied, result
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
        # Malformed retained evidence cannot partially satisfy a rule.
        result.update(reason='invalid_face_evidence', recovered=[])
        return points, result
