"""Pose admission rules for retained face measurements (not head angles).

When unset, the anchor restriction is disabled for compatibility.
"""
import json
import math


def compact_quality(raw):
    """Keep validated admission measurements and their visibility contract."""
    try:
        record=json.loads(raw)
        selected={}
        for key in ('pixels','sharpness','pose_deviation','roll_deviation'):
            number=record.get(key)
            if type(number) in (int,float) and math.isfinite(number) and number>=0:
                selected[key]=number
        from .visibility import cached
        if cached(record):
            selected.update(visible_face_fraction=record['visible_face_fraction'],
                            visibility={'contract':record['visibility']['contract']})
        return json.dumps(selected,separators=(',',':'))
    except (TypeError, ValueError, AttributeError):
        pass
    return '{}'


def visibility_allows(quality,minimum):
    if not minimum:
        return True
    from .visibility import cached
    try:
        quality=json.loads(quality) if isinstance(quality,str) else quality
        return cached(quality) and quality['visible_face_fraction']>=minimum
    except (ValueError,TypeError,AttributeError):
        return False


def anchor_visibility(grouping):
    return max(grouping.get('min_anchor_visible_face_fraction') or 0,
               grouping.get('min_auxiliary_visible_face_fraction') or 0)


def restricted(grouping):
    return grouping['max_anchor_pose_deviation'] is not None or anchor_visibility(grouping)>0


def measurement_required(config):
    return config.data['detection']['min_visible_face_fraction']>0 or anchor_visibility(config.data['clustering'])>0


def is_anchor(quality, grouping):
    if not visibility_allows(quality,anchor_visibility(grouping)):
        return False
    maximum=grouping['max_anchor_pose_deviation']
    if maximum is None:
        return True
    try:
        quality=json.loads(quality) if isinstance(quality,str) else quality
        pose=quality.get('pose_deviation')
        return type(pose) in (int,float) and math.isfinite(pose) and 0<=pose<=maximum
    except (ValueError,TypeError,AttributeError):
        return False


def visibility_sql(alias,minimum):
    if not minimum:
        return '1=1'
    from .visibility import CONTRACT_ID
    return (f"COALESCE(json_type({alias}.quality,'$.visible_face_fraction') IN ('integer','real') "
            f"AND json_extract({alias}.quality,'$.visible_face_fraction') BETWEEN {minimum!r} AND 1 "
            f"AND json_extract({alias}.quality,'$.visibility.contract')='{CONTRACT_ID}',0)")


def anchor_sql(alias, grouping):
    maximum = grouping['max_anchor_pose_deviation']
    pose=('1=1' if maximum is None else
          f"COALESCE(json_type({alias}.quality,'$.pose_deviation') IN ('integer','real') AND "
          f"json_extract({alias}.quality,'$.pose_deviation') BETWEEN 0 AND {maximum!r},0)")
    return '('+pose+') AND ('+visibility_sql(alias,anchor_visibility(grouping))+')'


def auxiliary_sql(alias, grouping):
    return '('+visibility_sql(alias,grouping.get('min_auxiliary_visible_face_fraction'))+') AND NOT ('+anchor_sql(alias,grouping)+')'
