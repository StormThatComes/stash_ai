"""Pose admission rules for retained face measurements (not head angles).

When unset, the anchor restriction is disabled for compatibility.
"""
import json
from .run_cache import metadata
import math


def compact_quality(raw):
    """Keep validated admission measurements and their visibility contract."""
    try:
        record=metadata(raw)
        selected={}
        for key in ('pixels','sharpness','pose_deviation','roll_deviation',
                    'alignment_error','alignment_scale','alignment_coverage','alignment_face_coverage'):
            number=record.get(key)
            if type(number) in (int,float) and math.isfinite(number) and number>=0:
                selected[key]=number
        for key in ('evidence_crop_sha256', 'evidence_pixel_sha256'):
            fingerprint = record.get(key)
            if isinstance(fingerprint,str) and len(fingerprint)==64 and all(c in '0123456789abcdef' for c in fingerprint):
                selected[key] = fingerprint
        thumbnail = record.get('evidence_thumbnail')
        if isinstance(thumbnail, str) and len(thumbnail) == 6144 and all(c in '0123456789abcdef' for c in thumbnail):
            selected['evidence_thumbnail'] = thumbnail
            selected['evidence_aspect'] = record.get('evidence_aspect')
        if type(record.get('alignment_valid')) is bool:
            selected['alignment_valid']=record['alignment_valid']
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
        quality=metadata(quality) if isinstance(quality,str) else quality
        return cached(quality) and quality['visible_face_fraction']>=minimum
    except (ValueError,TypeError,AttributeError):
        return False


def anchor_visibility(grouping):
    return max(grouping.get('min_anchor_visible_face_fraction') or 0,
               grouping.get('min_auxiliary_visible_face_fraction') or 0)


def restricted(grouping):
    return (grouping['max_anchor_pose_deviation'] is not None or anchor_visibility(grouping)>0 or
            any(grouping.get(k) is not None for k in ('min_anchor_face_pixels','min_anchor_sharpness')))


def measurement_required(config):
    return config.data['detection']['min_visible_face_fraction']>0 or anchor_visibility(config.data['clustering'])>0


def is_anchor(quality, grouping):
    if not quality_allows(quality,grouping,anchor=True) or not visibility_allows(quality,anchor_visibility(grouping)):
        return False
    maximum=grouping['max_anchor_pose_deviation']
    if maximum is None:
        return True
    try:
        quality=metadata(quality) if isinstance(quality,str) else quality
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
    return '('+pose+') AND ('+visibility_sql(alias,anchor_visibility(grouping))+') AND ('+quality_sql(alias,grouping,anchor=True)+')'


def auxiliary_sql(alias, grouping):
    return '('+query_sql(alias,grouping)+') AND NOT ('+anchor_sql(alias,grouping)+')'


def _floors(grouping,anchor):
    return {field:max([grouping.get('min_auxiliary_'+suffix) or 0]+
                     ([grouping.get('min_anchor_'+suffix) or 0] if anchor else []))
            for field,suffix in (('pixels','face_pixels'),('sharpness','sharpness'))}


def quality_allows(quality,grouping,*,anchor=False):
    try:
        quality=metadata(quality) if isinstance(quality,str) else quality
        if quality.get('alignment_valid') is False:
            return False
        return all(not floor or (type(quality.get(field)) in (int,float) and
            math.isfinite(quality[field]) and quality[field]>=floor)
            for field,floor in _floors(grouping,anchor).items())
    except (ValueError,TypeError,AttributeError):
        return False


def quality_sql(alias,grouping,*,anchor=False):
    conditions=[f"COALESCE(json_extract({alias}.quality,'$.alignment_valid'),1)<>0"]
    for field,floor in _floors(grouping,anchor).items():
        if floor:
            conditions.append(f"COALESCE(json_type({alias}.quality,'$.{field}') IN ('integer','real') AND "
                              f"json_extract({alias}.quality,'$.{field}')>={floor!r},0)")
    return ' AND '.join(conditions)


def query_allows(quality,grouping):
    return quality_allows(quality,grouping) and visibility_allows(quality,grouping.get('min_auxiliary_visible_face_fraction'))


def query_sql(alias,grouping):
    return '('+quality_sql(alias,grouping)+') AND ('+visibility_sql(alias,grouping.get('min_auxiliary_visible_face_fraction'))+')'
