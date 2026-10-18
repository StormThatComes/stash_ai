"""Pose admission rules for retained face measurements (not head angles).

When unset, the anchor restriction is disabled for compatibility.
"""
import json
import math


def compact_quality(raw):
    """Copy only a validated pose number into bounded RAM replay evidence."""
    try:
        value = json.loads(raw).get('pose_deviation')
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
            return json.dumps({'pose_deviation': value}, separators=(',', ':'))
    except (TypeError, ValueError, AttributeError):
        pass
    return '{}'


def is_anchor(quality, grouping):
    maximum = grouping['max_anchor_pose_deviation']
    if maximum is None:
        return True
    try:
        if isinstance(quality, str):
            quality = json.loads(quality)
        pose = quality.get('pose_deviation')
        return type(pose) in (int, float) and math.isfinite(pose) and 0 <= pose <= maximum
    except (ValueError, TypeError, AttributeError):
        return False


def anchor_sql(alias, grouping):
    maximum = grouping['max_anchor_pose_deviation']
    if maximum is None:
        return '1=1'
    # Only validated immutable numeric configuration reaches this SQL expression.
    return f"COALESCE(json_extract({alias}.quality,'$.pose_deviation'), 2) BETWEEN 0 AND {maximum!r}"


def auxiliary_sql(alias, grouping):
    return f'NOT ({anchor_sql(alias, grouping)})'
