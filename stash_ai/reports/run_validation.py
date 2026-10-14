"""Validate historical comparison fields without coercion or database writes."""
import math


def finite_nonnegative(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def json_safe(value, depth=0):
    if depth > 64:
        return False
    if value is None or type(value) in (str, bool):
        return True
    if type(value) is int:
        return finite_nonnegative(abs(value))
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(json_safe(item, depth+1) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key,str) and json_safe(item,depth+1) for key,item in value.items())
    return False


def sanitize(run, warn):
    """Retain usable values, isolate records whose comparison fields are invalid."""
    if not isinstance(run, dict) or any(not isinstance(run.get(key),str) or not run[key]
                                       for key in ('command','status')):
        warn('invalid_run')
        return None, True
    result, invalid = dict(run), False
    def bad(code):
        nonlocal invalid
        invalid = True
        warn(code)
    def counters(value, field):
        if not isinstance(value,dict):
            bad('invalid_' + field)
            return {}
        valid = {}
        for key,item in value.items():
            if not isinstance(key,str) or not key or type(item) is not int or not finite_nonnegative(item):
                bad('invalid_' + field + '_counter')
            else:
                valid[key] = item
        return valid
    result['counts'] = counters(run.get('counts',{}), 'counts')
    outcomes = run.get('outcomes',{})
    result['outcomes'] = {}
    if not isinstance(outcomes,dict):
        bad('invalid_outcomes')
    else:
        for model,values in outcomes.items():
            if not isinstance(model,str) or not model:
                bad('invalid_outcomes')
                continue
            result['outcomes'][model] = counters(values, 'outcomes')
    for field in ('wall_seconds','cpu_seconds'):
        value = run.get(field)
        if value is not None and not finite_nonnegative(value):
            bad('invalid_' + field)
            value = None
        result[field] = value
    for field in ('models','workload'):
        value = run.get(field,{})
        if not isinstance(value,dict) or not json_safe(value) or (field=='models' and
                any(not isinstance(item,dict) for item in value.values())):
            bad('invalid_' + field)
            value = {}
        result[field] = value
    result['issues'] = counters(run.get('issues',{}), 'issues')
    for field in ('configuration',):
        if not json_safe(result.get(field)):
            bad('invalid_' + field)
            result[field] = None if field=='configuration' else {}
    if 'partial' in run and type(run['partial']) is not bool:
        bad('invalid_partial')
        result.pop('partial')
    # Invalid optional metadata cannot make otherwise usable JSON unexportable.
    for field in tuple(result):
        if field != 'metrics' and not json_safe(result[field]):
            bad('invalid_record_value')
            result[field] = None
    return result, invalid
