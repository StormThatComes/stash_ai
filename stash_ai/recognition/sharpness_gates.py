"""One decision rule for scans, retained evidence, SQL replay and inspection."""
import math

FIELDS = ('sharpness_mode', 'min_focus_sharpness', 'min_motion_sharpness')


def defaults(prefix=''):
    return {prefix+k: v for k, v in zip(FIELDS, ('legacy', 0., 0.))}


def policy(settings, prefix='', legacy='min_sharpness'):
    return dict(min_sharpness=settings.get(legacy) or 0,
                **{key: settings.get(prefix+key, value) for key, value in defaults().items()})


def valid(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def estimated(quality):
    return (valid(quality.get('sharpness_confidence')) and .6 <= quality['sharpness_confidence'] <= 1
            and all(valid(quality.get(k)) and quality[k] <= 100 for k in ('focus_sharpness', 'motion_sharpness')))


def floors(quality, settings):
    if settings.get('sharpness_mode') == 'split' and estimated(quality):
        return [('min_'+kind+'_sharpness', kind+'_sharpness', settings.get('min_'+kind+'_sharpness', 0))
                for kind in ('focus', 'motion')]
    threshold = settings.get('profile_min_sharpness') if quality.get('face_profile') else None
    return [('min_sharpness', 'sharpness', settings.get('min_sharpness', 0) if threshold is None else threshold)]


def failure(quality, settings):
    for key, metric, floor in floors(quality, settings):
        if floor and (not valid(quality.get(metric)) or quality[metric] < floor):
            return metric
    return None


def active(settings):
    return bool(settings.get('min_sharpness') or settings.get('sharpness_mode') == 'split'
                and (settings.get('min_focus_sharpness') or settings.get('min_motion_sharpness')))


def sql(alias, settings):
    def minimum(metric, floor):
        if not floor:
            return '1=1'
        return (f"COALESCE(json_type({alias}.quality,'$.{metric}') IN ('integer','real') AND "
                f"json_extract({alias}.quality,'$.{metric}') BETWEEN {floor!r} AND 1.7976931348623157e308,0)")
    legacy = minimum('sharpness', settings.get('min_sharpness') or 0)
    if settings.get('sharpness_mode') != 'split':
        return legacy
    ready = ' AND '.join(f"COALESCE(json_type({alias}.quality,'$.{key}') IN ('integer','real') AND "
                        f"json_extract({alias}.quality,'$.{key}') BETWEEN {low} AND {high},0)"
                        for key, low, high in (('sharpness_confidence', .6, 1),
                                              ('focus_sharpness', 0, 100), ('motion_sharpness', 0, 100)))
    split = ' AND '.join(minimum(kind+'_sharpness', settings.get('min_'+kind+'_sharpness', 0))
                         for kind in ('focus', 'motion'))
    return f'(CASE WHEN {ready} THEN ({split}) ELSE ({legacy}) END)'


def validate(settings, name, prefix=""):
    from .config import _number
    if settings.get(prefix+'sharpness_mode', 'legacy') not in ('legacy', 'split'):
        raise ValueError(name+'.'+prefix+'sharpness_mode must be legacy or split.')
    for key in FIELDS[1:]:
        _number(settings.get(prefix+key, 0), name+'.'+prefix+key, 0, 100)
