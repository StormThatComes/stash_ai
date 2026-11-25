"""Readable retained blur measurements, separate from admission/advice gates."""
from .sharpness_gates import estimated, valid
from .split_blur import METHOD

REASONS = {
    'input_too_small': 'Input is too small for the 15x15 measurement footprint',
    'no_foreground_interior': 'No foreground interior contains the full 15x15 footprint',
    'insufficient_edge_support': 'Too few reliable edges or tiles in one or more directions',
    'invalid_spread': 'Edge widths do not support a valid blur covariance',
    'inconsistent_edge_widths': 'Directional edge widths disagree with the blur model',
    'mask_unavailable': 'The required foreground mask is unavailable',
}
FIELDS = ('sharpness', 'focus_sharpness', 'motion_sharpness', 'sharpness_confidence',
          'sharpness_split_method', 'sharpness_split_status', 'sharpness_split_reason',
          'sharpness_split_diagnostics', 'sharpness_space', 'sharpness_input_size',
          'sharpness_method', 'sharpness_pixels')


def summary(quality):
    return {key: quality.get(key) for key in FIELDS}


def reason(quality):
    if estimated(quality):
        return 'Supported edge directions and a consistent blur fit'
    retained = quality.get('sharpness_split_reason')
    if retained in REASONS:
        return REASONS[retained]
    if quality.get('sharpness_confidence') is None:
        return 'Split measurements were not retained; rerun recognition for this scene'
    if valid(quality.get('sharpness_confidence')) and 0 < quality['sharpness_confidence'] < .6:
        return 'Fit confidence is below the required 0.60'
    if retained == 'estimated':
        return 'Retained components or confidence are invalid or incomplete; rerun recognition'
    return 'Exact cause was not retained with this measurement; rerun recognition for this scene'


def value(quality, key, precision=2):
    measured = quality.get(key)
    return f'{measured:.{precision}f}' if valid(measured) else 'unavailable'


def gate_note(quality, settings):
    if settings.get('sharpness_mode', 'legacy') != 'split':
        return 'Legacy mode: uses min_sharpness; split measurements are informational.'
    confidence = value(quality, 'sharpness_confidence', 3)
    if estimated(quality):
        return f'Split mode: uses focus/motion floors in place of min_sharpness; split confidence {confidence}.'
    return f'Legacy fallback (split requested): {reason(quality)}. Split confidence {confidence}.'


def describe(quality):
    lines = ['Sharpness measurements (0-100; higher means sharper)',
             f"Legacy sharpness: {value(quality, 'sharpness')}",
             f"Focus sharpness (isotropic softness): {value(quality, 'focus_sharpness')}",
             f"Motion sharpness (directional smear): {value(quality, 'motion_sharpness')}",
             f"Split confidence: {value(quality, 'sharpness_confidence', 3)} (required >= 0.60)",
             'Confidence 0 means the split estimate is unavailable, not zero sharpness.',
             'Confidence measures fit consistency; it is not a calibrated probability or a diagnosis of the blur cause.',
             f"Split status: {'estimated' if estimated(quality) else 'unavailable'}; {reason(quality)}."]
    method = quality.get('sharpness_split_method')
    lines.append(f'Estimation method: {method or "not retained"}')
    if method != METHOD:
        lines.append('Older or missing split measurement: rerun recognition and refresh to use equal-length profiles and retain the cause.')
    size = quality.get('sharpness_input_size')
    dimensions = 'x'.join(map(str, size)) if isinstance(size, (list, tuple)) else 'not retained'
    lines.append(f"Measured input: {dimensions}; space: {quality.get('sharpness_space') or 'not retained'}")
    lines.append(f"Region: {quality.get('sharpness_method') or 'not retained'}; "
                 f"supported interior pixels: {value(quality, 'sharpness_pixels', 0)}")
    diagnostics = quality.get('sharpness_split_diagnostics') or {}
    edges, tiles = diagnostics.get('edge_counts'), diagnostics.get('tile_counts')
    if isinstance(edges, list) and isinstance(tiles, list) and len(edges) == len(tiles) == 4:
        lines.append('Reliable edges / tiles by direction (requires >=8 edges and >=2 tiles in each):')
        lines.append('  '+ '; '.join(f'{angle} deg: {edge} / {tile}'
                     for angle, edge, tile in zip((0, 90, 45, 135), edges, tiles)))
    lines.append(f"Fit residual: {value(diagnostics, 'fit_residual', 3)} (required < 0.30)")
    return lines


def decision_text(admission):
    from .admission_panel import SOURCES
    lines = describe(admission.get('sharpness') or {})
    lines.extend(('', 'Sharpness gates by stage'))
    for group in admission.get('groups', []):
        checks = [row for row in group['checks'] if row['parameter'].endswith('_sharpness')]
        if not checks:
            continue
        lines.extend(('', group['stage'], SOURCES[group['configuration_source']]))
        for row in checks:
            lines.append(f"[{row['status'].upper()}] {row.get('yaml_parameter', row['parameter'])}: "
                         f"{value(row, 'measured')} {row['operator']} {row['threshold']}")
            if row.get('note'):
                lines.append(row['note'])
    return '\n'.join(lines)
