"""Separate readable PNG/JSON companions; retained clean/annotated crops stay intact."""
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap

from .alignment_audit import _png, _write_json


COLORS = {'pass': '#167044', 'fail': '#b42318', 'unknown': '#975b00', 'disabled': '#667085'}
SOURCES = {
    'current_configuration': 'Current YAML quality gates',
    'current_configuration_hash_verified': 'Settings verified against retained policy',
    'retained_configuration_hash_verified': 'Historical settings verified against retained policy',
    'current_configuration_only_historical_unavailable': 'Current YAML only; historical settings unavailable',
}


def _value(value):
    if value is None:
        return 'unknown'
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def export(image, admission, directory, name, *, observation, role, rejection):
    base = directory / name
    png = base.with_name(base.stem+'-thresholds.png')
    sidecar = base.with_name(base.stem+'-thresholds.json')
    payload = dict(observation_id=observation, recorded_role=role, retained_rejection=rejection,
                   **admission)
    _write_json(sidecar, payload)
    result = dict(status=image['status'], file=None, json_file=sidecar.name)
    if image['status'] != 'exported':
        return result
    lines = [(f'{admission["evidence_kind"].upper()} ADMISSION | recorded role: {role}', '#101828'),
             (base.name, '#475467'),
             ('PASS/FAIL describe individual gates, not proof of identity or a complete rescan.', '#475467'),
             (f'Retained rejection: {rejection or "none"}. Exact values and gate boundaries: companion JSON.', '#475467')]
    for group in admission['groups']:
        lines.append(('', '#101828'))
        lines.append((group['stage'].replace('_', ' ').upper()+' | '+group['status'].upper()+
                      ' | '+SOURCES[group['configuration_source']], '#101828'))
        for row in group['checks']:
            state = row['status']
            text = (f'[{state.upper()}] {row["parameter"]}: {_value(row["measured"])} '
                    f'{row["operator"]} {_value(row["threshold"])}')
            if state == 'disabled':
                configured = 'null' if row['threshold'] is None else _value(row['threshold'])
                text = f'[DISABLED] {row["parameter"]} = {configured} (sample: {_value(row["measured"])})'
            if row.get('label'):
                text = f'{row["label"]} | '+text
            lines.append((text, COLORS[state]))
        if group.get('context'):
            lines.append(('Frame context only: '+', '.join(f'{k}={v}' for k,v in group['context'].items()), '#475467'))
        for text in wrap(group.get('note', ''), width=120):
            lines.append((text, '#475467'))
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow 9.4 remains supported by the application.
        font = ImageFont.load_default()
    width = max(900, int(max(font.getlength(text) for text, _ in lines))+32)
    # Reuse the overlay when it exists, preserving its pixels in this companion.
    source = image.get('annotations', {}).get('file') or image['file']
    with Image.open(directory / source) as crop:
        width = max(width, crop.width+32)
        with Image.new('RGB', (width, crop.height+48+22*len(lines)), '#f8fafc') as canvas:
            canvas.paste(crop, (16, 16))
            draw = ImageDraw.Draw(canvas)
            for index, (text, color) in enumerate(lines):
                draw.text((16, crop.height+32+22*index), text, fill=color, font=font)
            _png(canvas, png)
    return dict(result, status='exported', file=png.name)
