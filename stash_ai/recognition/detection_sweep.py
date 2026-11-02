"""Independent parameter trials on one original VTT cue, stacked without resizing."""
import csv
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from stash_ai.runtime.cleanup import closing
from .alignment_audit import _png
from .config import _freeze, parse_yaml, plain, validate


MAX_VARIANTS = 64  # In addition to the baseline.
MAX_PIXELS = 40_000_000  # Bound the final RGB canvas before allocating it.
FACE_PARAMETERS = {'models.detector', 'detection.score_threshold',
                   'detection.nms_threshold', 'detection.max_faces_per_frame'}


def load_variants(path, config):
    """Validate the whole table before any I/O to Stash or detector construction."""
    path = Path(path).expanduser()
    with path.open('rb') as stream:
        raw = stream.read(131073)
    if len(raw) > 131072:
        raise ValueError('Detection sweep table exceeds the byte limit.')
    if path.suffix.lower() == '.csv':
        reader = csv.DictReader(StringIO(raw.decode('utf-8-sig')), strict=True)
        if reader.fieldnames != ['parameter', 'value']:
            raise ValueError('Detection sweep CSV header must be parameter,value.')
        rows = []
        for index, row in enumerate(reader, 1):
            if None in row or row['value'] is None or not row['value'].strip():
                raise ValueError(f'Detection sweep row {index}: expected parameter,value.')
            rows.append(dict(parameter=row['parameter'].strip(),
                             value=parse_yaml(row['value'].encode(), context=f'Sweep row {index}')))
    else:
        rows = parse_yaml(raw, context='Detection sweep')
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_VARIANTS:
        raise ValueError(f'Detection sweep requires a table of 1..{MAX_VARIANTS} parameter/value rows.')
    variants = [(config, dict(variant_index=0, parameter=None, value=None, baseline_value=None))]
    for index, row in enumerate(rows, 1):
        prefix = f'Detection sweep row {index}'
        if not isinstance(row, dict) or set(row) != {'parameter', 'value'}:
            raise ValueError(f'{prefix}: expected exactly parameter and value.')
        parameter = row['parameter']
        if not isinstance(parameter, str) or not (
                parameter in FACE_PARAMETERS or parameter.startswith('sapiensid.pose.')):
            raise ValueError(f'{prefix}: unsupported detector parameter {parameter!r}. '
                             'Use face score/NMS/max_faces_per_frame, models.detector or sapiensid.pose leaves; '
                             'quality, anchor and identity gates are not applied by this audit.')
        data = plain(config.data)  # Every row starts from baseline, never the previous row.
        target = data
        parts = parameter.split('.')
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f'{prefix}: unknown parameter {parameter!r}.')
            target = target[part]
        key = parts[-1]
        if not isinstance(target, dict) or key not in target or isinstance(target[key], (dict, list)):
            raise ValueError(f'{prefix}: unknown parameter or non-leaf {parameter!r}.')
        old = target[key]
        target[key] = row['value']
        try:
            validate(data)
        except (ValueError, TypeError) as error:
            raise ValueError(f'{prefix} ({parameter}): {error}') from error
        variants.append((replace(config, data=_freeze(data)),
                         dict(variant_index=index, parameter=parameter, value=row['value'], baseline_value=old)))
    return variants


def _value(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def _header_lines(record, scene_id):
    title = 'Baseline (current YAML)' if record['variant_index'] == 0 else f"Trial {record['variant_index']}"
    lines = [f"{title} | scene {scene_id} | cue {record['cue_index']:06d} | "
             f"{record['start_seconds']:.3f}-{record['end_seconds']:.3f}s"]
    if record['parameter'] is not None:
        lines.append(f"{record['parameter']} = {_value(record['value'])} "
                     f"(baseline: {_value(record['baseline_value'])})")
    lines.append(f"Faces: {len(record['faces'])} | People: {len(record['people'])} | "
                 f"Matched faces: {sum(face['person_index'] is not None for face in record['faces'])}")
    return lines


def _wrap(lines, font, width):
    # Pixel-based wrapping also handles long dotted names and local model paths.
    result = []
    for line in lines:
        chunk = ''
        for char in line:
            if chunk and font.getlength(chunk + char) > width:
                result.append(chunk)
                chunk = ''
            chunk += char
        result.append(chunk)
    return result


def _stack(records, paths, scene_id, destination):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    bounds = font.getbbox('Ag')
    line_height = bounds[3] - bounds[1] + 7
    width = max(640, *(record['output_dimensions'][0] for record in records))
    height = 0
    headers = []
    for record in records:
        lines = _wrap(_header_lines(record, scene_id), font, width - 24)
        header_height = 20 + line_height * len(lines)
        record['panel_offset'] = [0, height]
        record['source_offset'] = [0, height + header_height]
        record['header_lines'] = lines
        headers.append((lines, header_height))
        height += header_height + record['output_dimensions'][1] + 12
    if width * height > MAX_PIXELS:
        raise ValueError(f'Detection sweep PNG would exceed {MAX_PIXELS:,} pixels; use fewer trials.')
    with Image.new('RGB', (width, height), (24, 24, 24)) as canvas:
        painter = ImageDraw.Draw(canvas)
        for record, path, (lines, header_height) in zip(records, paths, headers):
            y = record['panel_offset'][1]
            painter.rectangle((0, y, width - 1, y + header_height - 1), fill=(40, 44, 52))
            for line_index, line in enumerate(lines):
                painter.text((12, y + 8 + line_index * line_height), line, font=font, fill=(245, 245, 245))
            with Image.open(path) as panel:
                canvas.paste(panel, tuple(record['source_offset']))
        _png(canvas, destination)
    return [width, height]


def audit_sweep(source, variants, scene_id, cue_index, directory, services, detector_factory, report, *, mask_options=None):
    from stash_ai.runtime.console import activity
    from .detection_audit import render_frame
    from .progress import WorkProgress

    records, paths = [], []
    filename = f'scene-{scene_id}_cue-{cue_index:06d}_sweep.png'
    # Read the unannotated crop once. Reusing rendered PNGs would contaminate the
    # input with annotations; reusing detection results would hide threshold effects.
    with closing(source.frames_indices([cue_index]), name='close sweep frames') as images:
        try:
            index, cue, image = next(images)
        except StopIteration:
            raise ValueError('Detection sweep did not receive the selected VTT cue.') from None
        if index != cue_index:
            raise ValueError('Detection sweep source returned a different cue index.')
        with closing(image, name='close sweep image'), TemporaryDirectory(dir=directory) as temporary:
            # Keep only one model set and one rendered panel in RAM at a time.
            with WorkProgress(report, 'Detection parameter sweep', len(variants), unit='trial', key='trials') as progress:
                for current, metadata in variants:
                    label = 'baseline' if metadata['parameter'] is None else (
                        f"trial {metadata['variant_index']}: {metadata['parameter']}={_value(metadata['value'])}")
                    with activity(report, f'Load detectors for {label}'):
                        assets = services.assets(current, keys=[current.data['models']['detector']], write=report.write)
                        detector = detector_factory(current, assets)
                    with closing(detector, name='close sweep detectors'), activity(report, f'Detect {label}'):
                        # Isolate even a backend that mutates its input.
                        with image.copy() as trial_image, render_frame(
                                trial_image, detector, scene_id, index, cue,
                                mask_options=mask_options) as (annotated, record):
                            record.update(metadata, file=filename)
                            path = Path(temporary) / f"{metadata['variant_index']:03d}.png"
                            _png(annotated, path)
                            records.append(record)
                            paths.append(path)
                    progress.advance(faces=len(record['faces']), people=len(record['people']))
            dimensions = _stack(records, paths, scene_id, directory / filename)
    return dict(scene_id=scene_id, cue_index=cue_index, file=filename, output_dimensions=dimensions,
                comparison='baseline_then_independent_single_parameter_trials',
                counts=dict(cues_exported=1, variants_exported=len(records),
                            faces_detected=sum(len(r['faces']) for r in records),
                            people_detected=sum(len(r['people']) for r in records)),
                frames=records, vtt_source_fingerprint=source.fingerprint, vtt_coverage=dict(source.coverage))
