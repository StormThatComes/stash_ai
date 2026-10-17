"""Width-aware status sections; activity is visible before optional metrics."""
import os
import shutil
from textwrap import wrap


def terminal_width(stream):
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    # Leave the last column free so a full row cannot trigger terminal wrapping.
    return max(1, columns - 1)


def status_lines(row, fields, width, *, max_rows=12):
    fields = dict(fields)
    sections = [('Status', row['phase'])]
    stage = fields.pop('stage', None)
    if stage:
        sections.append(('Stage', stage))

    def take(label, select):
        keys = [key for key in fields if select(key)]
        if keys:
            sections.append((label, ' | '.join(f'{key}={fields.pop(key)}' for key in keys)))

    take('Work', lambda key: key in ('scene', 'person', 'model'))
    age = 'none yet' if row['last_progress_age'] is None else f"{row['last_progress_age']:.1f}s ago"
    timing = f"elapsed={row['elapsed']:.1f}s | last completed={age}"
    for key in ('backoff remaining', 'deadline remaining'):
        if key in fields:
            timing += f' | {key}={fields.pop(key)}'
    sections.append(('Time', timing))
    take('Progress', lambda key: key in ('scenes', 'frames', 'people', 'work', 'detail', 'progress', 'frame_cache', 'checkpoint'))
    take('Rates', lambda key: key.endswith(('_eta', '_rate')))
    take('CPU', lambda key: key.startswith('cpu_'))
    take('References', lambda key: key.startswith('reference'))
    take('Details', lambda key: True)

    lines = []
    for label, value in sections:
        text = label + ': ' + ' '.join(str(value).split())
        lines.extend(wrap(text, width=width, subsequent_indent='  ' if width > 8 else '',
                          break_on_hyphens=False))
    if len(lines) > max_rows:
        omitted = len(lines) - max_rows + 1
        lines = lines[:max_rows - 1] + [f'... {omitted} more status rows'[:width]]
    return lines
