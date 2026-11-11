"""Write v2 recognition YAML in pipeline order with help under each field."""
import textwrap

import yaml

from .config_help import HELP


def _scalar(value):
    # Use a single physical line for strings (including operator-supplied
    # paths). Let PyYAML format numbers so small floats round-trip as floats.
    if isinstance(value, str) and any(ord(char) < 32 or char in '\x85\u2028\u2029' for char in value):
        return yaml.safe_dump(value, default_style='"', width=2**31-1).rstrip()
    return yaml.safe_dump(value, default_flow_style=True, sort_keys=False,
                          width=2**31-1).rstrip().removesuffix('\n...')


def render_v2(document):
    if not document:
        return '{}\n'
    lines = []
    if 'version' in document:
        lines.extend(('# VTT recognition. Settings take effect when a recognition command runs.',
                      '# Face and body similarities are cosine scores. Support counts independent cues.',
                      '# Face/body sizes are source pixels. Face pose deviation is a landmark score.',
                      '# Memory/disk limits use MB (1 MB = 1048576 bytes = 1 MiB); fractions are allowed.',
                      ''))

    def write(mapping, prefix='', indent=''):
        for name, value in mapping.items():
            path = prefix + name
            group = isinstance(value, dict)
            if not prefix and lines and lines[-1]:
                lines.append('')
            line = indent + name + ':'
            lines.append(line if group else line + ' ' + _scalar(value))
            comment_indent = (indent + '  ' if group else indent) + '# '
            for part in textwrap.wrap(HELP[path], width=max(60, 112 - len(comment_indent)),
                                      break_long_words=False, break_on_hyphens=False):
                lines.append(comment_indent + part)
            if group:
                write(value, path + '.', indent + '  ')

    write(document)
    return '\n'.join(lines) + '\n'
