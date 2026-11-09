"""Validated detector-only edits, shared by the sweep and desktop tuner."""
from dataclasses import replace

import yaml

from .config import _freeze, effective_data
from .config_layout import INTERNAL_PATHS, VITPOSE_BATCH, get, put, v2_document


def apply_edits(config, edits):
    """Apply all leaf changes together so interdependent backends/gates can switch atomically."""
    if not edits:
        return config, {}
    document = v2_document(config.data, source=config.source)
    previous = {}
    for parameter, value in edits.items():
        if not isinstance(parameter, str):
            raise ValueError('Expected a dotted parameter name.')
        public = INTERNAL_PATHS.get(parameter, parameter)
        if not public.startswith(('pose.', 'face.detector.')):
            raise ValueError(f'Unsupported detector parameter {parameter!r}. Use face.detector or pose settings.')
        try:
            old = get(document, public)
        except (KeyError, TypeError) as error:
            raise ValueError(f'Unknown detector parameter {parameter!r}.') from error
        if isinstance(old, (dict, list)) or isinstance(value, (dict, list)):
            raise ValueError(f'Expected a scalar for {parameter!r}.')
        previous[parameter] = get(config.data, parameter) if old is None and parameter in INTERNAL_PATHS else old
        put(document, public, value)
        if parameter == 'sapiensid.pose.batch_size':
            put(document, VITPOSE_BATCH, value)
    _, data = effective_data(document)
    source = yaml.safe_dump(document, sort_keys=False).encode()
    return replace(config, data=_freeze(data), source=source, document_version=2), previous
