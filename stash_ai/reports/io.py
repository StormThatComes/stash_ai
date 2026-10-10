"""Exclusive report output, shared by all offline JSON report entry points."""
import json
import os
from pathlib import Path
import tempfile

from stash_ai.runtime.cleanup import cleanup


def write_json_report(report, destination):
    destination = Path(destination).expanduser().resolve()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # The temporary and destination share a filesystem. link() publishes
        # a complete file atomically and fails if another writer won the name.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=destination.parent, prefix='.' + destination.name + '.', suffix='.tmp',
                delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        cleanup([lambda: temporary.unlink(missing_ok=True) if temporary is not None else None])
    return destination
