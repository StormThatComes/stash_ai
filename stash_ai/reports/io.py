"""Exclusive report output, shared by all offline JSON report entry points."""
import json
from pathlib import Path


def write_json_report(report, destination):
    destination = Path(destination).expanduser().resolve()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
    return destination
