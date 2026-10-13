"""Exclusive report output, shared by all offline JSON report entry points."""
import json
import os
from pathlib import Path
import tempfile

from stash_ai.runtime.cleanup import cleanup


def write_json_report(report, destination, *, validate_json=False):
    destination = Path(destination).expanduser().resolve()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if validate_json:
        json.loads(payload)
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


class ReservedJsonReport:
    """Reserve before work, publish only complete bytes, release on process exit."""
    def __init__(self, destination):
        self.destination = Path(destination).expanduser().resolve()
        self.lock = None

    def __enter__(self):
        from stash_ai.storage.writer_lock import acquire_writer_lock
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock = acquire_writer_lock(self.destination)
        except RuntimeError as error:
            raise RuntimeError('Another command reserves this JSON destination.') from error
        try:
            if self.destination.exists():
                raise FileExistsError('JSON destination already exists; existing exports are never overwritten.')
            return self
        except BaseException:
            cleanup([self.close])
            raise

    def publish(self, report):
        return write_json_report(report, self.destination, validate_json=True)

    def close(self):
        lock, self.lock = self.lock, None
        if lock is not None:
            lock.close()

    def __exit__(self, *args):
        cleanup([self.close])
