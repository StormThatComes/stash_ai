"""Observe the temporary pager after flushes, without exposing a temp pathname."""
from stash_ai.runtime.console import status
import hashlib
import json


class SpoolLimitExceeded(RuntimeError):
    pass


def fingerprint(hasher, scene, model, serialized):
    for field in (scene, model, serialized):
        if not isinstance(field, str):
            raise ValueError('Invalid rethreshold spool key/value type; applied=0.')
        value = field.encode('utf-8')
        hasher.update(len(value).to_bytes(8, 'big'))
        hasher.update(value)


def validate_plans(spool, expected, count, requested):
    """Decode every row and prove its exact prepared keys/content before apply."""
    from stash_ai.storage.metadata import validate_record
    from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD, validate, effective
    hasher, checked = hashlib.sha256(), 0
    for scene, model, serialized in spool.execute('SELECT scene,model,outputs FROM prepared ORDER BY id'):
        fingerprint(hasher, scene, model, serialized)
        outputs = json.loads(serialized)
        if not isinstance(outputs, dict) or not outputs or set(outputs) - set(requested):
            raise ValueError('Invalid rethreshold spool outputs; applied=0.')
        for kind, record in outputs.items():
            if not isinstance(record, dict) or record.get('mode') != ('scene' if kind == 'scene' else 'frames'):
                raise ValueError('Invalid rethreshold spool output mode; applied=0.')
            validate_record({key: value for key, value in record.items() if key != PUBLICATION_THRESHOLD})
            validate(effective(record))
        checked += 1
    if checked != count or hasher.digest() != expected:
        raise ValueError('Rethreshold spool changed after preflight; applied=0.')
    return checked


class SpoolUsage:
    def __init__(self, spool, report, limit):
        self.spool, self.report, self.limit = spool, report, max(0, int(limit))
        self.plans, self.seconds = 0, 0.
        self.row = dict(plans=0, pager_bytes=0, peak_pager_bytes=0, write_seconds=0.,
                        limit_bytes=self.limit or None, state='preflight')
        report.workload['rethreshold_spool'] = self.row

    def flushed(self, plans, seconds):
        self.plans += plans
        self.seconds += seconds
        size = self.spool.execute('PRAGMA page_count').fetchone()[0] * self.spool.execute('PRAGMA page_size').fetchone()[0]
        self.row.update(plans=self.plans, pager_bytes=size,
                        peak_pager_bytes=max(size, self.row['peak_pager_bytes']), write_seconds=self.seconds)
        status(self.report, spool=f'queued={self.plans} pager={size}B writes={self.seconds:.3f}s', applied=0)
        if self.limit and size > self.limit:
            self.row['state'] = 'limit exceeded'
            raise SpoolLimitExceeded('Rethreshold spool exceeds the configured pager limit; applied=0.')
