"""Observe the temporary pager after flushes, without exposing a temp pathname."""
from stash_ai.runtime.console import status


class SpoolLimitExceeded(RuntimeError):
    pass


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
