"""The logical vector budget used by matching, not process RAM or GPU memory."""
import time


def usage_line(values):
    used, limit, peak = (values[key] for key in ('bytes', 'budget_bytes', 'peak_bytes'))
    mib = 1048576
    return (f'{used / mib:.2f}/{limit / mib:.2f} MiB ({100 * used / limit:.1f}%) '
            f'peak={peak / mib:.2f} MiB ({100 * peak / limit:.1f}%)')


class ReferenceUsage:
    """Account at existing budget checks; throttle optional console snapshots."""
    def __init__(self, limit, progress=None):
        self.limit, self.progress = limit, progress
        self.bytes = self.people = self.samples = self.peak = self.metadata_bytes = 0
        self.stage, self.last_report = 'loading', None
        self.last_values = None
        from .run_cache import current
        self.cache = current()
        if self.cache is not None:
            import weakref
            weakref.finalize(self,self.cache.reserve,('bank',id(self)),0)

    def begin(self, stage, *, reset=False, report=True):
        self.stage = stage
        if reset:
            self.bytes = self.people = self.samples = self.metadata_bytes = 0
            if self.cache is not None:
                self.cache.reserve(('bank',id(self)),0)
        if report:
            self.emit(force=True)

    def add(self, vectors, *, people=0, kind):
        self.add_counts(sum(vector.nbytes for vector in vectors), len(vectors), people=people, kind=kind)

    def add_counts(self, size, samples, *, people=0, kind):
        """Reuse exact accounting for an already validated immutable snapshot."""
        self.bytes += size
        self.people += people
        self.samples += samples
        self.peak = max(self.peak, self.bytes)
        if self.bytes > self.limit:
            self.emit(force=True)
            raise ValueError(
                f'{kind} face references exceed the configured RAM budget: '
                f'required={self.bytes} bytes, limit={self.limit} bytes '
                f'(clustering.max_reference_bytes), people={self.people}, samples={self.samples}.')
        if self.cache is not None:
            self.cache.reserve(('bank',id(self)),self.bytes+self.samples*128+self.people*256+self.metadata_bytes)
        self.emit()

    def add_metadata(self, size):
        self.metadata_bytes += size
        if self.cache is not None:
            self.cache.reserve(('bank',id(self)),self.bytes+self.samples*128+self.people*256+self.metadata_bytes)

    def emit(self, *, force=False):
        if self.progress is None:
            return
        now = time.monotonic()
        values = self.bytes, self.people, self.samples, self.peak, self.stage
        if values == self.last_values and self.last_report is not None and now - self.last_report < 1:
            return
        if force or self.last_report is None or now - self.last_report >= 1:
            self.last_values = values
            self.last_report = now
            self.progress(bytes=self.bytes, budget_bytes=self.limit, peak_bytes=self.peak,
                          people=self.people, samples=self.samples, stage=self.stage)
