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
        self.bytes = self.people = self.samples = self.peak = 0
        self.stage, self.last_report = 'loading', None

    def begin(self, stage, *, reset=False):
        self.stage = stage
        if reset:
            self.bytes = self.people = self.samples = 0
        self.emit(force=True)

    def add(self, vectors, *, people=0, kind):
        self.bytes += sum(vector.nbytes for vector in vectors)
        self.people += people
        self.samples += len(vectors)
        self.peak = max(self.peak, self.bytes)
        if self.bytes > self.limit:
            self.emit(force=True)
            raise ValueError(
                f'{kind} face references exceed the configured RAM budget: '
                f'required={self.bytes} bytes, limit={self.limit} bytes '
                f'(clustering.max_reference_bytes), people={self.people}, samples={self.samples}.')
        self.emit()

    def emit(self, *, force=False):
        if self.progress is None:
            return
        now = time.monotonic()
        if force or self.last_report is None or now - self.last_report >= 1:
            self.last_report = now
            self.progress(bytes=self.bytes, budget_bytes=self.limit, peak_bytes=self.peak,
                          people=self.people, samples=self.samples, stage=self.stage)
