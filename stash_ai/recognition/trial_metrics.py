"""Top-level replay cost, including work outside production rebuilding."""
from contextlib import contextmanager
import sys
import time


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024)
    except (ImportError, OSError):
        return None


class Metrics:
    def __init__(self, enabled):
        self.enabled, self.stages = enabled, {}
        self.started, self.cpu = time.perf_counter(), time.process_time()

    @contextmanager
    def stage(self, name):
        from .progress import stage
        started, cpu = time.perf_counter(), time.process_time()
        with stage(name):
            try:
                yield
            finally:
                if self.enabled:
                    self.stages[name] = dict(elapsed_seconds=time.perf_counter()-started,
                                            cpu_seconds=time.process_time()-cpu)

    def snapshot(self):
        if not self.enabled:return None
        return dict(elapsed_seconds=time.perf_counter()-self.started, cpu_seconds=time.process_time()-self.cpu,
                    stages=self.stages, process_peak_rss_bytes=peak_rss_bytes(),
                    rss_scope='process lifetime high water mark, not incremental trial allocation')
