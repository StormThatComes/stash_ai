"""Run counters, bounded issues and user-facing progress coordination."""
from collections import Counter
from .timing import RunTimings
from .summary import write_summary


class RunReport:
    """Fixed-size counters and bounded error examples, even for many scenes."""
    heartbeat_seconds = 5.0
    def __init__(self, verbose=False, write=print, max_groups=6, examples=2):
        self.verbose, self.write = verbose, write
        self.max_groups, self.examples = max_groups, examples
        self.groups = {}
        self.overflow = 0
        self.counts = Counter()
        self._timing = RunTimings(self)
        self.timings, self.timing_calls = self._timing.timings, self._timing.timing_calls
        self.threshold_preflight = None
        self.diagnostics = None
        self.unread_scenes = {}
        self.dry_run_plan = None
        self.active_requests = {}

    def measure(self, stage):
        return self._timing.measure(stage)

    def request(self, operation, read):
        return self._timing.request(operation, read)

    def response_size(self, size):
        self._timing.response_size(size)

    def record_counts(self, **values):
        if self.diagnostics is not None:
            self.diagnostics.record_counts(values)

    def detail(self, message):
        if self.verbose:
            self.write(message)

    def issue(self, stage, source, scene, exc, message):
        key = (stage, source, type(exc).__name__)
        if key not in self.groups:
            if len(self.groups) >= self.max_groups:
                self.overflow += 1
                if self.verbose:
                    self.write(f'ERROR {stage} {source} scene {scene}: {message}')
                return
            self.groups[key] = dict(count=0, examples=[])
        group = self.groups[key]
        group['count'] += 1
        example = f"scene {scene}: {message}" if scene is not None else message
        example = example.replace('\n', ' ')[:350]
        if len(group['examples']) < self.examples:
            group['examples'].append(example)
        if self.verbose or group['count'] == 1:
            self.write(f"ERROR {stage} {source}: {example}")

    def progress(self, bar, counts):
        bar.set_postfix(processed=counts['processed'], reused=counts['reused'],
                        skipped=counts['skipped'], errors=counts['errors'], refresh=False)

    def record_threshold_preflight(self, checked, failed, reasons=None, applied=0, *,
                                   selected=None, skipped_no_evidence=0):
        """Keep one compact summary for an atomic threshold update attempt."""
        checked, skipped = int(checked), int(skipped_no_evidence)
        self.threshold_preflight = dict(
            selected=int(checked + skipped if selected is None else selected),
            checked=checked, skipped_no_evidence=skipped,
            failed=int(failed), applied=int(applied),
            reasons=dict(reasons or {}),
        )

    def finish(self, title, counts, cache=None):
        write_summary(self, title, counts, cache)
