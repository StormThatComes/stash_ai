"""Recognition work rows on the tagger's single console rendering owner."""
from contextlib import contextmanager
import time

from tqdm import tqdm

from stash_ai.runtime.console import progress as console_bar, status


class WorkProgress:
    """Throttle UI updates independently of inference and SQLite checkpoints."""
    def __init__(self, report, label, total, *, unit='scene', key='scenes', initial=0, transient=False):
        self.report, self.label, self.total = report, label, total
        self.unit, self.key, self.initial = unit, key, initial
        self.n = self.rendered = initial
        self.transient, self.bar = transient, None
        self.started = self.last_render = time.monotonic()
        self.postfix = {}

    def __enter__(self):
        console = getattr(self.report, 'console', None)
        self.enabled = console is not None and console.style != 'off'
        # Redirected runs get stage milestones and heartbeats, never a pair of
        # frame milestones for every scene. Terminal rows have fixed positions.
        if self.enabled and (console.tty or not self.transient):
            self.bar = console_bar(tqdm, total=self.total, initial=self.initial, desc=self.label,
                                   unit=self.unit, position=1 if self.transient else 0,
                                   leave=False, dynamic_ncols=True, disable=None)
        self._render(force=True)
        return self

    def advance(self, amount=1, **postfix):
        self.set_completed(self.n + amount, **postfix)

    def set_completed(self, completed, **postfix):
        self.n = completed
        self.postfix.update(postfix)
        self._render(force=self.n == self.total)

    def _render(self, *, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_render < .25:
            return
        elapsed, observed = now - self.started, self.n - self.initial
        rate = observed / elapsed if observed > 0 and elapsed > 0 else None
        eta = ('00:00' if self.n == self.total else
               tqdm.format_interval((self.total - self.n) / rate) if rate else 'warming up')
        delta = self.n - self.rendered
        status(self.report, completed=delta, **{
            self.key: f'{self.n}/{self.total}', self.key + '_eta': eta,
            self.key + '_rate': f'{rate:.2f} {self.unit}/s' if rate else 'warming up',
        })
        if self.bar is not None:
            if self.postfix:
                self.bar.set_postfix(refresh=False, **self.postfix)
            if delta:
                self.bar.update(delta)
        self.rendered, self.last_render = self.n, now

    def __exit__(self, *error):
        try:
            self._render(force=True)
        finally:
            if self.bar is not None:
                self.bar.close()
            status(self.report, **{self.key: None, self.key + '_eta': None, self.key + '_rate': None})


@contextmanager
def frame_progress(report, scene):
    """The scan supplies its checkpoint cursor before loading any model."""
    work = None
    def update(*, completed, total, reused=False):
        nonlocal work
        if work is None:
            work = WorkProgress(report, f'VTT scene {scene}', total, unit='frame', key='frames',
                                initial=completed, transient=True)
            work.__enter__()
            status(report, frame_cache='reused' if reused else 'resumed' if completed else 'fresh')
        else:
            work.set_completed(completed)
    try:
        yield update
    finally:
        if work is not None:
            work.__exit__(None, None, None)
        status(report, frame_cache=None)
