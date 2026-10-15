"""Recognition work rows on the tagger's single console rendering owner."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

from tqdm import tqdm

from stash_ai.runtime.console import progress as console_bar, status, activity, current_report

_work_rows = ContextVar('recognition_work_rows', default=())


class WorkProgress:
    """Throttle UI updates independently of inference and SQLite checkpoints."""
    def __init__(self, report, label, total, *, unit='scene', key='scenes', initial=0, transient=False, position=None):
        self.report, self.label, self.total = report, label, total
        self.unit, self.key, self.initial = unit, key, initial
        self.n = self.rendered = initial
        self.transient, self.bar = transient, None
        self.position = (1 if transient else 0) if position is None else position
        self.started = self.last_render = time.monotonic()
        self.postfix = {}

    def __enter__(self):
        console = getattr(self.report, 'console', None)
        self.enabled = console is not None and console.style != 'off'
        self.suspended = next((row for row in reversed(_work_rows.get())
                               if row.position == self.position and row.bar is not None), None)
        if self.suspended is not None:
            self.suspended.bar.close()
            self.suspended.bar = None
        self._row_token = _work_rows.set((*_work_rows.get(), self))
        self._open_bar()
        self._render(force=True)
        return self

    def _open_bar(self):
        console = getattr(self.report, 'console', None)
        # Redirected runs get stage milestones and heartbeats, never a pair of
        # frame milestones for every scene. Terminal rows have fixed positions.
        if self.enabled and (console.tty or not self.transient):
            self.bar = console_bar(tqdm, total=self.total, initial=self.n, desc=self.label,
                                   unit=self.unit, position=self.position,
                                   leave=False, dynamic_ncols=True, disable=None)

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
        eta = ('unknown' if self.total is None else '00:00' if self.n == self.total else
               tqdm.format_interval((self.total - self.n) / rate) if rate else 'warming up')
        delta = self.n - self.rendered
        status(self.report, completed=delta, **{
            self.key: f'{self.n}/{self.total if self.total is not None else "?"}', self.key + '_eta': eta,
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
            _work_rows.reset(self._row_token)
            if self.suspended is not None:
                self.suspended._open_bar()
                self.suspended._render(force=True)


@contextmanager
def work(label, total=None, *, unit='item', position=1, **values):
    """Shared run-owned progress, also for calls below injected service boundaries."""
    report = current_report()
    console = getattr(report, 'console', None)
    if callable(total):
        # Counting is optional telemetry, never another BLOB materialization.
        if console is None or console.style == 'off':
            total = None
        else:
            try:
                total = total()
            except Exception:
                report.counts['recognition_progress_total_failures'] += 1
                total = None
    with activity(report, label, **values), WorkProgress(
            report, label, total, unit=unit, key='work' if position == 0 else 'detail',
            transient=position != 0, position=position) as progress:
        yield progress


class _Tracked:
    def __init__(self, iterable, progress):
        self.iterator, self.progress, self.pending = iter(iterable), progress, False

    def completed(self):
        if self.pending:
            self.progress.advance()
            self.pending = False

    def __iter__(self):
        return self

    def __next__(self):
        # Advancing means the previous loop body completed. An exception in
        # that body must not contribute a completed record.
        self.completed()
        value = next(self.iterator)
        self.pending = True
        return value


@contextmanager
def tracking(iterable, label, *, total=None, unit='item', position=1, **values):
    """Close rows deterministically, including early return, failure and Ctrl+C."""
    report = current_report()
    console = getattr(report, 'console', None)
    if console is None or console.style == 'off':
        yield iterable
        return
    if total is None and hasattr(iterable, '__len__'):
        total = len(iterable)
    with work(label, total, unit=unit, position=position, **values) as progress:
        items = _Tracked(iterable, progress)
        try:
            yield items
            items.completed()
        finally:
            items.iterator = None


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
