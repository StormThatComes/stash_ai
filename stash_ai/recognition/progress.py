"""Recognition work rows on the tagger's single console rendering owner."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import time

from tqdm import tqdm

from stash_ai.runtime.console import status, activity, current_report, phase, skipped

_work_rows = ContextVar('recognition_work_rows', default=())
_stage_observer = ContextVar('recognition_stage_observer', default=None)


@contextmanager
def observe_stages(callback):
    token = _stage_observer.set(callback)
    try:
        yield
    finally:
        _stage_observer.reset(token)



class RunProgress:
    """One ordered phase plan, with optional branches explicitly skipped."""
    def __init__(self, report, *, portraits=False, metadata=True):
        self.report = report
        self.plan = ['Recognition selection', 'Recognition assets']
        if not portraits:
            self.plan += ['Recognition VTT', 'Recognition identity rebuild', 'Recognition Stash publication']
        self.plan += ['Recognition performer selection']
        if metadata:
            if not portraits:
                self.plan += ['Recognition names preflight']
            self.plan += ['Recognition portraits' if portraits else 'Recognition metadata']
        if not portraits:
            self.plan += ['Recognition names starter']
        self.cursor = 0
        report.recognition_progress = self
        report.console.status_layout = partial(status_lines, verbose=report.verbose)
        report.console.heartbeat_layout = partial(heartbeat_line, verbose=report.verbose)
        report.console.heartbeat_on_progress = True
        report.console.heartbeat = 15.

    @contextmanager
    def phase(self, name, **values):
        index = self.plan.index(name)
        self._skip_until(index)
        self.cursor = index + 1
        with phase(self.report, name, phase_step=f'{index + 1}/{len(self.plan)}', **values):
            # Liveness belongs to this phase, not a previous scan or replay.
            with self.report.console.lock:
                self.report.console.completed = None
            yield

    def _skip_until(self, index):
        for name in self.plan[self.cursor:index]:
            reason = ('no eligible scenes' if name in ('Recognition identity rebuild', 'Recognition Stash publication')
                      else 'no eligible performers')
            if name == 'Recognition Stash publication' and self.report.console.phase_outcomes.get(
                    'Recognition identity rebuild') == 'failed':
                reason = 'identity rebuild failed'
            skipped(self.report, name, reason)
        self.cursor = index

    def finish(self):
        self._skip_until(len(self.plan))


def status_lines(row, fields, width, *, verbose=False):
    """Bars own counts/rates; the status area explains their place and activity."""
    from stash_ai.runtime.console_layout import status_lines as layout
    fields = dict(fields)
    step = fields.pop('phase_step', None)
    row = dict(row, phase=(f'[{step}] ' if step else '') + row['phase'])
    # The two fixed work rows already display these numbers and estimates.
    for key in ('scenes', 'frames', 'people', 'work', 'detail'):
        for suffix in ('', '_eta', '_rate', '_label', '_counts'):
            fields.pop(key + suffix, None)
    section = fields.pop('section', None)
    if section:
        stage = fields.get('stage')
        fields['stage'] = section
        if stage and stage != section and stage not in (
                row['values'].get(key + '_label') for key in ('scenes', 'frames', 'people', 'work', 'detail')):
            fields['activity'] = stage
    if not verbose:
        fields = {k: v for k, v in fields.items() if not k.startswith('reference') and
                  k not in ('cpu_blocks', 'cpu_queue', 'cpu_scenes')}
    return layout(row, fields, width)


def heartbeat_line(row, fields, *, verbose=False):
    """One periodic log line, including progress during continuously busy work."""
    step = fields.get('phase_step')
    parts = [(f'[{step}] ' if step else '') + row['phase'], f"elapsed={row['elapsed']:.0f}s"]
    for key in ('section', 'stage', 'scene', 'person'):
        if fields.get(key) and (key != 'stage' or fields[key] != fields.get('section')):
            parts.append(f'{key}={fields[key]}')
    for key in ('scenes', 'frames', 'people', 'work', 'detail'):
        if key in fields:
            text = f"{fields.get(key + '_label', key)}: {fields[key]}"
            for suffix, label in (('_rate', 'rate'), ('_eta', 'ETA'), ('_counts', 'counts')):
                if fields.get(key + suffix):
                    text += f" {label}={fields[key + suffix]}"
            parts.append(text)
    for key in ('cpu_workers', 'cpu_wait', 'frame_cache', 'backoff remaining', 'deadline remaining'):
        if fields.get(key):
            parts.append(f'{key}={fields[key]}')
    if verbose and fields.get('references'):
        parts.append('references=' + fields['references'])
    age = row['last_progress_age']
    parts.append('last completed=' + ('none in this phase' if age is None else f'{age:.0f}s ago'))
    return ' | '.join(' '.join(str(part).split()) for part in parts)


@contextmanager
def stage(label, **values):
    """Name a rebuild substage even while SQL/native work has no measurable total."""
    observer = _stage_observer.get()
    if observer is not None:
        observer(label)
    report = current_report()
    console = getattr(report, 'console', None)
    if console is None or console.style == 'off':
        yield
        return
    with console.lock:
        previous = console.state.get('section')
    status(report, section=label)
    if console.style == 'verbose':
        console.message('Step: ' + label, decorative=True)
    try:
        with activity(report, label, **values):
            yield
    finally:
        status(report, section=previous)


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
            self.bar = console.bar(tqdm, total=self.total, initial=self.n, desc=self.label,
                                   unit=self.unit, position=self.position,
                                   leave=False, dynamic_ncols=True, disable=None,
                                   status_key=self.key + '_counts',
                                   milestones=not hasattr(self.report, 'recognition_progress') or console.style == 'verbose')

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
            self.key + '_label': self.label,
            self.key + '_counts': ', '.join(f'{k}={v}' for k, v in self.postfix.items()) or None,
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
            status(self.report, **{self.key + suffix: None for suffix in ('', '_eta', '_rate', '_label', '_counts')})
            _work_rows.reset(self._row_token)
            if self.suspended is not None:
                self.suspended._open_bar()
                self.suspended._render(force=True)


@contextmanager
def work(label, total=None, *, unit='item', position=1, initial=0, **values):
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
            transient=position != 0, position=position, initial=initial) as progress:
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
