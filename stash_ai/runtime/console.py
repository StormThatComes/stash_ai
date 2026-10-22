"""Bounded console state and one rendering owner, also during blocking work."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from queue import Queue, Empty, Full
import sys
import threading
import time

_current = ContextVar('console_presenter', default=None)
_quiet_activity = ContextVar('quiet_console_activity', default=False)


class ConsoleDeliveryError(RuntimeError):
    """Critical output was not confirmed; an already started OS write may finish."""


class _Action:
    def __init__(self, callback):
        self.callback, self.done, self.lock = callback, threading.Event(), threading.Lock()
        self.started = self.cancelled = False
        self.result = None
        self.error = None

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if not self.started:
                self.callback = None
                self.done.set()

    def run(self):
        with self.lock:
            if self.cancelled:
                return
            self.started = True
            callback, self.callback = self.callback, None
        try:
            self.result = callback()
        except BaseException as error:
            self.error = error
        finally:
            self.done.set()


def progress(factory, *args, **kwargs):
    presenter = _current.get()
    return presenter.bar(factory, *args, **kwargs) if presenter is not None else factory(*args, **kwargs)


class ConsolePresenter:
    """Workers update snapshots; only the console thread touches tqdm/output."""
    def __init__(self, write, *, style='auto', stream=None, refresh=.25, heartbeat=5., coordination_timeout=1.,
                 status_position=4):
        self.write, self.style = write, style
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(self.stream.isatty()) and style != 'off'
        self.refresh, self.heartbeat = refresh, heartbeat
        self.status_position = status_position
        self.lock = threading.RLock()
        self.actions = Queue(maxsize=64)
        self.state, self.counts = {}, Counter()
        self.started, self.completed = time.monotonic(), None
        self.phase_name, self.phase_started = 'setup', self.started
        self.thread = None
        self.closed = False
        self._bars, self._status, self._last_line = [], None, None
        self._status_rows = []
        self._last_heartbeat = self.started
        self.samplers = {}
        self._resource_row = None
        self.stopping = self.finalized = False
        self.stop_started = None
        self.phase_outcomes = {}
        self.skipped_reasons = {}
        self.last_phase = None
        self.failure_phase = None
        self.coordination_timeout = coordination_timeout
        self.stopped = threading.Event()
        self.degraded = False
        self.status_layout = None
        self.heartbeat_layout = None
        self.heartbeat_on_progress = False

    def start(self):
        self.thread = threading.Thread(target=self._run, name='stash-console', daemon=True)
        self.thread.start()
        return self

    def snapshot(self):
        with self.lock:
            moment = time.monotonic()
            return dict(phase=self.phase_name, elapsed=moment-self.phase_started,
                        last_progress_age=moment-self.completed if self.completed is not None else None,
                        values=dict(self.state), counts=dict(self.counts))

    def update(self, *, completed=0, **values):
        if self.closed or self.style == 'off':
            return
        with self.lock:
            # Callers use fixed field names. Never retain per-scene objects/events.
            for name, value in values.items():
                if name in self.state or len(self.state) < 32:
                    if value is None:
                        self.state.pop(name, None)
                    else:
                        self.state[name] = str(value)[:180]
            if completed:
                self.completed = time.monotonic()

    def count(self, name, amount=1):
        with self.lock:
            if name in self.counts or len(self.counts) < 64:
                self.counts[name] += amount

    @contextmanager
    def phase(self, name, *, errors=None, **values):
        before_errors = errors() if errors is not None else 0
        with self.lock:
            previous = self.phase_name, self.phase_started, dict(self.state)
            self.phase_name = name
            self.last_phase = name
            self.phase_started = time.monotonic()
            self.state = {k: str(v)[:180] for k, v in values.items() if v is not None}
            if name in self.phase_outcomes or len(self.phase_outcomes) < 32:
                self.phase_outcomes[name] = 'running'
        suffix = f" [{values['phase_step']}]" if 'phase_step' in values else ''
        self.message('Phase: ' + name + suffix, decorative=True)
        status = 'complete'
        try:
            yield
        except BaseException as error:
            status = 'failed' if isinstance(error, Exception) else 'interrupted'
            if self.failure_phase is None:
                self.failure_phase = name
            if isinstance(error, KeyboardInterrupt):
                self.begin_stopping()
            raise
        finally:
            if status == 'complete' and errors is not None and errors() > before_errors:
                status = 'partial'
            self.message(f'Phase: {name} {status}{suffix}', decorative=True)
            if status == 'partial' and self.failure_phase is None:
                self.failure_phase = name
            with self.lock:
                if name in self.phase_outcomes or len(self.phase_outcomes) < 32:
                    self.phase_outcomes[name] = status
                if status == 'complete':
                    self.completed = time.monotonic()
                self.phase_name, self.phase_started, self.state = previous

    def message(self, text, *, decorative=False):
        if decorative and self.style == 'off':
            return
        if decorative:
            self.call(lambda: self._write_status(text))
        else:
            self.call(lambda: self.write(text), critical=True)

    def _write_status(self, text):
        from tqdm import tqdm
        tqdm.write(text, file=self.stream)

    def call(self, callback, default=None, *, critical=False):
        def unavailable(error=None):
            if critical:
                self.count('console_critical_unconfirmed')
                raise ConsoleDeliveryError('Console output delivery was not confirmed; renderer unavailable.') from error
            return default
        if self.closed or self.degraded:
            return unavailable()
        if self.thread is None or threading.current_thread() is self.thread:
            try:
                return callback()
            except Exception as error:
                self.count('console_failures')
                return unavailable(error)
        action = _Action(callback)
        deadline = time.monotonic() + self.coordination_timeout
        try:
            while not self.closed and not self.degraded and self.thread.is_alive():
                left = deadline-time.monotonic()
                if left <= 0:
                    break
                try:
                    self.actions.put(action, timeout=min(left, .05))
                    break
                except Full:
                    continue
            else:
                action.cancel()
                return unavailable()
            while not action.done.is_set():
                left = deadline-time.monotonic()
                if left <= 0 or self.closed or self.degraded or not self.thread.is_alive():
                    action.cancel()
                    self._abandon()
                    return unavailable()
                action.done.wait(min(left, .05))
        except BaseException:
            action.cancel()
            self._abandon()
            raise
        if action.cancelled:
            return unavailable()
        if action.error is not None:
            self.count('console_failures')
            return unavailable(action.error)
        return action.result

    def _release_pending(self):
        while True:
            try:
                self.actions.get_nowait().cancel()
            except Empty:
                return

    def _abandon(self):
        # Arbitrary blocked OS sinks cannot be killed safely. Quarantine this
        # one daemon owner; never spawn replacement writers or replay actions.
        if not self.degraded:
            self.count('console_stalls')
        self.degraded = True
        self.stopped.set()
        self._release_pending()

    def bar(self, factory, iterable=None, **kwargs):
        return OwnedBar(self, factory, iterable, kwargs)

    def register_sampler(self, name, callback):
        with self.lock:
            if len(self.samplers) < 3:
                self.samplers[name] = callback

    def remove_sampler(self, name):
        with self.lock:
            self.samplers.pop(name, None)

    def begin_stopping(self):
        with self.lock:
            if self.stopping:
                return
            self.stopping, self.stop_started = True, time.monotonic()
        self.message('Stopping: interruption received', decorative=True)

    def finish_rendering(self):
        self.finalized = True
        def close_rows():
            for bar in tuple(self._bars):
                bar.close()
            for row in (self._resource_row, *reversed(self._status_rows)):
                if row is not None:
                    row.close()
        self.call(close_rows)

    def _render(self):
        if self.style == 'off' or self.finalized:
            return
        with self.lock:
            samplers = tuple(self.samplers.values()) if not self.stopping else ()
        resources = None
        for sample in samplers:
            try:
                candidate = sample()
                if candidate:
                    resources = candidate
                    completed = (candidate.get('workers') or {}).get('last_completion')
                    if completed:
                        with self.lock:
                            self.completed = max(self.completed or 0., completed)
            except BaseException:
                self.count('resource_sample_failures')
        row = self.snapshot()
        if self.stopping:
            row.update(phase='Stopping', elapsed=time.monotonic()-self.stop_started)
        fields = {k: v for k, v in row['values'].items() if not k.startswith('_')}
        for key, label in (('_backoff_until', 'backoff remaining'), ('_deadline', 'deadline remaining')):
            if key in row['values']:
                fields[label] = f"{max(0., float(row['values'][key])-time.monotonic()):.1f}s"
        values = ' | '.join(f'{k}={v}' for k, v in fields.items())
        age = 'none yet' if row['last_progress_age'] is None else f"{row['last_progress_age']:.1f}s ago"
        line = (f"Status: {row['phase']} elapsed={row['elapsed']:.1f}s "
                f"last completed={age}" + (' | ' + values if values else ''))
        if self.tty:
            from .console_layout import status_lines, terminal_width
            layout = self.status_layout or status_lines
            self._render_status(layout(row, fields, terminal_width(self.stream)))
            from .resources import resource_line
            # Inference already owns buffer/cache rows; no duplicate resource row.
            text = resource_line(resources) if row['phase'] != 'inference' else ''
            if text:
                if self._resource_row is None:
                    from tqdm import tqdm
                    self._resource_row = tqdm(total=0, position=self.status_position + len(self._status_rows), leave=False, file=self.stream,
                                              bar_format='{desc}', dynamic_ncols=True)
                self._resource_row.set_description_str(text)
            elif self._resource_row is not None:
                self._resource_row.close()
                self._resource_row = None
        elif (time.monotonic()-self._last_heartbeat >= self.heartbeat and
              (self.heartbeat_on_progress or
               (row['last_progress_age'] if row['last_progress_age'] is not None else row['elapsed']) >= self.heartbeat)):
            self._write_status('Still working: ' + (self.heartbeat_layout(row, fields) if self.heartbeat_layout else line))
            if resources is not None:
                from .resources import resource_line
                self._write_status(resource_line(resources))
            self._last_heartbeat = time.monotonic()

    def _render_status(self, lines):
        from tqdm import tqdm
        if len(lines) != len(self._status_rows) and self._resource_row is not None:
            # A changed status height moves the resource row; clear its old slot
            # before any status row can reuse it.
            self._resource_row.close()
            self._resource_row = None
        while len(self._status_rows) > len(lines):
            self._status_rows.pop().close()
        for index, text in enumerate(lines):
            if index == len(self._status_rows):
                self._status_rows.append(tqdm(total=0, position=self.status_position + index,
                                             leave=False, file=self.stream, bar_format='{desc}', dynamic_ncols=True))
            self._status_rows[index].set_description_str(text)
        self._status = self._status_rows[0]

    def _run(self):
        next_render = time.monotonic()+self.refresh
        try:
            while not self.stopped.is_set():
                try:
                    action = self.actions.get(timeout=min(.05, max(0., next_render-time.monotonic())))
                except Empty:
                    action = None
                if action is not None:
                    action.run()
                if self.stopped.is_set():
                    break
                if time.monotonic() >= next_render:
                    try:
                        self._render()
                    except BaseException:
                        self.count('console_failures')
                    next_render = time.monotonic()+self.refresh
        finally:
            self._release_pending()
            try:
                self._dispose()
            except BaseException:
                self.count('console_failures')

    def _dispose(self):
        from .cleanup import cleanup
        callbacks = []
        for bar in tuple(self._bars):
            bar.closed = True
            if bar.target is not None:
                callbacks.append(bar.target.close)
        self._bars.clear()
        for row in (self._resource_row, *reversed(self._status_rows)):
            if row is not None:
                callbacks.append(row.close)
        self._status = self._resource_row = None
        self._status_rows.clear()
        cleanup(callbacks)

    def close(self):
        if self.closed:
            return
        self.finalized = True
        if self.thread is None or self.thread.ident is None:
            self.closed = True
            self.stopped.set()
            return
        if not self.degraded:
            self.call(self._dispose)
        self.closed = True
        self.stopped.set()
        self._release_pending()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=self.coordination_timeout)
            if self.thread.is_alive():
                self.count('console_owner_quarantined')


class OwnedBar:
    def __init__(self, owner, factory, iterable, options):
        self.owner, self.iterable, self.options = owner, iterable, dict(options)
        self.status_key = self.options.pop('status_key', 'progress')
        self.milestones = self.options.pop('milestones', True)
        self.n = self.options.get('initial', 0)
        self._total = self.options.get('total')
        if self.total is None and hasattr(iterable, '__len__'):
            self.total = len(iterable)
        self.disable = not owner.tty
        self.closed = False
        self.options.update(disable=self.disable, mininterval=owner.refresh, dynamic_ncols=True)
        self.target = owner.call(lambda: factory(total=self.total, **{
            k: v for k, v in self.options.items() if k != 'total'}))
        owner._bars.append(self)
        if self.milestones and not owner.tty and owner.style != 'off' and self.options.get('desc'):
            owner.message(self.options['desc'] + ': started', decorative=True)

    def __iter__(self):
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def update(self, count=1):
        self.n += count
        if count:
            self.owner.update(completed=count)
        return self.owner.call(lambda: self.target.update(count)) if self.target is not None and not self.disable else False

    def reset(self, total=None):
        self.n = 0
        if total is not None:
            self._total = total
        if self.target is not None:
            self.owner.call(lambda: self.target.reset(total=total))

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._total = value
        if getattr(self, 'target', None) is not None:
            self.owner.call(lambda: setattr(self.target, 'total', value))

    def set_postfix(self, *args, **kwargs):
        self.owner.update(**{self.status_key: ', '.join(f'{k}={v}' for k, v in kwargs.items() if k != 'refresh')})
        if self.target is not None and not self.disable:
            self.owner.call(lambda: self.target.set_postfix(*args, **kwargs))

    def set_postfix_str(self, text, **kwargs):
        self.owner.update(**{self.status_key: text})
        if self.milestones and self.disable and text != getattr(self, '_postfix', None):
            self.owner.message(f"{self.options.get('desc', 'Progress')}: {text}", decorative=True)
        self._postfix = text
        if self.target is not None and not self.disable:
            self.owner.call(lambda: self.target.set_postfix_str(text, **kwargs))

    def refresh(self):
        if self.target is not None and not self.disable:
            self.owner.call(self.target.refresh)

    def set_description_str(self, text, **kwargs):
        self.options['desc'] = text
        if self.target is not None and not self.disable:
            self.owner.call(lambda: self.target.set_description_str(text, **kwargs))

    def close(self):
        if not self.closed:
            self.closed = True
            if self.target is not None:
                self.owner.call(self.target.close)
            if self in self.owner._bars:
                self.owner._bars.remove(self)
            if self.milestones and not self.owner.tty and self.owner.style != 'off' and self.options.get('desc'):
                total = '?' if self.total is None else self.total
                self.owner.message(f"{self.options['desc']}: inspected {self.n}/{total} {self.options.get('unit', 'item')}", decorative=True)


@contextmanager
def present(report, style, *, status_position=4):
    presenter = ConsolePresenter(report.write, style=style, status_position=status_position)
    presenter.report = report
    original = report.write
    report.console = presenter
    report.watch_counts(presenter.counts)
    token = None
    try:
        presenter.start()
        report.write = presenter.message
        token = _current.set(presenter)
        yield presenter
    finally:
        from .cleanup import cleanup
        try:
            cleanup([('finalize console', presenter.close)], report=report)
        finally:
            if token is not None:
                _current.reset(token)
            report.write = original


@contextmanager
def phase(report, name, **values):
    presenter = getattr(report, 'console', None)
    if presenter is None or presenter.closed:
        yield
    else:
        with presenter.phase(name, **values):
            yield


def status(report, *, completed=0, **values):
    presenter = getattr(report, 'console', None)
    if presenter is not None:
        presenter.update(completed=completed, **values)


def current_report():
    presenter = _current.get()
    return getattr(presenter, 'report', None)


def skipped(report, name, reason):
    presenter = getattr(report, 'console', None)
    if presenter is not None and not presenter.closed:
        with presenter.lock:
            if name in presenter.phase_outcomes or len(presenter.phase_outcomes) < 32:
                presenter.phase_outcomes[name] = 'skipped'
                presenter.skipped_reasons[name] = str(reason)[:160]
        presenter.message(f'Phase: {name} skipped ({reason})', decorative=True)


@contextmanager
def quiet_activity():
    """Lookahead prepares future work without replacing the consumer's stage."""
    token = _quiet_activity.set(True)
    try:
        yield
    finally:
        _quiet_activity.reset(token)


@contextmanager
def activity(report, name, **values):
    presenter = getattr(report, 'console', None)
    if presenter is None or presenter.closed or _quiet_activity.get():
        yield
        return
    keys = ('stage', *values)
    with presenter.lock:
        previous = {key: presenter.state.get(key) for key in keys}
    presenter.update(stage=name, **values)
    if presenter.style == 'verbose':
        presenter.message('Activity: ' + name, decorative=True)
    try:
        yield
    finally:
        presenter.update(**previous)
