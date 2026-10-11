"""Bounded console state and one rendering owner, also during blocking work."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from queue import Queue, Empty
import sys
import threading
import time

_current = ContextVar('console_presenter', default=None)
_quiet_activity = ContextVar('quiet_console_activity', default=False)


def progress(factory, *args, **kwargs):
    presenter = _current.get()
    return presenter.bar(factory, *args, **kwargs) if presenter is not None else factory(*args, **kwargs)


class ConsolePresenter:
    """Workers update snapshots; only the console thread touches tqdm/output."""
    def __init__(self, write, *, style='auto', stream=None, refresh=.25, heartbeat=5.):
        self.write, self.style = write, style
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(self.stream.isatty()) and style != 'off'
        self.refresh, self.heartbeat = refresh, heartbeat
        self.lock = threading.RLock()
        self.actions = Queue(maxsize=64)
        self.state, self.counts = {}, Counter()
        self.started = self.completed = time.monotonic()
        self.phase_name, self.phase_started = 'setup', self.started
        self.thread = None
        self.closed = False
        self._bars, self._status, self._last_line = [], None, None
        self._last_heartbeat = self.started

    def start(self):
        self.thread = threading.Thread(target=self._run, name='stash-console', daemon=True)
        self.thread.start()
        return self

    def snapshot(self):
        with self.lock:
            moment = time.monotonic()
            return dict(phase=self.phase_name, elapsed=moment-self.phase_started,
                        last_progress_age=moment-self.completed, values=dict(self.state), counts=dict(self.counts))

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
            previous = self.phase_name, self.phase_started, self.completed, dict(self.state)
            self.phase_name = name
            self.phase_started = self.completed = time.monotonic()
            self.state = {k: str(v)[:180] for k, v in values.items() if v is not None}
        self.message('Phase: ' + name, decorative=True)
        status = 'complete'
        try:
            yield
        except BaseException as error:
            status = 'failed' if isinstance(error, Exception) else 'interrupted'
            raise
        finally:
            if status == 'complete' and errors is not None and errors() > before_errors:
                status = 'partial'
            self.message(f'Phase: {name} {status}', decorative=True)
            with self.lock:
                self.phase_name, self.phase_started, self.completed, self.state = previous

    def message(self, text, *, decorative=False):
        if decorative and self.style == 'off':
            return
        if decorative:
            self.call(lambda: self._write_status(text))
        else:
            self.call(lambda: self.write(text))

    def _write_status(self, text):
        from tqdm import tqdm
        tqdm.write(text, file=self.stream)

    def call(self, callback, default=None):
        if self.closed or self.thread is None or threading.current_thread() is self.thread:
            try:
                return callback()
            except Exception:
                self.count('console_failures')
                return default
        done, result = threading.Event(), []
        self.actions.put((callback, done, result))
        done.wait()
        return result[0] if result else default

    def bar(self, factory, iterable=None, **kwargs):
        return OwnedBar(self, factory, iterable, kwargs)

    def _render(self):
        if self.style == 'off':
            return
        row = self.snapshot()
        fields = {k: v for k, v in row['values'].items() if not k.startswith('_')}
        for key, label in (('_backoff_until', 'backoff remaining'), ('_deadline', 'deadline remaining')):
            if key in row['values']:
                fields[label] = f"{max(0., float(row['values'][key])-time.monotonic()):.1f}s"
        values = ' | '.join(f'{k}={v}' for k, v in fields.items())
        line = (f"Status: {row['phase']} elapsed={row['elapsed']:.1f}s "
                f"last completed={row['last_progress_age']:.1f}s ago" + (' | ' + values if values else ''))
        if self.tty:
            if self._status is None:
                from tqdm import tqdm
                self._status = tqdm(total=0, position=4, leave=False, file=self.stream,
                                    bar_format='{desc}', dynamic_ncols=True)
            self._status.set_description_str(line)
        elif (row['last_progress_age'] >= self.heartbeat
              and time.monotonic()-self._last_heartbeat >= self.heartbeat):
            self._write_status('Still working: ' + line)
            self._last_heartbeat = time.monotonic()

    def _run(self):
        next_render = time.monotonic()+self.refresh
        while True:
            try:
                callback, done, result = self.actions.get(timeout=max(0., next_render-time.monotonic()))
            except Empty:
                callback = None
            if callback is False:
                done.set()
                return
            if callback is not None:
                try:
                    result.append(callback())
                except Exception:
                    self.count('console_failures')
                finally:
                    done.set()
            if time.monotonic() >= next_render:
                try:
                    self._render()
                except Exception:
                    self.count('console_failures')
                next_render = time.monotonic()+self.refresh

    def close(self):
        if self.closed:
            return
        def close_rows():
            for bar in tuple(self._bars):
                bar.close()
            if self._status is not None:
                self._status.close()
        self.call(close_rows)
        done = threading.Event()
        self.actions.put((False, done, []))
        done.wait()
        self.thread.join()
        self.closed = True


class OwnedBar:
    def __init__(self, owner, factory, iterable, options):
        self.owner, self.iterable, self.options = owner, iterable, dict(options)
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
        if not owner.tty and owner.style != 'off' and self.options.get('desc'):
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
        self.owner.update(progress=', '.join(f'{k}={v}' for k, v in kwargs.items() if k != 'refresh'))
        if self.target is not None and not self.disable:
            self.owner.call(lambda: self.target.set_postfix(*args, **kwargs))

    def set_postfix_str(self, text, **kwargs):
        self.owner.update(progress=text)
        if self.disable and text != getattr(self, '_postfix', None):
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
            if not self.owner.tty and self.owner.style != 'off' and self.options.get('desc'):
                total = '?' if self.total is None else self.total
                self.owner.message(f"{self.options['desc']}: inspected {self.n}/{total} {self.options.get('unit', 'item')}", decorative=True)


@contextmanager
def present(report, style):
    presenter = ConsolePresenter(report.write, style=style).start()
    original = report.write
    report.console = presenter
    report.write = presenter.message
    token = _current.set(presenter)
    try:
        yield presenter
    finally:
        presenter.close()
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


def skipped(report, name, reason):
    presenter = getattr(report, 'console', None)
    if presenter is not None and not presenter.closed:
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
