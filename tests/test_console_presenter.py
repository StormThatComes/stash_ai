"""Rendering remains live and single-owned during blocked execution."""
from io import StringIO
from threading import Event, Thread, current_thread
import time

import pytest
from tqdm import tqdm
from stash_ai.runtime.console import ConsolePresenter, progress, present
from stash_ai.runtime.report import RunReport


class Terminal(StringIO):
    def isatty(self): return True


def test_one_owner_renders_bar_messages_and_heartbeat_during_blocked_work():
    owner_threads, messages, tick = set(), [], Event()
    class Bar(tqdm):
        def update(self, count=1):
            owner_threads.add(current_thread().name)
            return super().update(count)
    def write(message):
        owner_threads.add(current_thread().name)
        messages.append(message)
        if message.startswith('Still working:') and 'wait=evidence' in message: tick.set()
    class Output(StringIO):
        def write(self, value):
            owner_threads.add(current_thread().name)
            messages.append(value)
            if value.startswith('Still working:') and 'wait=evidence' in value: tick.set()
            return super().write(value)
    ui = ConsolePresenter(write, stream=Output(), refresh=.01, heartbeat=.02).start()
    try:
        bar = ui.bar(Bar, total=2)
        Thread(target=lambda: ui.message('warning')).start()
        bar.update(1)
        ui.update(wait='evidence')
        assert tick.wait(2), 'No liveness while execution is blocked'
        assert owner_threads == {'stash-console'}
        assert any('wait=evidence' in message for message in messages)
        bar.close()
    finally:
        ui.close()
    count = len(messages)
    time.sleep(.04)
    assert len(messages) == count and not ui.thread.is_alive()


def test_redraws_and_submissions_do_not_reset_completed_work_age():
    ui = ConsolePresenter(lambda _: None, stream=StringIO())
    assert ui.snapshot()['last_progress_age'] is None
    ui.completed = time.monotonic()-10
    ui.update(wait='worker', queued=3)
    assert ui.snapshot()['last_progress_age'] >= 10
    ui.update(completed=1)
    assert ui.snapshot()['last_progress_age'] < 1
    assert len(ui.state) <= 32
    ui.update(**{f'field{i}': i for i in range(100)})
    assert len(ui.state) == 32


def test_off_keeps_critical_messages_without_decorative_output():
    messages=[]
    ui=ConsolePresenter(messages.append, style='off', stream=Terminal()).start()
    try:
        with ui.phase('test'):
            ui.message('ERROR: important')
            ui.update(wait='read')
        ui._render()
        assert messages == ['ERROR: important']
        assert ui._status is None
    finally: ui.close()


def test_rendering_failure_does_not_change_operation_outcome():
    def fail(_): raise OSError('closed output')
    ui=ConsolePresenter(fail, stream=StringIO()).start()
    try:
        ui.call(lambda: fail('optional status'))
        assert ui.counts['console_failures'] == 1
    finally: ui.close()


def test_phase_failure_is_partial_and_cleans_owner(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr('sys.stderr', stream)
    messages=[]; report=RunReport(write=messages.append)
    with pytest.raises(ValueError):
        with present(report, 'auto') as ui:
            with ui.phase('preflight'):
                raise ValueError('invalid')
    assert 'Phase: preflight failed' in stream.getvalue()
    assert not ui.thread.is_alive()
    assert not any('preflight complete' in message for message in messages)


def test_reset_and_redirected_milestones_preserve_stdout_contract():
    output, diagnostics = [], StringIO()
    ui = ConsolePresenter(output.append, stream=diagnostics).start()
    try:
        with ui.bar(tqdm, total=2, desc='Maintenance') as bar:
            bar.update(2)
            bar.reset(total=3)
            assert bar.n == 0 and bar.total == 3
            bar.update(1)
            ui.message('Result')
        assert output == ['Result']
        assert 'inspected 1/3' in diagnostics.getvalue()
    finally:
        ui.close()
