"""Bounded spawned trials; only the caller applies results and writes journals."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import AbstractContextManager
import multiprocessing
import signal
import time

_cancel = None


def initialize(event):
    global _cancel
    from stash_ai.runtime.cancellation import Cancellation
    # Ctrl-C belongs to the coordinator; children receive cooperative cancellation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _cancel = Cancellation()
    _cancel.event = event


def work(request):
    from stash_ai.runtime.cancellation import use
    from .solver_trial import evaluate
    with use(_cancel):
        return evaluate(request)


class Trials(AbstractContextManager):
    def __init__(self, execution, factory, context, journal=None, progress=None):
        self.execution, self.factory, self.context, self.journal = execution, factory, context, journal
        self.pending, self.ready = OrderedDict(), {}
        self.progress, self.last_progress = progress, 0.
        self.history = list(journal.document.get('schedule', ())) if journal else []
        self.cursor = 0
        self.executor = None
        self.event = multiprocessing.get_context('spawn').Event()
        self.started = time.monotonic()
        self.peak = 0
        from pathlib import Path
        from .solver_shared import Arrays
        self.shared_directory = str(Path(context['seed_path']).parent/'numerical')
        Arrays.prepare(self.shared_directory)
        self.shared_bytes = min(execution.memory_bytes // 16, 1024**3)

    def __enter__(self):
        return self

    def event_record(self, kind, name):
        if self.journal:
            self.journal.schedule(kind, name)

    def dispatch(self, name, settings, *, restored=False):
        from .trial_evaluation import settings_for_replay
        settings = settings_for_replay(settings, self.execution)
        if name in self.pending:
            raise ValueError('Duplicate pending trial.')
        if self.progress:
            self.progress(name, 'start', None)
        future = None
        completed = self.journal and name in self.journal.records
        if self.journal:
            status = self.journal.document.get('tasks', {}).get(name, {}).get('status', 'completed') if completed else 'queued'
            self.journal.task(name, settings, status)
        cached = self.journal.raw_result(name, settings) if self.journal else None
        if cached is not None:
            self.ready[name] = cached
        if not completed and cached is None:
            if self.executor is None:
                self.executor = ProcessPoolExecutor(max_workers=self.execution.trials,
                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                    initargs=(self.event,))
            from dataclasses import replace
            request = replace(self.factory(name, settings, self.context),
                shared_directory=self.shared_directory, shared_bytes=self.shared_bytes)
            future = self.executor.submit(work, request)
        self.pending[name] = settings, future
        self.peak = max(self.peak, len(self.pending))
        if self.journal and not completed:
            self.journal.task(name, settings, 'running' if future else 'queued')
        if not restored:
            self.event_record('dispatch', name)

    def harvest(self, *, stopping=False):
        now = time.monotonic()
        if self.progress and now - self.last_progress >= 5 and self.pending:
            self.last_progress = now
            self.progress('workers', f"running/queued={len(self.pending)} limit={self.execution.trials}", None)
        for name, (settings, future) in self.pending.items():
            if future is not None and future.done() and name not in self.ready:
                try:
                    result = future.result()
                except BaseException:
                    if stopping:
                        continue
                    raise
                self.ready[name] = result
                if self.journal:
                    self.journal.save_raw_result(name, settings, result)

    def retire(self, name, *, restored=False):
        settings, future = self.pending[name]
        while future is not None and not future.done():
            self.harvest()
            wait((future,), timeout=.2)
        self.harvest()
        if not restored:
            self.event_record('retire', name)
        del self.pending[name]
        return name, settings

    def entries(self, source):
        source = iter(source)
        # Reconstruct exactly the original dispatch/retirement order, including
        # async completion order. Unfinished requests are evaluated again.
        for event in self.history:
            if event['kind'] == 'dispatch':
                name, settings = next(source)
                if name != event['name']:
                    raise ValueError('Checkpoint dispatch reconstruction diverged.')
                self.dispatch(name, settings, restored=True)
            elif event['kind'] == 'retire':
                yield self.retire(event['name'], restored=True)
            else:
                raise ValueError('Invalid checkpoint schedule event.')
        exhausted = False
        while self.pending or not exhausted:
            fill = self.execution.scheduling == 'async' or not self.pending
            if fill:
                while not exhausted and len(self.pending) < self.execution.trials:
                    try:
                        name, settings = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    self.dispatch(name, settings)
            if not self.pending:
                break
            name = next(iter(self.pending))
            if self.execution.scheduling == 'async':
                while True:
                    self.harvest()
                    available = [n for n, (_, f) in self.pending.items() if f is None or f.done()]
                    if available:
                        name = available[0]
                        break
                    wait([f for _, f in self.pending.values()], timeout=.2, return_when=FIRST_COMPLETED)
            yield self.retire(name)

    def __exit__(self, *error):
        try:
            # Persist completed-but-not-yet-applied results before cancelling.
            self.harvest(stopping=bool(error[0]))
        finally:
            self.event.set()
            for name, (settings, future) in self.pending.items():
                if future:
                    future.cancel()
                if self.journal:
                    self.journal.task(name, settings, 'interrupted')
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
