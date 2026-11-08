"""Bounded spawned trials; only the caller applies results and writes journals."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import AbstractContextManager, nullcontext
from queue import Empty
import multiprocessing
import signal
import time

_cancel = None
_updates = None
_context = None
_reuse = None


def initialize(event, updates=None, context=None, images=None, slots=None):
    global _cancel, _updates, _context, _reuse
    if _reuse is not None:_reuse.close()
    from .solver_worker_reuse import Reuse
    from multiprocessing.util import Finalize
    _reuse=Reuse()
    Finalize(_reuse,_reuse.close,exitpriority=10)
    from stash_ai.runtime.cancellation import Cancellation
    # Ctrl-C belongs to the coordinator; children receive cooperative cancellation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _cancel = Cancellation()
    _cancel.event = event
    _updates = updates
    from .solver_images import worker
    worker(images)
    _context = context
    if slots is not None:
        from .solver_slot_index import open_maps
        db, observations, baseline = open_maps(slots)
        Finalize(_reuse, db.close, exitpriority=9)
        _context = dict(context, observations=observations, baseline_predictions=baseline)
    if updates is not None:
        # Best-effort telemetry must never delay worker shutdown on a full pipe.
        updates.cancel_join_thread()


def work(request):
    from stash_ai.runtime.cancellation import use
    from .solver_trial import evaluate, Task, restore
    from .solver_progress import LiveProgress
    telemetry = LiveProgress(request.name, _updates) if _updates is not None else nullcontext(None)
    with use(_cancel), telemetry as progress:
        if isinstance(request, Task):
            if progress:
                progress.starting('Restore trial inputs')
            request = restore(request, _context)
        if progress:
            progress.starting('Prepare worker cache')
        reused = _reuse.prepare(request) if _reuse is not None else False
        try:
            result = evaluate(request, progress=progress)
            result['worker_cache_reused'] = reused
            return result
        finally:
            if _reuse is not None:
                _reuse.completed()


class Trials(AbstractContextManager):
    def __init__(self, execution, factory, context, journal=None, progress=None):
        self.execution, self.factory, self.context, self.journal = execution, factory, context, journal
        self.pending, self.ready = OrderedDict(), {}
        self.progress, self.last_progress = progress, 0.
        self.history = list(journal.document.get('schedule', ())) if journal else []
        self.cursor = 0
        self.executor = None
        context_mp = multiprocessing.get_context('spawn')
        self.event = context_mp.Event()
        self.telemetry = progress is not None and getattr(progress, 'telemetry_enabled', True)
        self.updates = context_mp.Queue(maxsize=execution.trials * 4) if self.telemetry else None
        self.snapshots, self.dispatched = {}, {}
        self.process_samples = OrderedDict()
        self.wait_reason = ''
        self.retired = 0
        self.started = time.monotonic()
        self.peak = 0
        from pathlib import Path
        from .solver_shared import Arrays
        self.shared_directory = str(Path(context['seed_path']).parent/'numerical')
        self.shared_bytes = min(execution.memory_bytes // 16, 1024**3)
        if not Arrays.prepare(self.shared_directory):
            self.shared_directory = None
            self.shared_bytes = 0

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
            self.ready[name] = dict(cached, recovered_evaluation=True)
        if not completed and cached is None:
            from dataclasses import replace
            from .solver_trial import split
            request = replace(self.factory(name, settings, self.context),
                shared_directory=self.shared_directory, shared_bytes=self.shared_bytes)
            context, task = split(request)
            if self.executor is None:
                from .solver_images import current
                from .solver_slot_index import build
                images = current()
                slots = None
                if images is not None and context.get('baseline_predictions') is not None:
                    slots = build(images, context['observations'], context['baseline_predictions'])
                if slots is not None:
                    context = {key: value for key, value in context.items()
                               if key not in ('observations', 'baseline_predictions')}
                self.executor = ProcessPoolExecutor(max_workers=self.execution.trials,
                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                    initargs=(self.event, self.updates, context, images, slots))
            future = self.executor.submit(work, task)
        self.pending[name] = settings, future
        self.dispatched[name] = time.monotonic()
        self.peak = max(self.peak, len(self.pending))
        if self.journal and not completed:
            self.journal.task(name, settings, 'running' if future else 'queued')
        if not restored:
            self.event_record('dispatch', name)

    def harvest(self, *, stopping=False):
        self.drain_updates()
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
                if self.telemetry and not stopping:
                    first = next(iter(self.pending))
                    waiting = first if self.execution.scheduling == 'batch' and first != name else None
                    elapsed = result.get('elapsed_seconds', time.monotonic()-self.dispatched[name])
                    state = 'failed' if result.get('failed') else 'finished'
                    detail = f"evaluation={state} elapsed={elapsed:.1f}s awaiting_application"
                    if waiting:
                        detail += f' waiting_for={waiting}'
                    self.progress(name, 'ready', dict(text=detail,
                        compact='ready · waiting for ' + waiting if waiting else 'ready · awaiting application'))
        if not stopping:
            self.report_status()

    def drain_updates(self):
        if self.updates is None:
            return
        # Bounded drain: a busy producer cannot starve scheduling/cancellation.
        for _ in range(self.execution.trials * 4):
            try:
                name, snapshot = self.updates.get_nowait()
            except Empty:
                break
            if snapshot.get('resources') and snapshot.get('pid') is not None:
                pid = snapshot['pid']
                self.process_samples[pid] = (snapshot.get('reported', snapshot['updated']), snapshot['resources'])
                self.process_samples.move_to_end(pid)
                while len(self.process_samples) > self.execution.trials:
                    self.process_samples.popitem(last=False)
            if name in self.pending and name not in self.ready:
                self.snapshots[name] = snapshot

    def report_status(self):
        now = time.monotonic()
        if not self.telemetry or not self.pending or now - self.last_progress < 5:
            return
        from .solver_progress import worker_text
        self.last_progress = now
        ready = {n for n, (_, f) in self.pending.items() if n in self.ready or f is None}
        running = set(self.pending).intersection(self.snapshots) - ready
        waiting = next(iter(self.pending)) if self.execution.scheduling == 'batch' else 'any completed trial'
        idle = max(0, self.execution.trials - len(self.pending))
        occupied = f'Pool {len(running)} running · {len(self.pending)-len(running)-len(ready)} starting · {len(ready)} ready'
        if idle and self.wait_reason:
            occupied += f' · {idle} idle: {self.wait_reason}'
        self.progress('workers', 'pool', dict(compact=occupied, process_samples=dict(self.process_samples), text=(
            f'running={len(running)} queued/starting={len(self.pending)-len(running)-len(ready)} '
            f'ready={len(ready)} retired={self.retired} limit={self.execution.trials} '
            f'elapsed={now-self.started:.0f}s scheduling={self.execution.scheduling} waiting_for={waiting}')))
        for name in self.pending:
            if name in ready:
                text = 'result ready; awaiting application'
            elif name in running:
                text = worker_text(self.snapshots[name], now)
            else:
                text = f'queued/starting elapsed={now-self.dispatched[name]:.0f}s; worker start not yet observed'
            brief = 'queued / starting'
            if name in ready:
                brief = ('ready · waiting for ' + waiting if self.execution.scheduling == 'batch'
                         and name != waiting else 'ready · awaiting application')
            self.progress(name, 'status', dict(text=text, compact=brief,
                          snapshot=self.snapshots.get(name) if name in running else None))

    def retire(self, name, *, restored=False):
        settings, future = self.pending[name]
        while future is not None and not future.done():
            self.harvest()
            wait((future,), timeout=.2)
        self.harvest()
        if not restored:
            self.event_record('retire', name)
        del self.pending[name]
        self.snapshots.pop(name, None)
        self.dispatched.pop(name, None)
        self.retired += 1
        return name, settings

    def entries(self, source):
        source = iter(source)
        # Reconstruct exactly the original dispatch/retirement order, including
        # async completion order. Unfinished requests are evaluated again.
        for event in self.history:
            if event['kind'] == 'dispatch':
                proposal = next(source)
                # Drain barriers do not dispatch a task. During reconstruction
                # the intervening retire events may already have been applied.
                while proposal is None:
                    proposal = next(source)
                name, settings = proposal
                if name != event['name']:
                    raise ValueError('Checkpoint dispatch reconstruction diverged.')
                self.dispatch(name, settings, restored=True)
            elif event['kind'] == 'retire':
                yield self.retire(event['name'], restored=True)
            else:
                raise ValueError('Invalid checkpoint schedule event.')
        exhausted = False
        while self.pending or not exhausted:
            if not exhausted:
                # Completed later trials no longer occupy CPU. Keep their
                # results bounded and preserve canonical batch retirement.
                self.harvest()
                while (not exhausted and len(self.pending) < self.execution.trials * (2 if getattr(self.factory, "independent", False) else 1)
                       and sum(n not in self.ready for n in self.pending) < self.execution.trials):
                    try:
                        proposal = next(source)
                        if proposal is None:
                            if not self.pending:
                                raise ValueError('Solver requested results without pending trials.')
                            self.wait_reason = 'search needs pending results'
                            break
                        self.wait_reason = ''
                        name, settings = proposal
                    except StopIteration:
                        exhausted = True
                        self.wait_reason = 'no further proposals'
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
            elif getattr(self.factory, 'independent', False) and self.pending[name][1] is not None and not self.pending[name][1].done():
                active = [future for _, future in self.pending.values() if future is not None and not future.done()]
                if active:
                    wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                    if not self.pending[name][1].done() and not self.wait_reason and not exhausted:
                        continue
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
            try:
                if self.executor:
                    try:
                        if self.telemetry and self.pending:
                            self.progress('workers', 'pool', dict(text='cancelling pending trials; waiting for worker shutdown'))
                    finally:
                        from .solver_shutdown import shutdown
                        shutdown(self.executor, (f for _,f in self.pending.values()))
            finally:
                if self.updates is not None:
                    self.updates.close()
                    self.updates.join_thread()
