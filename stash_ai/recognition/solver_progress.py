"""Bounded, best-effort worker telemetry; only the coordinator writes output."""
from contextlib import AbstractContextManager, contextmanager, nullcontext
from queue import Full
from threading import Event, Lock, Thread
import os
import time


class ReplayProgress:
    """Snapshots describe observed work, never an invented whole-trial percentage."""
    def __init__(self, name, callback):
        self.name, self.callback = name, callback
        self.started = time.monotonic()
        self.stages, self.rows = [], {}
        self.last_completed = None

    def update(self, kind, value):
        now = time.monotonic()
        if kind == 'stage':
            self.stages.append(dict(value, started=now))
        elif kind == 'stage_end':
            self.stages.pop()
        elif kind == 'work':
            previous = self.rows.get(value['row'])
            if previous is not None and value['completed'] > previous['completed']:
                self.last_completed = now
            self.rows[value['row']] = value
        elif kind == 'work_end':
            self.rows.pop(value['row'], None)
        stage = self.stages[-1] if self.stages else dict(label='Finalize trial', started=now)
        # Nested rows at the same position temporarily replace their parent.
        rows = {r['position']: r for r in self.rows.values()}
        snapshot = dict(pid=os.getpid(), started=self.started, updated=now,
            last_completed=self.last_completed, stage=stage['label'], stage_started=stage['started'],
            scene=next((s['scene'] for s in reversed(self.stages) if 'scene' in s), None),
            work=[rows[p] for p in sorted(rows)][:2])
        self.callback(self.name, 'stage: ' + stage['label'], snapshot)


class LiveProgress(AbstractContextManager):
    """Publish the latest snapshot once a second, even during blocking SQL/native work.

    Repeated publication does not advance `updated` or `last_completed`. A full
    transport drops telemetry, never stalls a trial. The next tick retries the
    latest snapshot rather than building an unbounded backlog of stage events.
    """
    def __init__(self, name, queue):
        self.name, self.queue = name, queue
        self.lock, self.stop = Lock(), Event()
        now = time.monotonic()
        self.snapshot = dict(pid=os.getpid(), started=now, updated=now,
            last_completed=None, stage='Initialize trial', stage_started=now, scene=None, work=[])

    def __enter__(self):
        self.publish()
        self.thread = Thread(target=self.run, name='solver-telemetry', daemon=True)
        self.thread.start()
        return self

    def __call__(self, name, stage, snapshot):
        with self.lock:
            self.snapshot = snapshot

    def publish(self):
        with self.lock:
            snapshot = self.snapshot
        try:
            self.queue.put_nowait((self.name, dict(snapshot, reported=time.monotonic())))
        except (Full, OSError, ValueError):
            # Observability must not make cancellation or worker exit wait on IPC.
            pass

    def run(self):
        while not self.stop.wait(1.):
            self.publish()

    def __exit__(self, *error):
        self.stop.set()
        self.thread.join()
        self.publish()


@contextmanager
def heartbeat(name, callback, label='Initialize trial'):
    """Give serial/native work the same liveness contract as spawned trials."""
    if callback is None or not getattr(callback, 'telemetry_enabled', True):
        yield None
        return
    if isinstance(callback, LiveProgress):
        yield callback  # A worker already owns its periodic publisher.
        return
    class Output:
        def put_nowait(self, event):
            trial, snapshot = event
            callback(trial, 'stage: ' + snapshot['stage'], snapshot)
    class SerialProgress(LiveProgress):
        def __call__(self, name, stage, snapshot):
            super().__call__(name, stage, snapshot)
            self.publish()  # Preserve short stage/error locations between ticks.
    live = SerialProgress(name, Output())
    live.snapshot['stage'] = label
    with live:
        yield live


def startup(args):
    if getattr(args, 'progress_style', 'off') == 'off':
        return nullcontext()
    def report(name, stage, item):
        import sys
        print('Merge report initialization: '+worker_text(item, time.monotonic()), file=sys.stderr, flush=True)
    return heartbeat('initialization', report, 'Read snapshot and resolve goals')


def worker_text(snapshot, now):
    text = (f"pid={snapshot['pid']} elapsed={now-snapshot['started']:.0f}s "
            f"stage={snapshot['stage']} stage_elapsed={now-snapshot['stage_started']:.0f}s "
            f"last_update={now-snapshot['updated']:.0f}s ago")
    if 'reported' in snapshot:
        text += f" last_report={now-snapshot['reported']:.0f}s ago"
    if snapshot.get('scene') is not None:
        text += f" scene={snapshot['scene']}"
    for row in snapshot['work']:
        total = row['total'] if row['total'] is not None else '?'
        text += f" | {row['label']}={row['completed']}/{total} {row['unit']}"
    completed = snapshot['last_completed']
    text += ' | last_completed=' + ('none observed' if completed is None else f'{now-completed:.0f}s ago')
    return text
