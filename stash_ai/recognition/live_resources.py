"""Lightweight measurements of the process and the files owned by this run."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
import shutil
import time

from stash_ai.runtime.memory import process_rss

_active = ContextVar('recognition_resource_sampler', default=None)


def gib(value):
    return 'n/a' if value is None else f'{value / 2**30:.2f} GiB'


class ProcessMeter:
    """Sample process CPU seconds per elapsed second and current RSS."""
    def __init__(self):
        self.lock = Lock()
        self.then, self.cpu = time.monotonic(), time.process_time()
        self.row = {}
        self.scratch = None
        self.trial_budget = None

    def sample(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.then
            if self.row and elapsed < 2:
                return self.row
            cpu = time.process_time()
            row = dict(rss=process_rss(), cores=(cpu - self.cpu)/elapsed if elapsed >= .5 else None,
                       trial_budget=self.trial_budget)
            scratch = self.scratch
            if scratch is not None:
                row.update(scratch_reserved=scratch.used, scratch_limit=scratch.limit, scratch_free=scratch.free)
            self.row = row
            self.then, self.cpu = now, cpu
            return row


def process_line(row, cpus=None):
    cores = row.get('cores')
    cpu = 'warming up' if cores is None else f'{cores:.1f}' + (f'/{cpus}' if cpus else '') + ' cores'
    return f'CPU {cpu} · RSS {gib(row.get("rss"))}'


@contextmanager
def monitor(meter):
    token = _active.set(meter)
    try:
        yield
    finally:
        _active.reset(token)


@contextmanager
def scratch_usage(budget):
    meter = _active.get()
    if meter is None:
        yield
        return
    previous, meter.scratch = meter.scratch, budget
    try:
        yield
    finally:
        meter.scratch = previous


class ConsoleResources:
    def __init__(self, config, *, budget=None, cpus=None):
        from .compute import available_cpus
        self.meter = ProcessMeter()
        self.database = Path(config.database)
        self.budget = budget
        self.cpus = min(cpus or available_cpus(), available_cpus())
        self.disk_checked, self.disk_text = float('-inf'), ''
        self.disk_limit = config.data['execution'].get('disk_total_bytes')
        self.gpu = None

    def disk(self, now):
        if now - self.disk_checked < 10:
            return self.disk_text
        self.disk_checked = now
        volumes = self.budget.volumes if self.budget is not None else {}
        if volumes:
            # Budget replaces the complete snapshot atomically. Its collector
            # already walked the charged files on the coordinator thread.
            dev, row = min(volumes.items(), key=lambda kv: kv[1]['free'] - kv[1]['reserved'])
            self.disk_text = (f'volume {dev}: used {gib(row["used"])} · '
                              f'held {gib(row["reserved"])} · free {gib(row["free"])}')
            if self.disk_limit:
                self.disk_text += f' · cap {gib(self.disk_limit)}'
        else:
            # A report may have no run-wide disk ledger. Scope this value to
            # known files on the source volume; never walk the cache directory.
            try:
                size = sum(file_size(Path(str(self.database) + suffix)) for suffix in ('', '-wal', '-shm'))
                free = shutil.disk_usage(self.database.parent).free
                self.disk_text = f'source DB {gib(size)} · free {gib(free)}'
                if self.disk_limit:
                    self.disk_text += f' · configured cap {gib(self.disk_limit)}'
            except OSError:
                self.disk_text = 'source volume unavailable'
        return self.disk_text

    def fields(self):
        row = self.meter.sample()
        values = dict(process=process_line(row, self.cpus), disk=self.disk(time.monotonic()))
        if self.gpu and self.gpu.display:
            values['vram'] = self.gpu.display
        if 'scratch_reserved' in row:
            values['scratch'] = f'held {gib(row["scratch_reserved"])} / {gib(row["scratch_limit"])}'
        return values


def file_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
