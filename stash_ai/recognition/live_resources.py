"""Lightweight measurements of the process and the files owned by this run."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import os
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
        self.children = {}
        self.scratch = None
        self.trial_budget = None

    def child_sample(self, pid, row):
        with self.lock:
            previous = self.children.get(pid)
            if previous is None or row['sampled'] >= previous['sampled']:
                self.children[pid] = row

    def release_children(self, pids):
        with self.lock:
            for pid in pids:
                self.children.pop(pid, None)

    def sample(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.then
            if self.row and elapsed < 2:
                return self.row
            cpu = time.process_time()
            row = dict(rss=process_rss(), cores=(cpu - self.cpu)/elapsed if elapsed >= .5 else None,
                       trial_budget=self.trial_budget, sampled=now, pid=os.getpid())
            if self.children:
                row['children'] = dict(self.children)
            scratch = self.scratch
            if scratch is not None:
                row.update(scratch_reserved=scratch.used, scratch_limit=scratch.limit, scratch_free=scratch.free)
            self.row = row
            self.then, self.cpu = now, cpu
            return row


def process_line(row, cpus=None):
    cores = row.get('cores')
    cpu = 'warming up' if cores is None else f'{cores:.1f}' + (f'/{cpus}' if cpus else '') + ' cores'
    line = f'CPU {cpu} · RSS {gib(row.get("rss"))}'
    if row.get('children'):
        line += ' · inner ' + sampled_processes(row['children'].values(), time.monotonic())
    return line


def sampled_processes(rows, now):
    rows = list(rows)
    known = [r['rss'] for r in rows if r.get('rss') is not None]
    active = [r['cores'] for r in rows if now-r['sampled'] <= 5 and r.get('cores') is not None]
    age = max(now-r['sampled'] for r in rows)
    cpu = f'{sum(active):.1f} cores' if active else 'n/a'
    return f'{len(rows)} PIDs · sampled RSS {gib(sum(known) if known else None)} · CPU {cpu} · oldest {age:.0f}s'


def current_meter():
    return _active.get()


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
        self.gpu = self.prefetch = None

    def disk(self, now):
        if now - self.disk_checked < 10:
            return self.disk_text
        self.disk_checked = now
        volumes = self.budget.volumes if self.budget is not None else {}
        if volumes:
            # Budget replaces the complete snapshot atomically. Its collector
            # already walked the charged files on the coordinator thread.
            def margin(row):
                free = row['free'] - row['reserved'] - getattr(self.budget, 'headroom', 0)
                if self.disk_limit:
                    cap = self.disk_limit - row.get('requested', row['used'] + row['reserved'])
                    if cap <= free:
                        return cap, 'cap'
                return free, 'free reserve'
            dev, row = min(volumes.items(), key=lambda kv: margin(kv[1])[0])
            remaining, reason = margin(row)
            age = now - getattr(self.budget, 'checked', now)
            self.disk_text = (f'volume {dev}: {reason} margin {gib(remaining)} · sample {max(0,age):.0f}s ago'
                              f' · used {gib(row["used"])} · '
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
        if self.gpu:
            if self.gpu.display:
                values['vram'] = self.gpu.live_display()
            batch = self.gpu.batch_line()
            if batch:
                values['gpu_batch'] = batch
        if self.prefetch and self.prefetch.taken:
            values['prefetch'] = self.prefetch.line()
        if 'scratch_reserved' in row:
            values['scratch'] = f'held {gib(row["scratch_reserved"])} / {gib(row["scratch_limit"])}'
        return values


def file_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
