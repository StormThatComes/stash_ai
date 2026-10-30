"""Solver-wide admission limits, independent of recognition policy/YAML."""
from .resource_errors import ResourceLimit
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import os

_active = ContextVar('solver_execution', default=None)


@dataclass(frozen=True)
class Execution:
    cpus: int
    trials: int
    inner_workers: int
    memory_bytes: int
    trial_bytes: int
    profile: bool = False
    scheduling: str = 'batch'

    def document(self):
        return asdict(self)


def plan(config, options=None):
    options = dict(options or {})
    allowed = {'cpus', 'trials', 'memory_bytes', 'profile', 'scheduling'}
    if options.keys() - allowed:
        raise ValueError('Unknown solver execution option.')
    available = max(1, getattr(os, 'process_cpu_count', os.cpu_count)() or 1)
    requested = options.get('cpus', min(16, available))
    trials = options.get('trials', 1)
    for name, value, high in (('cpus', requested, 16), ('trials', trials, 16)):
        if type(value) is not int or not 1 <= value <= high:
            raise ValueError(f'Solver {name} must be an integer in 1..{high}.')
    cpus = min(requested, available)
    trials = min(trials, cpus)
    settings = config.data['execution']
    # Include both replay databases and separate reference/plan ceilings.
    # This is admission accounting, not a promise about total process RSS.
    person = config.data['sapiensid']['execution']
    trial_bytes = (2 * settings['rebuild_memory_bytes'] + config.data['clustering']['max_reference_bytes']
                   + person['reference_bytes'] + person['plan_bytes'] + settings['rebuild_queue_bytes'])
    memory = options.get('memory_bytes', trial_bytes * trials)
    if type(memory) is not int or memory < trial_bytes:
        raise ValueError(f'Solver memory_bytes must admit one trial ({trial_bytes} bytes).')
    trials = min(trials, memory // trial_bytes)
    profile = options.get('profile', False)
    scheduling = options.get('scheduling', 'batch')
    if type(profile) is not bool or scheduling not in ('batch', 'async'):
        raise ValueError('Solver profile must be boolean; scheduling must be batch or async.')
    # Reserve one coordinator CPU inside each trial's allocation when possible.
    inner = max(1, cpus // trials - 1)
    return Execution(cpus, trials, inner, memory, trial_bytes, profile, scheduling)


def current():
    return _active.get()


@contextmanager
def scope(value):
    token = _active.set(value)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas') if value is not None else nullcontext():
            yield value
    finally:
        _active.reset(token)


def chunks(items, cost, *, target, maximum):
    """Bounded cost-balanced jobs; an oversized item must use a serial path."""
    batch, used = [], 0
    for item in items:
        amount = cost(item)
        if type(amount) is not int or amount < 0 or amount > maximum:
            raise ResourceLimit('Work item exceeds the admitted job budget.')
        if batch and used + amount > min(target, maximum):
            yield batch
            batch, used = [], 0
        batch.append(item)
        used += amount
    if batch:
        yield batch
