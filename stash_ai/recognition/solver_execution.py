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
    shared_input_bytes: int = 0

    def document(self):
        return asdict(self)


def plan(config, options=None, *, source=None, scope=None, namespace=None):
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
    # Trial history and the inner production scene replay both have hard RAM
    # page ceilings. Immutable source evidence lives on shared disk separately.
    # This is admission accounting, not a promise about total process RSS.
    person = config.data['sapiensid']['execution']
    shared = 0
    face_bound = config.data['clustering']['max_reference_bytes']
    body_bound = person['reference_bytes']
    if source is not None:
        rows = 0
        for table in ('observations', 'body_observations'):
            if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                amount, count = source.execute(f'SELECT COALESCE(SUM(COALESCE(length(embedding),0)+length(quality)+length(geometry)),0),COUNT(*) '
                    f'FROM {table} WHERE scan IN (SELECT id FROM scans WHERE scope=? AND namespace=?)', (scope, namespace)).fetchone()
                shared += amount; rows += count
        if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_evidence'").fetchone():
            amount, count = source.execute('SELECT COALESCE(SUM(COALESCE(length(embedding),0)+length(metadata)),0),COUNT(*) '
                'FROM person_evidence WHERE observation IN (SELECT o.id FROM observations o JOIN scans s ON s.id=o.scan '
                'WHERE s.scope=? AND s.namespace=?)', (scope, namespace)).fetchone()
            shared += amount; rows += count
        # Bound materialized vectors, metadata and Python wrappers conservatively.
        evidence_bound = 16*shared + 4096*rows + 4*1024**2
        face_bound = min(face_bound, evidence_bound)
        body_bound = min(body_bound, evidence_bound)
    trial_bytes = (2 * settings['rebuild_memory_bytes'] + face_bound + body_bound
                   + person['plan_bytes'] + settings['rebuild_queue_bytes'])
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
    return Execution(cpus, trials, inner, memory, trial_bytes, profile, scheduling, shared)


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
