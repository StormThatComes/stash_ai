"""Run-local immutable goal addresses and evaluation-local vote memoization."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_index = ContextVar('merge_goal_index', default=None)
_evaluation = ContextVar('merge_goal_evaluation', default=None)


class GoalIndex:
    def __init__(self, goals):
        self.observation_ids = frozenset(r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations'])
        self.observation_ids |= frozenset(oid for values in goals.get('physical_alternatives', {}).values() for oid in values)
        self.observation_ids |= frozenset(r['id'] for g in goals['groups']
            for m in g['members'] + g.get('excluded_members', []) for r in m.get('excluded_observations', ()))
        self.outcome_ids = self.observation_ids | frozenset(oid for e in goals.get('scene_exclusions', ())
            for oid in e['observation_ids'] + e['target_observation_ids'])
        self.compiled = {}
        self.core_cohorts = {}
        for group in goals['groups']:
            evidence = group.get('identity_evidence')
            if evidence:
                self.cohorts(evidence)
                collections = [evidence['core_units'], evidence['soft_units']]
                collections += [v for scene in evidence['scenes'].values()
                                for k, v in scene.items() if k.endswith('_units')]
                collections += [m['identity_evidence'][key] for m in group['members'] for key in ('core_units', 'soft_units')]
                for units in collections:
                    self.addresses(units)

    def addresses(self, units):
        key = id(units)
        if key not in self.compiled:
            # Keep the list alive: IDs cannot be reused for a different collection.
            self.compiled[key] = units, tuple(tuple((r['id'], r['scene']) for r in u['observations']) for u in units)
        return self.compiled[key][1]

    def cohorts(self, evidence):
        key = id(evidence)
        if key not in self.core_cohorts:
            scenes = defaultdict(list)
            for unit in evidence['core_units']:
                for scene in {r['scene'] for r in unit['observations']}:
                    scenes[scene].append(dict(unit, observations=[r for r in unit['observations'] if r['scene'] == scene]))
            self.core_cohorts[key] = evidence, dict(scenes)
            for units in scenes.values():
                self.addresses(units)
        return self.core_cohorts[key][1]


def current():
    return _index.get()


@contextmanager
def prepared(goals):
    token = _index.set(GoalIndex(goals))
    try:
        yield _index.get()
    finally:
        _index.reset(token)


def solver_scope(function):
    @wraps(function)
    def wrapped(config, goals, *args, **kwargs):
        from .run_cache import scope
        from .solver_execution import plan, scope as execution_scope
        from .solver_progress import heartbeat
        from .trial_metrics import Metrics
        metrics = Metrics(bool((kwargs.get('execution') or {}).get('profile')))
        started_trials = False
        callback = kwargs.get('progress')
        if callback is not None:
            def progress(name, phase, item):
                nonlocal started_trials
                if phase == 'start':started_trials = True
                return callback(name, phase, item)
            progress.telemetry_enabled = getattr(callback, 'telemetry_enabled', True)
            kwargs['progress'] = progress
        try:
            with heartbeat('initialization', kwargs.get('progress'), 'Plan solver resources'), metrics.stage('Plan solver resources'):
                execution = plan(config, kwargs["execution"], source=args[0] if args else kwargs.get('db'),
                                 scope=goals['scope'], namespace=goals['namespace']) if kwargs.get("execution") is not None else None
            from .solver_journal_events import journal_scope
            with metrics.stage('Solve and finalize journal'):
                with journal_scope(), execution_scope(execution), scope(config), prepared(goals):
                    result = function(config, goals, *args, **kwargs)
        except KeyboardInterrupt:
            if started_trials:
                raise  # Replay owns interruption once a trial has started.
            result = dict(status='no_completed_trials', interrupted=True,
                selected_trial=None, best_result=None, recommendation=None,
                simulation=dict(format=1, kind='recognition_replay_simulation', source_read_only=True,
                                trials={}, selection=[], interrupted=True),
                search=dict(max_trials=kwargs.get('maximum',500), executed=0, attempted=0,
                            restored_trials=0, failed_trials=0, stop_reason='interrupted',
                            proves_optimality=False, proves_infeasibility=False))
        if metrics.enabled:
            result['performance'] = metrics.snapshot()
        return result
    return wrapped


def evaluation_scope(function):
    @wraps(function)
    def wrapped(goals, observations, *args, **kwargs):
        # Always create a fresh generation, even if a caller reused a dict.
        token = _evaluation.set({})
        try:
            return function(goals, observations, *args, **kwargs)
        finally:
            _evaluation.reset(token)
    return wrapped


def indexed_votes(units, observations, predicted):
    index, memo = current(), _evaluation.get()
    if index is None:
        return None
    counts, scenes = Counter(), defaultdict(Counter)
    for addresses in index.addresses(units):
        key = (addresses, id(observations), id(predicted))
        if memo is not None and key in memo:
            destination, covered = memo[key]
        else:
            destination, covered = None, set()
            conflict = False
            for oid, scene in addresses:
                value = predicted.get(oid) if oid in observations else None
                if value is not None:
                    if destination is not None and value != destination:
                        conflict = True
                        break
                    destination = value
                    covered.add(scene)
            if conflict:
                destination, covered = None, ()
            if memo is not None:
                memo[key] = destination, covered
        if destination is not None:
            counts[destination] += 1
            for scene in covered:
                scenes[scene][destination] += 1
    return counts, scenes
