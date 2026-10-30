"""Pickle-safe trial requests and isolated production evaluation."""
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
import sqlite3
import os
import time

from .config import plain


@dataclass(frozen=True)
class Request:
    name: str
    config: object
    seed_path: str
    scope: str
    namespace: str
    installation: str
    scans: list
    observations: dict
    goals: dict
    feasibility: object
    initial_changes: tuple
    excluded: object
    targeted: bool
    visits: dict
    priorities: dict
    execution: object
    shared_directory: str | None = None
    shared_bytes: int = 0


def evaluate(request):
    from .trial_evaluation import evaluate as replay_trial
    from .solver_feedback import collect
    from .merge_solver_feedback import QueryFeedback
    from .merge_goal_index import prepared
    from .hybrid_query_observer import observe_queries
    from .rebuild import rebuild_selected
    from .run_cache import scope as cache_scope
    from .solver_execution import scope as execution_scope
    from .merge_solver_feasibility import resource_limit
    started, cpu = time.perf_counter(), time.process_time()
    from .trial_evaluation import settings_for_replay
    config = settings_for_replay(request.config, request.execution)
    observer = QueryFeedback(request.goals)
    observer.visits, observer.priorities = dict(request.visits), dict(request.priorities)
    collected = None
    def inspect(db, settings):
        nonlocal collected
        collected = collect(db, settings, request.goals, request.feasibility, observer,
                            request.initial_changes, request.excluded, request.targeted)
        return collected[0]
    from .solver_shared import scope as shared_scope
    with execution_scope(request.execution), shared_scope(request.shared_directory, request.shared_bytes), cache_scope(config), prepared(request.goals), observe_queries(observer):
        uri = Path(request.seed_path).resolve().as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as seed:
            try:
                result = replay_trial(request.name, config, seed=seed, source=None,
                    scope=request.scope, namespace=request.namespace, installation=request.installation,
                    scans=request.scans, observations=request.observations, rebuild=rebuild_selected,
                    inspect_trial=inspect)
            except Exception as exc:
                if not resource_limit(exc):
                    raise
                return dict(failed=True, reason='resource_limit', error=str(exc),
                    diagnostics=list(getattr(exc, '__notes__', ())),
                    stage=getattr(exc, 'replay_stage', 'Prepare replay'),
                    elapsed_seconds=time.perf_counter()-started)
    result['feedback'] = collected
    result['elapsed_seconds'] = time.perf_counter()-started
    result['cpu_seconds'] = time.process_time()-cpu
    result['worker_pid'] = os.getpid()
    result['started_monotonic_seconds'] = started
    result['finished_monotonic_seconds'] = time.perf_counter()
    return result
