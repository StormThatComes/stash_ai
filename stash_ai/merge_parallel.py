"""Parallel read-only merge evidence preparation.

Workers own isolated read-only StateStore instances. The parent process keeps
all Stash access, tag resolution, publication intent and SQLite writes serial.
"""
from __future__ import annotations

import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from stash_ai.state import StateStore


_WORKER_APP = None
_WORKER_STORE = None
_WORKER_CORE_POLICIES = None
_WORKER_SCENE_POLICIES = None

_RUNTIME_SETTINGS = (
    "APPLY_SCENE_TAGS",
    "CREATE_MARKERS",
    "MAX_TAGS",
    "MARKER_MIN_HITS",
    "MARKER_MERGE_GAP_SECONDS",
    "MARKER_MAX_TAGS_PER_FRAME",
    "MAX_MARKERS_PER_SCENE",
    "MARKER_TAG_BLOCKLIST",
)


def runtime_settings(app):
    """Capture mutable merge settings needed by spawned worker interpreters."""
    values = {name: getattr(app, name) for name in _RUNTIME_SETTINGS}
    values["TEMPORAL_POLICY"] = app.TEMPORAL_POLICY
    for name in ("MARKER_THRESHOLDS", "SCENE_THRESHOLDS"):
        if hasattr(app, name):
            values[name] = getattr(app, name)
    return values


def _init_worker(state_db, scope, settings, core_policies, scene_policies,
                 cache_bytes, memory_budget_bytes):
    global _WORKER_APP, _WORKER_STORE, _WORKER_CORE_POLICIES, _WORKER_SCENE_POLICIES
    from stash_ai import cli as app

    for name, value in settings.items():
        setattr(app, name, value)
    store = StateStore(
        state_db,
        scope,
        True,
        cache_bytes=cache_bytes,
        memory_budget_bytes=memory_budget_bytes,
        allow_cleanup=True,
    )
    atexit.register(store.close)
    _WORKER_APP = app
    _WORKER_STORE = store
    _WORKER_CORE_POLICIES = core_policies
    _WORKER_SCENE_POLICIES = scene_policies


def _worker_evidence(scene_id):
    """Return one scene's merge evidence or a serializable failure."""
    if _WORKER_APP is None or _WORKER_STORE is None:
        return str(scene_id), None, None, None, "Merge worker was not initialized."
    from stash_ai import merge

    diagnostics = Counter()
    try:
        scene_hits, spans = merge.evidence(
            _WORKER_APP,
            _WORKER_STORE,
            str(scene_id),
            _WORKER_CORE_POLICIES,
            _WORKER_SCENE_POLICIES,
            diagnostics,
        )
        return str(scene_id), scene_hits, spans, diagnostics, None
    except Exception as exc:
        return str(scene_id), None, None, None, f"{type(exc).__name__}: {exc}"


def iter_parallel_evidence(app, state_db, scope, scene_ids, core_policies,
                           scene_policies, *, workers, cache_bytes,
                           memory_budget_bytes):
    """Yield read-only evidence in input order while a single parent publishes it."""
    scene_ids = [str(scene_id) for scene_id in scene_ids]
    if not scene_ids:
        return
    workers = max(1, min(int(workers), len(scene_ids)))
    if workers == 1:
        raise ValueError("Parallel evidence iterator requires at least two workers.")

    # Reserve one share for the parent writer. Workers otherwise have independent
    # caches/RSS accounting and could each consume the full configured budget.
    divisor = workers + 1
    worker_cache = max(0, int(cache_bytes) // divisor)
    worker_budget = max(1, int(memory_budget_bytes) // divisor)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(
            str(Path(state_db).expanduser().resolve()),
            scope,
            runtime_settings(app),
            core_policies,
            scene_policies,
            worker_cache,
            worker_budget,
        ),
    ) as executor:
        yield from executor.map(
            _worker_evidence,
            scene_ids,
            chunksize=1,
            buffersize=max(workers, 2),
        )
