"""Parallel read-only merge evidence preparation.

Workers own isolated read-only StateStore instances. The parent process keeps
all Stash access, tag resolution, publication intent and SQLite writes serial.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from stash_ai.runtime.workers import open_read_worker


_WORKER_CONTEXT = None

_RUNTIME_SETTINGS = (
    "APPLY_SCENE_TAGS",
    "CREATE_MARKERS",
    "MAX_TAGS",
    "MARKER_MIN_HITS",
    "MARKER_MERGE_GAP_SECONDS",
    "MARKER_MAX_TAGS_PER_FRAME",
    "MAX_MARKERS_PER_SCENE",
    "MARKER_THRESHOLDS",
    "SCENE_THRESHOLDS",
)


def runtime_settings(app):
    """Capture mutable merge settings needed by spawned worker interpreters."""
    values = {name: getattr(app, name) for name in _RUNTIME_SETTINGS}
    values["TEMPORAL_POLICY"] = app.TEMPORAL_POLICY
    return values


def _init_worker(state_db, scope, settings, core_policies, scene_policies,
                 cache_bytes, memory_budget_bytes):
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is not None:
        _WORKER_CONTEXT.close()
    _WORKER_CONTEXT = open_read_worker(
        state_db, scope, settings, core_policies, scene_policies,
        cache_bytes, memory_budget_bytes,
        # Parent changes publication state, never inference evidence.
        allow_parent_publication_updates=True,
    )


def _worker_evidence(scene_id):
    """Return one scene's merge evidence or a serializable failure."""
    if _WORKER_CONTEXT is None:
        return str(scene_id), None, None, None, "Merge worker was not initialized."
    from stash_ai.evidence import pipeline as merge

    diagnostics = Counter()
    try:
        scene_hits, spans = merge.evidence(
            _WORKER_CONTEXT.settings,
            _WORKER_CONTEXT.store,
            str(scene_id),
            _WORKER_CONTEXT.core_policies,
            _WORKER_CONTEXT.scene_policies,
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
