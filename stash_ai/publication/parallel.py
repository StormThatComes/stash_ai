"""Parallel read-only merge evidence preparation.

Workers own isolated read-only StateStore instances. The parent process keeps
all Stash access, tag resolution, publication intent and SQLite writes serial.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import batched
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from stash_ai.runtime.workers import open_read_worker
from stash_ai.runtime.process_pool import read_process_pool
from stash_ai.runtime.settings import evidence_settings as runtime_settings
from stash_ai.runtime.worker_result import pack_chunk, scene_selection, chunk_size, validated_chunks


_WORKER_CONTEXT = None
_PROFILE = False
_TELEMETRY = None


@dataclass(frozen=True)
class MergeEvidence:
    scene_hits: list
    spans: list
    markers: list
    diagnostics: Counter

def _init_worker(state_db, scope, settings, core_policies, scene_policies,
                 cache_bytes, memory_budget_bytes, profile=False, telemetry=None):
    global _WORKER_CONTEXT, _PROFILE, _TELEMETRY
    _PROFILE = profile
    _TELEMETRY = telemetry
    if _WORKER_CONTEXT is not None:
        _WORKER_CONTEXT.close()
    _WORKER_CONTEXT = open_read_worker(
        state_db, scope, settings, core_policies, scene_policies,
        cache_bytes, memory_budget_bytes,
        # Parent changes publication state, never inference evidence.
        allow_parent_publication_updates=True,
    )


def _worker_evidence(scene_id):
    from stash_ai.runtime.profile import execute_worker
    store = _WORKER_CONTEXT.store if _WORKER_CONTEXT is not None else None
    return execute_worker(scene_id, store, lambda: _compute_evidence(scene_id),
                          'merge_worker_compute', profile=_PROFILE)


def _worker_evidence_chunk(scenes):
    if _TELEMETRY is None:
        return pack_chunk(_worker_evidence(scene) for scene in scenes)
    index, scenes = scenes
    _TELEMETRY.started(index)
    def computed():
        for scene in scenes:
            result = _worker_evidence(scene)
            _TELEMETRY.scene_finished(failed=result.error is not None)
            yield result
    try:
        return pack_chunk(computed())
    finally:
        _TELEMETRY.finished(index)


def _compute_evidence(scene_id):
    """Compute evidence; the outer envelope captures failures after profiling."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError('Merge worker was not initialized.')
    from stash_ai.evidence import pipeline as merge

    diagnostics = Counter()
    scene_hits, spans = merge.evidence(
        _WORKER_CONTEXT.settings, _WORKER_CONTEXT.store, str(scene_id),
        _WORKER_CONTEXT.core_policies, _WORKER_CONTEXT.scene_policies, diagnostics)
    from stash_ai.evidence.intervals import union_intervals
    markers = union_intervals(spans, borrow=True) if _WORKER_CONTEXT.settings.CREATE_MARKERS else []
    return MergeEvidence(scene_hits, spans, markers, diagnostics)


def iter_parallel_evidence(app, state_db, scope, scene_ids, core_policies,
                           scene_policies, *, workers, cache_bytes,
                           memory_budget_bytes, profile=False, telemetry=None):
    """Yield read-only evidence in input order while a single parent publishes it."""
    scene_ids = scene_selection(scene_ids)
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
    size = chunk_size(len(scene_ids), workers)
    def scheduled():
        for index, group in enumerate(batched(scene_ids, size)):
            telemetry.queued(index)
            yield index, group

    with read_process_pool(ProcessPoolExecutor,
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
            profile,
        ) + ((telemetry,) if telemetry is not None else ()),
    ) as executor:
        chunks = executor.map(
            _worker_evidence_chunk,
            scheduled() if telemetry is not None else batched(scene_ids, size),
            chunksize=1,
            buffersize=max(workers, 2),
        )
        for number, chunk in enumerate(validated_chunks(scene_ids, chunks, size)):
            if telemetry is not None:
                telemetry.received(number)
            for index, result in enumerate(chunk.results):
                # The serial publisher ingests this aggregate once. Evidence
                # and diagnostics stay attached to their original scene.
                yield replace(result, metrics=chunk.metrics if index == 0 else None)
