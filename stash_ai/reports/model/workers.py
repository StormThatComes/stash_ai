"""Read-only process workers for the offline model utility report."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from itertools import batched

from tqdm import tqdm

from stash_ai.models import SOURCES
from stash_ai.reports.model import report as model
from .analysis import scene_report, merge_scene_report
from stash_ai.storage.store import StateStore
from stash_ai.runtime.workers import open_read_worker
from stash_ai.runtime.process_pool import read_process_pool
from stash_ai.runtime.settings import evidence_settings
from stash_ai.runtime.worker_result import pack_chunk, scene_selection, chunk_size, validated_chunks

_WORKER_CONTEXT = None
_PROFILE = False
_PRELOAD = False


def _init_report_worker(state_db, scope, settings, core_policies,
                        scene_policies, cache_bytes, memory_budget_bytes,
                        preload_sources, profile=False):
    """Initialize one isolated read-only worker and keep its DB/cache hot."""
    global _WORKER_CONTEXT, _PROFILE, _PRELOAD
    _PROFILE = profile
    _PRELOAD = bool(preload_sources)
    if _WORKER_CONTEXT is not None:
        _WORKER_CONTEXT.close()
    _WORKER_CONTEXT = open_read_worker(
        state_db, scope,
        settings,
        core_policies, scene_policies, cache_bytes, memory_budget_bytes,
        preload_sources=False,
    )


def _report_worker_scene(scene):
    from stash_ai.runtime.profile import execute_worker
    store = _WORKER_CONTEXT.store if _WORKER_CONTEXT is not None else None
    return execute_worker(scene, store, lambda: _compute_scene(scene),
                          'model_report_worker_compute', profile=_PROFILE)


def _report_worker_scenes(scenes):
    """Preload only this bounded task, never the whole scene selection."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError('Model report worker was not initialized.')
    if _PRELOAD:
        _WORKER_CONTEXT.store.preload_sources(scenes)
    return _counter_chunk([_report_worker_scene(scene) for scene in scenes])


def _counter_chunk(results):
    totals, utility = Counter(), {source: Counter() for source in SOURCES}
    for result in results:
        if result.error is None:
            if result.payload is None:
                raise RuntimeError(f'Model report worker returned no counters for scene {result.scene_id}.')
            merge_scene_report(totals, utility, result.payload)
    return pack_chunk(results, payload=(totals, utility), keep_payloads=False)


def _merge_chunk(totals, utility, chunk, profile):
    if profile is not None and chunk.metrics is not None:
        profile.ingest(chunk.metrics)
    for result in chunk.results:
        if result.error is not None:
            raise RuntimeError(f'Scene {result.scene_id}: {result.error}')
    if chunk.payload is None:
        raise RuntimeError('Model report worker returned no counters.')
    merge_scene_report(totals, utility, chunk.payload)


def _compute_scene(scene):
    if _WORKER_CONTEXT is None:
        raise RuntimeError('Model report worker was not initialized.')
    return scene_report(
        _WORKER_CONTEXT.settings,
        _WORKER_CONTEXT.store,
        scene,
        _WORKER_CONTEXT.core_policies,
        _WORKER_CONTEXT.scene_policies,
    )


def build_report_parallel(app, state_db, scope, scene_ids, core_policies,
                          scene_policies, *, workers, cache_bytes,
                          memory_budget_bytes, preload_sources=True, profile=None):
    """Replay independent scenes in isolated CPU worker processes."""
    scene_ids = scene_selection(scene_ids)
    if not scene_ids:
        return model.report_payload(
            Counter(), {source: Counter() for source in SOURCES})

    from stash_ai.db.read_source import consistent_source
    with consistent_source(state_db):
        return _build_report_parallel(app, state_db, scope, scene_ids, core_policies,
            scene_policies, workers=workers, cache_bytes=cache_bytes,
            memory_budget_bytes=memory_budget_bytes, preload_sources=preload_sources, profile=profile)


def _build_report_parallel(app, state_db, scope, scene_ids, core_policies,
                           scene_policies, *, workers, cache_bytes,
                           memory_budget_bytes, preload_sources, profile):
    workers = max(1, min(int(workers), len(scene_ids)))
    if workers == 1:
        with StateStore(
            state_db,
            scope,
            True,
            cache_bytes=cache_bytes,
            memory_budget_bytes=memory_budget_bytes,
            allow_cleanup=True,
        ) as store:
            if preload_sources:
                store.preload_sources(scene_ids)
            if profile is not None:
                # Keep the same measured scene/outcome contract for the
                # single-process fallback; the source remains read-only.
                from stash_ai.runtime.profile import execute_worker
                totals, utility = Counter(), {source: Counter() for source in SOURCES}
                for scene in scene_ids:
                    result = execute_worker(scene, store,
                        lambda: scene_report(app, store, scene, core_policies, scene_policies),
                        'model_report_worker_compute', profile=True)
                    _merge_chunk(totals, utility, _counter_chunk([result]), profile)
                return model.report_payload(totals, utility)
            return model.build_report(
                app, store, scene_ids, core_policies, scene_policies)

    totals = Counter()
    utility = {source: Counter() for source in SOURCES}
    worker_cache = cache_bytes // workers
    worker_budget = max(1, memory_budget_bytes // workers)
    chunksize = chunk_size(len(scene_ids), workers)

    with read_process_pool(ProcessPoolExecutor,
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_report_worker,
        initargs=(
            str(Path(state_db).expanduser().resolve()),
            scope,
            evidence_settings(app),
            core_policies,
            scene_policies,
            worker_cache,
            worker_budget,
            preload_sources,
            profile is not None,
        ),
    ) as executor:
        partials = executor.map(
            _report_worker_scenes, batched(scene_ids, chunksize),
            chunksize=1, buffersize=max(workers, 2))
        with tqdm(total=len(scene_ids), desc="Model report", unit="scene") as progress:
            for chunk in validated_chunks(scene_ids, partials, chunksize):
                _merge_chunk(totals, utility, chunk, profile)
                progress.update(len(chunk.results))

    return model.report_payload(totals, utility)
