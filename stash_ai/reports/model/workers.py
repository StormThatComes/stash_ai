"""Read-only process workers for the offline model utility report."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

from stash_ai.models import SOURCES
from stash_ai.reports.model import report as model
from .analysis import scene_report, merge_scene_report
from stash_ai.storage.store import StateStore
from stash_ai.runtime.workers import open_read_worker
from stash_ai.runtime.settings import evidence_settings

_WORKER_CONTEXT = None
_PROFILE = False


def _init_report_worker(state_db, scope, settings, core_policies,
                        scene_policies, cache_bytes, memory_budget_bytes,
                        preload_sources, profile=False):
    """Initialize one isolated read-only worker and keep its DB/cache hot."""
    global _WORKER_CONTEXT, _PROFILE
    _PROFILE = profile
    if _WORKER_CONTEXT is not None:
        _WORKER_CONTEXT.close()
    _WORKER_CONTEXT = open_read_worker(
        state_db, scope,
        settings,
        core_policies, scene_policies, cache_bytes, memory_budget_bytes,
        preload_sources=preload_sources,
    )


def _report_worker_scene(scene):
    from stash_ai.runtime.profile import execute_worker
    store = _WORKER_CONTEXT.store if _WORKER_CONTEXT is not None else None
    return execute_worker(scene, store, lambda: _compute_scene(scene),
                          'model_report_worker_compute', profile=_PROFILE)


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
    scene_ids = list(scene_ids)
    if not scene_ids:
        return model.report_payload(
            Counter(), {source: Counter() for source in SOURCES})

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
                store.preload_sources()
            return model.build_report(
                app, store, scene_ids, core_policies, scene_policies)

    totals = Counter()
    utility = {source: Counter() for source in SOURCES}
    worker_cache = cache_bytes // workers
    worker_budget = max(1, memory_budget_bytes // workers)
    chunksize = max(1, min(64, len(scene_ids) // (workers * 8) or 1))

    with ProcessPoolExecutor(
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
            _report_worker_scene, scene_ids, chunksize=chunksize, buffersize=max(workers, 2))
        for scene, partial in zip(scene_ids, tqdm(
                partials, total=len(scene_ids), desc="Model report", unit="scene")):
            if profile is not None and partial.metrics is not None:
                profile.ingest(partial.metrics)
            if partial.scene_id != str(scene):
                raise RuntimeError('Model report worker returned an inconsistent scene ID.')
            if partial.error is not None:
                raise RuntimeError(partial.error)
            if partial.payload is None:
                raise RuntimeError('Model report worker returned no counters.')
            merge_scene_report(totals, utility, partial.payload)

    return model.report_payload(totals, utility)
