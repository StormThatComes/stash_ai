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


def _init_report_worker(state_db, scope, settings, core_policies,
                        scene_policies, cache_bytes, memory_budget_bytes,
                        preload_sources):
    """Initialize one isolated read-only worker and keep its DB/cache hot."""
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is not None:
        _WORKER_CONTEXT.close()
    _WORKER_CONTEXT = open_read_worker(
        state_db, scope,
        settings,
        core_policies, scene_policies, cache_bytes, memory_budget_bytes,
        preload_sources=preload_sources,
    )


def _report_worker_scene(scene):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Model report worker was not initialized.")
    return scene_report(
        _WORKER_CONTEXT.settings,
        _WORKER_CONTEXT.store,
        scene,
        _WORKER_CONTEXT.core_policies,
        _WORKER_CONTEXT.scene_policies,
    )


def build_report_parallel(app, state_db, scope, scene_ids, core_policies,
                          scene_policies, *, workers, cache_bytes,
                          memory_budget_bytes, preload_sources=True):
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
        ),
    ) as executor:
        partials = executor.map(
            _report_worker_scene, scene_ids, chunksize=chunksize, buffersize=max(workers, 2))
        for partial in tqdm(
                partials, total=len(scene_ids), desc="Model report", unit="scene"):
            merge_scene_report(totals, utility, partial)

    return model.report_payload(totals, utility)
