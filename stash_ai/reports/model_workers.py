"""Read-only process workers for the offline model utility report."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import atexit
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

from stash_ai.models import SOURCES
from stash_ai.reports import model
from stash_ai.state import StateStore

_WORKER_APP = None
_WORKER_STORE = None
_WORKER_CORE_POLICIES = None
_WORKER_SCENE_POLICIES = None


def _init_report_worker(state_db, scope, temporal_policy, core_policies,
                        scene_policies, cache_bytes, memory_budget_bytes,
                        preload_sources):
    """Initialize one isolated read-only worker and keep its DB/cache hot."""
    global _WORKER_APP, _WORKER_STORE, _WORKER_CORE_POLICIES, _WORKER_SCENE_POLICIES
    from stash_ai import cli as app

    app.APPLY_SCENE_TAGS = True
    app.CREATE_MARKERS = True
    app.TEMPORAL_POLICY = temporal_policy
    store = StateStore(
        state_db,
        scope,
        True,
        cache_bytes=cache_bytes,
        memory_budget_bytes=memory_budget_bytes,
        allow_cleanup=True,
    )
    if preload_sources:
        store.preload_sources()
    atexit.register(store.close)
    _WORKER_APP = app
    _WORKER_STORE = store
    _WORKER_CORE_POLICIES = core_policies
    _WORKER_SCENE_POLICIES = scene_policies


def _report_worker_scene(scene):
    if _WORKER_STORE is None or _WORKER_APP is None:
        raise RuntimeError("Model report worker was not initialized.")
    return model.scene_report(
        _WORKER_APP,
        _WORKER_STORE,
        scene,
        _WORKER_CORE_POLICIES,
        _WORKER_SCENE_POLICIES,
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
            app.TEMPORAL_POLICY,
            core_policies,
            scene_policies,
            worker_cache,
            worker_budget,
            preload_sources,
        ),
    ) as executor:
        partials = executor.map(
            _report_worker_scene, scene_ids, chunksize=chunksize)
        for partial in tqdm(
                partials, total=len(scene_ids), desc="Model report", unit="scene"):
            model.merge_scene_report(totals, utility, partial)

    return model.report_payload(totals, utility)
