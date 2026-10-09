"""Shared lifecycle for isolated, read-only evidence workers."""
from dataclasses import dataclass
import atexit

from stash_ai.runtime.settings import RuntimeSettings
from stash_ai.storage.store import StateStore


@dataclass
class ReadWorker:
    settings: RuntimeSettings
    store: StateStore
    core_policies: dict
    scene_policies: dict

    def close(self):
        atexit.unregister(self.store.close)
        self.store.close()


def open_read_worker(state_db, scope, settings, core_policies, scene_policies,
                     cache_bytes, memory_budget_bytes, *, preload_sources=False,
                     allow_parent_publication_updates=False):
    """Create data-only settings and own the worker's store until process exit."""
    runtime = RuntimeSettings(**dict(settings, STASH_API_KEY=""))
    store = StateStore(
        state_db, scope, True, cache_bytes=cache_bytes,
        memory_budget_bytes=memory_budget_bytes, allow_cleanup=True,
        allow_parent_publication_updates=allow_parent_publication_updates,
    )
    try:
        if preload_sources:
            store.preload_sources(None if preload_sources is True else preload_sources)
    except BaseException:
        store.close()
        raise
    atexit.register(store.close)
    return ReadWorker(runtime, store, core_policies, scene_policies)
