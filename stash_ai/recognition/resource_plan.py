"""One admission plan for rebuild cache and queued numerical scratch.

SQLite keeps its existing half-memory ceiling. Reference banks and Python
metadata have separate budgets; this is not an OS-wide RSS guarantee.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    working: int
    queue: int
    cache: int


def plan(config):
    settings=config.data['execution']
    working=max(0,settings['rebuild_memory_bytes']//4)
    queue=min(settings.get('rebuild_queue_bytes',67108864),working)
    cache=min(settings.get('rebuild_cache_bytes',0),max(0,working-queue))
    return Plan(working,queue,cache)
