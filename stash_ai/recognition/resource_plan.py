"""Shared admission budget for live replay pages, reservations and caches.

Accounting covers owned work, not the process's complete OS RSS.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    working: int
    queue: int
    cache: int


def memory_limit(config):
    from .solver_execution import current
    runtime = current()
    configured = config.data['execution']['rebuild_memory_bytes']
    admitted = getattr(runtime, 'rebuild_memory_bytes', None)
    return min(configured, admitted) if admitted is not None else configured


def plan(config):
    settings=config.data['execution']
    working=max(0,memory_limit(config))
    queue=min(settings.get('rebuild_queue_bytes',67108864),working)
    cache=min(settings.get('rebuild_cache_bytes',0),max(0,working-queue))
    return Plan(working,queue,cache)
