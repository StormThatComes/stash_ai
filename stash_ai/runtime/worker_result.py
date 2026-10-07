"""Pickleable worker envelope, identical with or without profiling."""
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class WorkerResult(Generic[T]):
    scene_id: str
    payload: T | None = None
    error: str | None = None
    metrics: list[dict] | None = None
