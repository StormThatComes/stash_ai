"""Pickleable worker envelope, identical with or without profiling."""
from dataclasses import dataclass, replace
from itertools import batched, zip_longest
from typing import Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class WorkerResult(Generic[T]):
    scene_id: str
    payload: T | None = None
    error: str | None = None
    metrics: list[dict] | None = None


@dataclass(frozen=True)
class WorkerChunk(Generic[T]):
    results: tuple[WorkerResult, ...]
    metrics: list[dict] | None = None
    payload: T | None = None


def pack_chunk(results, *, payload=None, keep_payloads=True):
    """Combine bounded metric histograms once, preserving every scene outcome."""
    from .profile import ProfileMetrics
    results = tuple(results)
    metrics = ProfileMetrics() if any(row.metrics is not None for row in results) else None
    if metrics is not None:
        for row in results:
            if row.metrics is not None:
                metrics.ingest(row.metrics)
    return WorkerChunk(tuple(replace(row, metrics=None,
                                    payload=row.payload if keep_payloads else None) for row in results),
                       metrics.export() if metrics is not None else None, payload)


def scene_selection(scene_ids):
    scenes = tuple(map(str, scene_ids))
    if len(set(scenes)) != len(scenes):
        raise RuntimeError('Worker selection contains duplicate scene IDs.')
    return scenes


def chunk_size(total, workers):
    return max(1, min(8, total // (workers * 4) or 1))


def validated_chunks(scenes, chunks, size):
    """Reject missing, extra, reordered or duplicate scenes before ingestion."""
    missing = object()
    for expected, chunk in zip_longest(batched(scenes, size), chunks, fillvalue=missing):
        if expected is missing or chunk is missing:
            raise RuntimeError('Worker returned an incomplete or extra scene chunk.')
        if (not isinstance(chunk, WorkerChunk)
                or not all(isinstance(row, WorkerResult) for row in chunk.results)
                or tuple(row.scene_id for row in chunk.results) != expected):
            raise RuntimeError(f'Worker returned inconsistent scene IDs for chunk {expected}.')
        yield chunk
