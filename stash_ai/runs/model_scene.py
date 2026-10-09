"""Typed preparation outcomes and exclusive ownership of staged scene resources."""
from dataclasses import dataclass
from enum import Enum
from typing import Any

from stash_ai.runtime.cleanup import cleanup


@dataclass
class ModelScene:
    scene: str
    paths: dict
    cues: list
    media: Any
    signature: str
    input_options: Any = None
    verified_input: Any = None
    checkpoint: Any = None
    state: Any = None
    batches: Any = None
    prepared: Any = None
    batch_usage: Any = None  # Read-only status accessor; never owns stream cleanup.

    def close(self):
        resources = {id(value): value for value in (self.batches, self.prepared) if value is not None}
        self.batches = self.prepared = None
        self.batch_usage = None
        cleanup([value.close for value in resources.values()])


class PreparationStatus(Enum):
    REUSED = 'reused'
    BLOCKED = 'blocked'
    READY = 'ready'
    FAILED = 'failed'


@dataclass(frozen=True)
class ScenePreparation:
    status: PreparationStatus
    job: ModelScene | None = None
    error: BaseException | None = None

    def __post_init__(self):
        if (self.status is PreparationStatus.READY) != (self.job is not None):
            raise ValueError('Only a ready preparation owns a scene job.')
        if (self.status is PreparationStatus.FAILED) != (self.error is not None):
            raise ValueError('Only a failed preparation owns an exception.')


@dataclass(frozen=True)
class StagedScene:
    scene: str
    preparation: ScenePreparation
    observer: Any
