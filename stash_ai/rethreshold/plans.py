"""Explicit record access, validation policy and executable threshold plans."""
from dataclasses import dataclass
from enum import Enum

from stash_ai.evidence.cache_views import scene_threshold
from stash_ai.evidence.thresholds import marker_threshold


class RecordAccess(Enum):
    BATCH = 'batch'
    METADATA = 'metadata'

    def read(self, store, scene, kind, model):
        if self is RecordAccess.METADATA:
            return store.result_metadata(scene, kind, model)
        return store.model_record(scene, kind, model, borrow=True)


class PendingPolicy(Enum):
    VERIFY = 'verify'
    CHECKED_BY_EXECUTOR = 'checked_by_executor'


@dataclass(frozen=True)
class ThresholdRequest:
    kinds: tuple[str, ...] = ('scene',)
    access: RecordAccess = RecordAccess.BATCH
    pending: PendingPolicy = PendingPolicy.VERIFY
    marker_check_scene: bool | None = None

    def __post_init__(self):
        if not self.kinds or set(self.kinds) - {'scene', 'markers'}:
            raise ValueError('Unknown threshold output kinds.')


@dataclass
class ThresholdPlan:
    outputs: dict
    access: RecordAccess

    def __bool__(self):
        return bool(self.outputs)


def apply_plan(store, scene, model, plan: ThresholdPlan):
    """Apply a validated plan without reopening payloads or changing their bytes."""
    with store.connection:
        write_plan(store, scene, model, plan)


def write_plan(store, scene, model, plan):
    """The caller owns the transaction, including bounded multi-pair application."""
    for kind, record in plan.outputs.items():
        threshold = scene_threshold(record) if kind == 'scene' else marker_threshold(record)
        store.set_result_threshold(scene, model, kind, threshold, record,
                                   metadata_only=plan.access is RecordAccess.METADATA)
