"""An authorized reuse plan, shared by fetched and non-fetching evidence paths."""
from dataclasses import dataclass
from enum import Enum

from stash_ai.rethreshold.plans import ThresholdPlan
from .identity_types import ProvenanceIdentity


class ReuseAction(Enum):
    INFER = 'infer'
    REUSE = 'cache_reused'
    RETHRESHOLD = 'rethreshold'


@dataclass(frozen=True)
class ReuseDecision:
    action: ReuseAction
    thresholds: ThresholdPlan | None = None
    frame_input: str | None = None
    input_verified: bool = False
    reasons: tuple[str, ...] = ()
    identities: tuple[tuple[str, ProvenanceIdentity], ...] = ()

    @property
    def reusable(self):
        return self.action is not ReuseAction.INFER

    @property
    def rethreshold(self):
        return self.action is ReuseAction.RETHRESHOLD

    @classmethod
    def authorized(cls, thresholds, frame_input, *, input_verified, identities=()):
        return cls(ReuseAction.RETHRESHOLD if thresholds else ReuseAction.REUSE,
                   thresholds, frame_input, input_verified,
                   ('retained_evidence_covers_thresholds' if thresholds else 'matching_completed_identity',), identities)

    @classmethod
    def infer(cls, reason):
        return cls(ReuseAction.INFER, reasons=(reason,))
