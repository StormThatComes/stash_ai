"""Stable machine reasons independent of operator-facing error messages."""
from enum import StrEnum


class Reason(StrEnum):
    MISSING_SCENE = 'missing_scene_results'
    MISSING_SCENE_FRAMES = 'missing_scene_frames'
    SCENE_FLOOR = 'scene_floor_too_high'
    MISSING_MARKER_FRAMES = 'missing_marker_frames'
    MARKER_FLOOR = 'marker_floor_too_high'
    SCENE_THRESHOLD = 'scene_threshold_requires_inference'
    OBSOLETE = 'obsolete_inference'
    PENDING = 'pending_publication'


LABELS = {
    Reason.MISSING_SCENE: 'missing scene results',
    Reason.MISSING_SCENE_FRAMES: 'missing scene frame evidence',
    Reason.SCENE_FLOOR: 'scene cache floor too high',
    Reason.MISSING_MARKER_FRAMES: 'missing marker frames',
    Reason.MARKER_FLOOR: 'marker cache floor too high',
    Reason.SCENE_THRESHOLD: 'scene threshold requires inference',
    Reason.OBSOLETE: 'obsolete inference evidence',
    Reason.PENDING: 'pending publication',
}


class EvidenceError(ValueError):
    def __init__(self, reason_code: Reason, message: str):
        self.reason_code = reason_code
        super().__init__(message)


class PendingPublication(RuntimeError):
    reason_code = Reason.PENDING
