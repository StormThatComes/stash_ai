"""Dictionary contracts at evidence boundaries.

Retained records are discriminated by mode. Optional fields represent legacy
records and derived evidence, rather than a new storage format or defaults.
"""
from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict


class EvidenceMetadata(TypedDict, total=False):
    origin: Required[str]
    inference_signature: str
    frame_input_signature: str
    scene_threshold: float
    marker_threshold: float
    scene_score_floor: float
    marker_score_floor: float
    scene_support_floor: float
    marker_tag_blocklist: list[str]
    analysis_format: int


class CurrentInferenceMetadata(TypedDict):
    origin: Literal['inference']
    analysis_format: Literal[4]
    inference_signature: str
    frame_input_signature: str
    scene_threshold: float
    marker_threshold: float
    marker_score_floor: float
    scene_support_floor: float
    scene_score_floor: NotRequired[float]
    scene_frames_format: NotRequired[Literal[1]]


RetainedMetadata = CurrentInferenceMetadata | EvidenceMetadata


class MarkerBuildOptions(TypedDict):
    algorithm: NotRequired[Literal['temporal', 'historical']]
    policy: dict
    core_exclusions: dict[str, list[str]]
    tag_exclusions: dict[str, list[str]]
    marker_thresholds: dict[str, float]
    settings: dict[str, int | float]


class SceneBuildOptions(TypedDict):
    algorithm: NotRequired[Literal['temporal', 'historical']]
    policy: dict
    max_tags: int
    scene_exclusions: dict[str, list[str]]
    scene_thresholds: dict[str, float]


BuildOptions = MarkerBuildOptions | SceneBuildOptions


class SceneEntry(TypedDict):
    tag: str
    score: float | None
    statistics: NotRequired[dict]


class FrameRow(TypedDict):
    start: float
    end: float
    tags: list[tuple[str, float]]


class SceneContribution(SceneEntry):
    source: str
    meta: EvidenceMetadata
    qualification: NotRequired[str]
    derived: NotRequired[Literal['shared_scene_frames']]
    support_sources: NotRequired[list[str]]
    frame_context: NotRequired[dict[str, list[str]]]
    scene_context: NotRequired[dict[str, list[str]]]
    build_options: NotRequired[BuildOptions]


class MarkerContribution(TypedDict):
    tag: str
    source: str
    start: float
    end: float | None
    score: float | None
    meta: EvidenceMetadata
    hits: NotRequired[int]
    extra_tags: NotRequired[list[str]]
    extra_ids: NotRequired[list[str]]
    support_sources: NotRequired[list[str]]
    build_options: NotRequired[BuildOptions]
    derived: NotRequired[Literal['shared_frames', 'occurrence']]
    occurrence_evidence: NotRequired[list[dict]]
    scene_evidence: NotRequired[list[SceneContribution]]


class SceneRecord(TypedDict):
    mode: Literal['scene']
    meta: RetainedMetadata
    items: list[SceneEntry]
    frames: NotRequired[list[FrameRow]]


class FrameRecord(TypedDict):
    mode: Literal['frames']
    meta: RetainedMetadata
    items: list[FrameRow]


class SpanRecord(TypedDict):
    mode: Literal['spans']
    meta: EvidenceMetadata
    items: list[dict]


RetainedRecord = SceneRecord | FrameRecord | SpanRecord
RecordsBySource = dict[str, list[RetainedRecord]]


class MergedMarker(TypedDict):
    tag: str
    start: float
    end: float | None
    sources: list[str]
    contributors: list[MarkerContribution]
    extra_tags: list[str]
    extra_ids: list[str]
    extra_evidence: NotRequired[dict]
