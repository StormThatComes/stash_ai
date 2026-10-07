"""Dictionary contracts at evidence boundaries.

Active records contain current inference; contributions also describe derived
publication evidence and geometry retained by offline migration.
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
    analysis_format: int


class CurrentInferenceMetadata(TypedDict):
    origin: Literal['inference']
    analysis_format: Literal[5]
    inference_signature: str
    frame_input_signature: str
    scene_threshold: float
    marker_threshold: float
    marker_score_floor: float
    scene_support_floor: float
    scene_score_floor: NotRequired[float]



class MarkerBuildOptions(TypedDict):
    policy: dict
    core_exclusions: dict[str, list[str]]
    tag_exclusions: dict[str, list[str]]
    marker_thresholds: dict[str, float]
    settings: dict[str, int | float]


class SceneBuildOptions(TypedDict):
    policy: dict
    scene_exclusions: dict[str, list[str]]
    scene_thresholds: dict[str, float]


BuildOptions = MarkerBuildOptions | SceneBuildOptions


class SceneEntry(TypedDict):
    tag: str
    score: float | None
    statistics: dict | None


class FrameRow(TypedDict):
    start: float
    end: float
    tags: list[tuple[str, float]]


class BaseSceneContribution(TypedDict):
    tag: str
    score: float | None
    source: str
    meta: EvidenceMetadata
    qualification: NotRequired[str]


class SourceSceneContribution(BaseSceneContribution):
    derived: NotRequired[None]
    statistics: dict | None
    build_options: NotRequired[SceneBuildOptions]


class SharedSceneContribution(BaseSceneContribution):
    derived: Literal['shared_scene_frames']
    support_sources: list[str]
    scene_context: dict[str, str]
    source_evidence: dict[str, dict]
    joint_statistics: dict
    build_options: SceneBuildOptions


SceneContribution = SourceSceneContribution | SharedSceneContribution


class BaseMarkerContribution(TypedDict):
    tag: str
    source: str
    start: float
    end: float | None
    score: float | None
    meta: EvidenceMetadata
    hits: NotRequired[int]
    extra_tags: NotRequired[list[str]]
    support_sources: NotRequired[list[str]]
    build_options: NotRequired[BuildOptions]
    extra_support: NotRequired[dict]
    evidence_signature: NotRequired[str]
    frame_context: NotRequired[dict[str, str]]
    support_context: NotRequired[dict[str, str]]


class SourceMarkerContribution(BaseMarkerContribution):
    derived: NotRequired[None]


class SharedMarkerContribution(BaseMarkerContribution):
    derived: Literal['shared_frames']
    source_evidence: dict[str, dict]
    shared_hits: int
    shared_strong_hits: int
    support_sources: list[str]
    support_context: dict[str, str]
    frame_context: dict[str, str]
    evidence_signature: str
    build_options: MarkerBuildOptions


class OccurrenceContribution(BaseMarkerContribution):
    derived: Literal['occurrence']
    occurrence_evidence: list[dict]
    scene_evidence: list[SceneContribution]


MarkerContribution = SourceMarkerContribution | SharedMarkerContribution | OccurrenceContribution


class SceneRecord(TypedDict):
    mode: Literal['scene']
    meta: CurrentInferenceMetadata
    items: list[SceneEntry]
    frames: list[FrameRow]


class FrameRecord(TypedDict):
    mode: Literal['frames']
    meta: CurrentInferenceMetadata
    items: list[FrameRow]


InferenceRecord = SceneRecord | FrameRecord
RecordsBySource = dict[str, InferenceRecord]

class MergedMarker(TypedDict):
    tag: str
    start: float
    end: float | None
    sources: list[str]
    contributors: list[MarkerContribution]
    extra_tags: list[str]
    extra_evidence: NotRequired[dict]
