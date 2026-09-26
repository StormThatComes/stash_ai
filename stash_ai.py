#!/usr/bin/env python
r"""
Stash AI: multi-model scene tags, source evidence and timestamped markers.
Python 3.14.7. Keep the stash_*.py modules and YAML files beside this script.
See README_multimodel.md for configuration, migration and all options.

    python stash_ai.py --models all
    python stash_ai.py --models all --scene-id 123 --restart
    python stash_ai.py --merge-only --scene-id 123 --dry-run
    python stash_ai.py --prune-empty-tags-only --dry-run
    python stash_ai.py --explain-tag ai:blue_hair --scene-id 123

JoyTag, WD, DTQ and OppaiOracle run sequentially, one loaded model at a time.
The default merged mode writes common ai: labels and unions overlapping/touching
intervals with identical labels. Source scores, qualifying frames and original
legacy intervals remain in tagger_sources.sqlite3. Identical VTT inputs can
qualify shared frame runs before short-run rejection, without averaging scores.
Marker extras can come from any compatible model, with separate provenance. There is no
label allowlist or synonym mapping. temporal_policy.yml controls automatic
scene qualification, marker hysteresis and extras from the whole interval.

Matching completed scene/model results are reused by default. Partial checkpoints
resume from the last committed batch. --restart requests fresh inference only
for selected scenes/models/output kinds; omit it when resuming that run.
No image files or full below-threshold score matrices are cached on disk.
Up to four next batches are CPU-prepared in RAM; --prefetch-batches 0 disables it.
--ram-cache-gb budgets retained results/indexes (24 GiB by default, allocated on demand).
Short contiguous extra-tag episodes and shared scene tags reuse saved VTT evidence.
Default progress summarizes the collection; --verbose enables per-scene details.
Visual input comes exclusively from Stash VTT/sprites over HTTP(S).
No video files are opened, streamed or decoded by this script.

Global exclusions cover scene tags, every marker role and hierarchy creation.
Core-only exclusions still allow scene tags and marker extras. --merge-only
reapplies policies to saved evidence without inference. --prune-empty-tags-only deletes
unused leaf tag definitions in the selected AI/model prefixes; --dry-run previews.
Pruning also rechecks newly empty parents and all available Stash usage counters.

--output-mode separate retains per-model prefixes and full inference per run;
cross-model votes and checkpoint resume apply to merged mode. --dry-run changes neither
Stash nor persistent source state, but may download missing model assets.
The source database admits one writer and keeps intent for retrying partial API
writes. Launch the application with stash_ai.py.
"""

from __future__ import annotations

# Re-read the entry point under the installation lock before any runtime imports.
if __name__ == "__main__" and not globals().get("_STASH_INSTALL_LOCKED"):
    from update_stash_ai import launch_application
    raise SystemExit(launch_application(__file__))

import argparse
import hashlib
import math
import os
import re
import sys
from collections import Counter, OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

if not globals().get("_STASH_INSTALL_LOCKED"):
    from update_stash_ai import application_guard
    _installation_lock = application_guard(Path(__file__).resolve().parent)

DEFAULT_OUTPUT_MODE = "merged"
DEFAULT_RESUME = True
CHECKPOINT_BATCHES = 8
PREFETCH_BATCHES = 4  # Current batch plus up to four CPU-prepared batches; 0 disables.
RAM_CACHE_GB = 24.0  # Shared result/index/media cache; allocated on demand.
RAM_BUDGET_GB = 32.0  # Process target; reserves 25% for models and active work.
PREPARE_WORKERS = 4  # CPU transforms only. One model/inference call at a time.
MERGE_STATE_FILE = Path(__file__).resolve().with_name("tagger_sources.sqlite3")

from stash_taggers import (JOYTAG, SOURCES, ModelSource, ModelSettings, Vocabulary,
                           select_models, load_vocabulary, load_tagger, ensure_files, weight_files)
from stash_temporal import TemporalPolicy
from stash_domain import Cue, FramePrediction, MarkerSpec, TagHierarchy
from stash_client import StashClient, absolute_stash_url, safe_error
from stash_tags import TagResolver

DEFAULT_TEMPORAL = True
TEMPORAL_POLICY = TemporalPolicy()
TEMPORAL_POLICY_FILE = Path(__file__).resolve().with_name("temporal_policy.yml")


# ============================== CONFIG ==============================

STASH_URL = os.getenv("STASH_URL", "http://localhost:9999")
STASH_API_KEY = os.getenv("STASH_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiI0cmNoMG4iLCJzdWIiOiJBUElLZXkiLCJpYXQiOjE3ODg4ODI0MDl9.NjmFv7abYBiHnCbp1PwQrk8FYQ7I_A-8VA8PxhXhPog")

# Whole-scene tags: original behavior, independently switchable.
APPLY_SCENE_TAGS = True
JOYTAG_DEFAULT_SCENE_THRESHOLD = 0.40
MAX_TAGS = 0  # 0 = all qualifying scene tags.
REPLACE_SCENE_TAGS = True
AGGREGATION = "topk_mean"  # "max" or "topk_mean"
TOP_K_FRAMES = 2

# Timestamped tags are attached to Stash SceneMarkers.
CREATE_MARKERS = True
DEFAULT_MARKER_THRESHOLD = 0.65
MARKER_MAX_TAGS_PER_FRAME = 0  # 0 = all qualifying primary and additional tags.
# Additional tags use the top N threshold/block-filtered detections (including
# core-excluded labels), with the current primary tag removed from that list.
MARKER_MIN_HITS = 2
MARKER_MERGE_GAP_SECONDS = 0.0  # 0 joins touching cues only; >0 tolerates short gaps.
MAX_MARKERS_PER_SCENE = 0  # 0 = unlimited; otherwise keep highest-confidence runs.
MARKER_USE_END_TIME = True
REPLACE_MARKERS = True  # False retains obsolete markers except excluded cores.

# Optional raw-label blocklist for both primary and additional marker tags.
# Applies to each model; no allowlist is used.
MARKER_TAG_BLOCKLIST: set[str] = set()

# Marker ownership identifiers are defined in stash_taggers.py and stash_merge.py.
MARKER_MATCH_TOLERANCE_SECONDS = 0.25

CREATE_MISSING = True
DRY_RUN = False
SKIP_EXISTING_OUTPUTS = False  # Checked per model and enabled output type.
MAX_FRAMES_PER_SCENE = 0  # 0 = all VTT cues. A cap can miss short events.
JOYTAG_DEFAULT_BATCH_SIZE = 16
SCENE_PAGE_SIZE = 100
REQUEST_TIMEOUT = (5, 600)
HTTP_RETRIES = 3  # Media GETs and read-only GraphQL retry transient failures; GraphQL writes never retry.

JOYTAG_HIERARCHY_FILE = Path(__file__).resolve().with_name("joytag_hierarchy.yml")
MARKER_CORE_EXCLUSIONS_FILE = Path(__file__).resolve().with_name("marker_core_exclusions.yml")
if not MARKER_CORE_EXCLUSIONS_FILE.exists():
    # Existing installations can retain the old filename or pass an explicit path.
    MARKER_CORE_EXCLUSIONS_FILE = MARKER_CORE_EXCLUSIONS_FILE.with_name("joytag_marker_core_exclusions.yml")
SCENE_TAG_EXCLUSIONS_FILE = Path(__file__).resolve().with_name("scene_tag_exclusions.yml")
MODELS_CONFIG_FILE = Path(__file__).resolve().with_name("tagger_models.yml")

JOYTAG_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
JOYTAG_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TIMESTAMP_RE = re.compile(r"(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})")
RECT_RE = re.compile(r"(.*?)#xywh=(?:pixel:)?(\d+),(\d+),(\d+),(\d+)")


def vocabulary_digest(labels: list[str]) -> str:
    """Set fingerprint; the hierarchy never changes the model's label order."""
    return hashlib.sha256(("\n".join(sorted(labels)) + "\n").encode("utf-8")).hexdigest()


def load_unique_yaml(path: Path, context: str) -> Any:
    """Use safe YAML constructors and reject duplicate or non-string keys."""
    import yaml

    class UniqueSafeLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            self.flatten_mapping(node)
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError(f"{context} YAML mapping keys must be strings; quote numeric names.")
                if key in result:
                    raise ValueError(f"Duplicate YAML key: {key!r}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    return yaml.load(path.read_text(encoding="utf-8-sig"), Loader=UniqueSafeLoader)


def load_exclusion_policy(path: Path, key: str) -> dict[str, frozenset[str]]:
    """Common raw labels plus optional per-model additions, never a whitelist."""
    if not path.is_file():
        raise FileNotFoundError(f"Exclusions YAML missing: {path}. Use {key}: [] for an empty blacklist.")
    data = load_unique_yaml(path, "Tag exclusions")
    if key == "excluded_scene_tags" and isinstance(data, dict) and "excluded_tags" in data:
        if key in data:
            raise ValueError("Use excluded_tags or excluded_scene_tags, not both.")
        data = {key if name == "excluded_tags" else name: value for name, value in data.items()}
    if (not isinstance(data, dict) or not {"version", key} <= set(data)
            or set(data) - {"version", key, "by_model"}):
        raise ValueError(f"{path.name} must contain version, {key}, and optionally by_model.")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Exclusions YAML version must be 1.")
    def checked(entries, context):
        if not isinstance(entries, list):
            raise ValueError(f"{context} must be a list of quoted raw model labels; use [] for none.")
        for label in entries:
            if (not isinstance(label, str) or not label or label != label.strip()
                    or "\n" in label or "\r" in label
                    or label.startswith("ai:") or any(label.startswith(source.prefix) for source in SOURCES.values())):
                raise ValueError(f"Invalid excluded tag {label!r}; use a raw label without jt:/wd:/dtq:/op:/ai:.")
        if len(entries) != len(set(entries)):
            raise ValueError(f"Duplicate label in {context}.")
        return frozenset(entries)
    common = checked(data[key], key)
    additions = data.get("by_model", {})
    if not isinstance(additions, dict) or set(additions) - set(SOURCES):
        raise ValueError("by_model must contain only joytag, wd, dtq, and/or oppai keys.")
    return {name: common | checked(additions.get(name, []), f"by_model.{name}") for name in SOURCES}


def load_model_settings(path: Path, chosen: list[str] | None, device: str | None,
                        offline: bool = False) -> list[ModelSettings]:
    data = load_unique_yaml(path, "Model configuration")
    allowed = {"version", "enabled_models", "device", "download_missing", *SOURCES}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Unknown model configuration keys.")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Model configuration version must be 1.")
    selected = select_models(chosen if chosen is not None else data.get("enabled_models", ["joytag"]))
    settings = []
    for name in SOURCES:
        values = data.get(name, {})
        if not isinstance(values, dict) or set(values) - {"model_dir", "scene_threshold", "marker_threshold", "batch_size"}:
            raise ValueError(f"Invalid {name} settings in {path.name}.")
        if name not in selected:
            continue
        def local_path(key: str, default: Path) -> Path:
            environment_name = f"{name.upper()}_MODEL_PATH"
            raw = os.getenv(environment_name)
            if raw is not None:
                return Path(raw).expanduser().resolve()
            if key not in values:
                return default
            raw = values[key]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{name}.{key} must be a non-empty path.")
            value = Path(raw).expanduser()
            return value if value.is_absolute() else (path.resolve().parent / value).resolve()
        model_dir = local_path("model_dir", path.resolve().parent / "models" / name)
        config = ModelSettings(
            source=SOURCES[name], model_dir=model_dir,
            scene_threshold=values.get("scene_threshold", JOYTAG_DEFAULT_SCENE_THRESHOLD if name == "joytag" else
                                       {"wd": 0.35, "dtq": 0.20, "oppai": 0.65}[name]),
            marker_threshold=values.get("marker_threshold", 0.80 if name == "oppai" else DEFAULT_MARKER_THRESHOLD),
            batch_size=values.get("batch_size", JOYTAG_DEFAULT_BATCH_SIZE if name == "joytag" else 4),
            device=device or data.get("device", "auto"),
            download_missing=False if offline else data.get("download_missing", True))
        config.validate()
        settings.append(config)
    return sorted(settings, key=lambda item: selected.index(item.source.key))


def metadata_hierarchy(vocabulary: Vocabulary, source: ModelSource) -> TagHierarchy:
    prefix = source.prefix + "category:"
    root = prefix + source.title
    categories = {root: ()}
    leaves = {}
    for label in vocabulary.labels:
        category = prefix + vocabulary.categories.get(label, "Uncategorized")
        categories[category] = (root,)
        leaves[label] = (category,)
    collisions = set(categories) & ({source.prefix + label for label in vocabulary.labels} | {source.owner_tag})
    if collisions:
        raise ValueError(f"Hierarchy categories collide with predicted/owner tags: {sorted(collisions)!r}")
    return TagHierarchy(leaves, categories, prefix, "replace_managed", source.hierarchy_description)


def load_tag_hierarchy(path: Path, labels: list[str], source: ModelSource = JOYTAG) -> TagHierarchy:
    if not path.is_file():
        raise FileNotFoundError(f"Hierarchy YAML missing: {path}. Place it beside the script or use --no-hierarchy.")
    data = load_unique_yaml(path, "Hierarchy")
    allowed = {"version", "source", "root", "category_prefix", "parent_sync", "groups"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Hierarchy YAML must be a mapping using only: " + ", ".join(sorted(allowed)))
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Hierarchy YAML version must be 1.")
    root, prefix = data.get("root"), data.get("category_prefix")
    if not all(isinstance(value, str) and value.strip() and value == value.strip()
               and "\n" not in value and "\r" not in value for value in (root, prefix)):
        raise ValueError("Hierarchy root and category_prefix must be non-empty strings without surrounding whitespace.")
    if not prefix.startswith(source.prefix):
        raise ValueError(f"Hierarchy category_prefix must start with {source.prefix!r} for {source.key}.")
    mode = data.get("parent_sync", "replace_managed")
    if mode not in {"replace_managed", "add"}:
        raise ValueError("parent_sync must be replace_managed or add.")
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Hierarchy groups must be a non-empty mapping of category paths to label lists.")
    root_name = prefix + root
    categories: dict[str, tuple[str, ...]] = {root_name: ()}
    leaves: dict[str, set[str]] = {}
    canonical_groups: set[str] = set()
    for group_path, group_labels in groups.items():
        parts = [part.strip() for part in group_path.split("/")]
        if not all(part and part not in {".", ".."} and "\n" not in part and "\r" not in part for part in parts):
            raise ValueError(f"Invalid hierarchy category path: {group_path!r}")
        canonical = " / ".join(parts)
        if canonical in canonical_groups:
            raise ValueError(f"Duplicate normalized category path: {canonical!r}")
        canonical_groups.add(canonical)
        if not isinstance(group_labels, list) or any(not isinstance(label, str) or not label for label in group_labels):
            raise ValueError(f"{group_path!r} must contain a list of quoted model labels.")
        if len(group_labels) != len(set(group_labels)):
            raise ValueError(f"Duplicate label within group {group_path!r}.")
        parent = root_name
        for depth in range(1, len(parts) + 1):
            category = prefix + " / ".join(parts[:depth])
            if category == root_name:
                raise ValueError("A category path collides with the hierarchy root.")
            categories[category] = (parent,)
            parent = category
        for label in group_labels:
            leaves.setdefault(label, set()).add(parent)
    label_set = set(labels)
    missing, unknown = label_set - set(leaves), set(leaves) - label_set
    if missing or unknown:
        raise ValueError(
            f"Hierarchy/model vocabulary mismatch: {len(missing)} unmapped labels "
            f"({', '.join(sorted(missing)[:12])}); {len(unknown)} unknown labels "
            f"({', '.join(sorted(unknown)[:12])}). Edit the YAML to match the selected model's vocabulary."
        )
    collisions = set(categories) & ({source.prefix + label for label in labels} | {source.owner_tag})
    if collisions:
        raise ValueError(f"Hierarchy categories collide with model/marker tags: {sorted(collisions)!r}")
    source_metadata = data.get("source", {})
    if not isinstance(source_metadata, dict):
        raise ValueError("Hierarchy source metadata must be a mapping.")
    if source_metadata.get("tag_count", len(labels)) != len(labels):
        raise ValueError("Hierarchy source.tag_count does not match the model vocabulary.")
    if source_metadata.get("vocabulary_sha256", vocabulary_digest(labels)) != vocabulary_digest(labels):
        raise ValueError("Hierarchy source.vocabulary_sha256 does not match the model vocabulary.")
    return TagHierarchy({tag: tuple(sorted(parents)) for tag, parents in leaves.items()}, categories, prefix, mode, source.hierarchy_description)


def timestamp_seconds(value: str) -> float:
    match = TIMESTAMP_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid VTT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError(f"Invalid VTT timestamp: {value!r}")
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def format_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def parse_vtt_cues(vtt_text: str) -> list[Cue]:
    """Keep timing and image identity, including reused rectangles at new times."""
    text = vtt_text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    seen: set[Cue] = set()
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or re.match(r"^(NOTE|STYLE|REGION)(?:\s|$)", lines[0]):
            continue
        timing_lines = [i for i, line in enumerate(lines) if "-->" in line]
        if not timing_lines:
            continue  # WEBVTT header or metadata.
        if len(timing_lines) != 1:
            raise ValueError("Malformed VTT: expected one timing line per cue.")
        i = timing_lines[0]
        timing = re.fullmatch(r"(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[i])
        if not timing:
            raise ValueError("Malformed VTT cue timing.")
        start, end = map(timestamp_seconds, timing.groups())
        if end <= start:
            raise ValueError("VTT cue end must be after its start.")
        payloads = [RECT_RE.fullmatch(line) for line in lines[i + 1:]]
        payloads = [match for match in payloads if match is not None]
        if len(payloads) != 1:
            raise ValueError("VTT cue is missing a single valid sprite rectangle.")
        image, *coordinates = payloads[0].groups()
        rect = tuple(map(int, coordinates))
        if rect[2] == 0 or rect[3] == 0:
            raise ValueError("VTT contains a zero-sized thumbnail.")
        cue = Cue(start, end, image, rect)
        if cue not in seen:
            cues.append(cue)
            seen.add(cue)
    cues.sort(key=lambda cue: (cue.start, cue.end, cue.image, cue.rect))
    if not cues:
        raise ValueError("No timestamped sprite thumbnails in VTT; generate Stash sprites first.")
    if any(b.start < a.end - 0.001 for a, b in zip(cues, cues[1:])):
        raise ValueError("Overlapping VTT thumbnail cues are ambiguous; regenerate the sprite.")
    return cues


def evenly_sample(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def create_stash_client(dry_run: bool) -> StashClient:
    """Bind the current CLI settings once for each client instance."""
    return StashClient(dry_run, url=STASH_URL, api_key=STASH_API_KEY,
                       request_timeout=REQUEST_TIMEOUT, http_retries=HTTP_RETRIES,
                       scene_page_size=SCENE_PAGE_SIZE, create_markers=CREATE_MARKERS)


def fetch_cues(stash: StashClient, paths: dict) -> tuple[list[Cue], int]:
    if not paths.get("vtt"):
        raise ValueError("No VTT path; generate Stash sprites before running scene marker analysis.")
    content = stash.get_media(absolute_stash_url(paths["vtt"], base_url=STASH_URL))
    cues = parse_vtt_cues(content.decode("utf-8-sig"))
    # Resolve before sampling: a sample containing only one sheet from a
    # multi-sheet VTT must not be mistaken for a single-sheet scene.
    single_sheet = len({cue.image for cue in cues}) == 1
    cues = [Cue(cue.start, cue.end,
                absolute_stash_url(paths["sprite"], base_url=STASH_URL) if single_sheet and paths.get("sprite")
                else urljoin(absolute_stash_url(paths["vtt"], base_url=STASH_URL), cue.image), cue.rect)
            for cue in cues]
    return evenly_sample(cues, MAX_FRAMES_PER_SCENE), len(cues)


def iter_sprite_frames(stash: StashClient, paths: dict, cues: list[Cue]) -> Iterator[tuple[Cue, Any]]:
    from PIL import Image

    sheets: OrderedDict[str, Any] = OrderedDict()
    try:
        for cue in cues:
            url = cue.image
            if url not in sheets:
                with Image.open(BytesIO(stash.get_media(url))) as opened:
                    sheets[url] = opened.convert("RGB")
                if len(sheets) > 2:
                    _, old = sheets.popitem(last=False)
                    old.close()
            sheets.move_to_end(url)
            sprite = sheets[url]
            x, y, width, height = cue.rect
            if x + width > sprite.width or y + height > sprite.height:
                raise ValueError(f"VTT rectangle exceeds sprite bounds at {format_time(cue.start)}.")
            frame = sprite.crop((x, y, x + width, y + height))
            try:
                yield cue, frame
            finally:
                frame.close()
    finally:
        for sprite in sheets.values():
            sprite.close()


def prepare_image(image: Any, target_size: int) -> Any:
    import torchvision.transforms.functional as TVF
    from PIL import Image

    side = max(image.size)
    with Image.new("RGB", (side, side), (255, 255, 255)) as padded:
        padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        with padded.resize((target_size, target_size), Image.Resampling.BICUBIC) as resized:
            tensor = TVF.pil_to_tensor(resized).float().div_(255.0)
    return TVF.normalize(tensor, mean=JOYTAG_CLIP_MEAN, std=JOYTAG_CLIP_STD)


def select_scores(scores: list[float], labels: list[str], threshold: float, limit: int) -> list[tuple[str, float]]:
    import numpy as np
    if isinstance(scores, np.ndarray):
        # Match Python float comparisons exactly at the threshold, including
        # float32 values just below a float64 policy boundary.
        indexes = np.flatnonzero(scores.astype(np.float64, copy=False) >= threshold)
        matches = [(labels[int(i)], float(scores[i])) for i in indexes]
    else:
        matches = [(label, float(score)) for label, score in zip(labels, scores) if score >= threshold]
    matches.sort(key=lambda item: (-item[1], item[0]))
    return matches[:limit] if limit else matches


def predict_scene(tagger: Any, stash: StashClient, paths: dict, cues: list[Cue],
                  scene_exclusions: frozenset[str] = frozenset(), *, raw_scene: bool = False,
                  checkpoint=None, prefetch: int = 0, analysis: dict | None = None, report=None, prepare_workers: int = 1
                  ) -> tuple[list[tuple[str, float]], list[FramePrediction]]:
    import numpy as np
    from stash_taggers import validate_scores
    from stash_progress import prepared_batches
    from stash_temporal import SceneAccumulator, marker_floor, scene_reason
    if not cues:
        raise ValueError("Cannot predict a scene without VTT cues.")
    settings, labels = tagger.settings, tagger.labels
    retain_scene = raw_scene and APPLY_SCENE_TAGS
    scene_floor = settings.scene_threshold * (TEMPORAL_POLICY.scene_support_ratio if TEMPORAL_POLICY.enabled else 1.0)
    processed, top_scores, saved = checkpoint.restore() if checkpoint else (0, np.empty((0, len(labels)), np.float32), [])
    accumulator = (SceneAccumulator(settings.scene_threshold, TEMPORAL_POLICY,
                   checkpoint.analysis if checkpoint else None) if TEMPORAL_POLICY.enabled and APPLY_SCENE_TAGS else None)
    if accumulator is not None and accumulator.frames != processed:
        raise ValueError("Checkpoint scene statistics do not match completed cues; use --restart.")
    if checkpoint and (CREATE_MARKERS or retain_scene) and len(saved) != processed:
        raise ValueError("Checkpoint frame count does not match completed cues.")
    frame_predictions = [FramePrediction(cues[index], tuple((tag, score) for tag, score in frame["tags"]))
                         for index, frame in enumerate(saved)] if CREATE_MARKERS else []
    scene_frames = [dict(start=cues[i].start, end=cues[i].end, tags=row["scene_tags"])
                    for i, row in enumerate(saved)] if retain_scene else []
    if processed:
        if report is not None:
            report.counts['checkpoint_resumes'] += 1
            report.detail(f"  Resume {settings.source.key}: {processed}/{len(cues)} frames already processed.")
    keep = 1 if AGGREGATION == "max" else min(TOP_K_FRAMES, len(cues))
    batches = prepared_batches(sys.modules[__name__], tagger, stash, paths, cues[processed:], prefetch, prepare_workers)
    try:
        for prepared, batch_cues in batches:
            result = validate_scores(tagger.predict_batch(prepared), len(prepared), labels, settings.source.title)
            combined = np.concatenate((top_scores, result), axis=0)
            count = min(keep, len(combined))
            next_top = np.partition(combined, len(combined) - count, axis=0)[-count:]
            batch_frames = []
            if CREATE_MARKERS:
                for cue, row in zip(batch_cues, result):
                    matches = select_scores(row, labels, marker_floor(TEMPORAL_POLICY, settings.marker_threshold), 0)
                    matches = [(tag, score) for tag, score in matches if tag not in MARKER_TAG_BLOCKLIST]
                    batch_frames.append(FramePrediction(cue, tuple(matches)))
            batch_scene = [dict(start=cue.start, end=cue.end,
                           tags=select_scores(row, labels, scene_floor, 0))
                           for cue, row in zip(batch_cues, result)] if retain_scene else []
            scene_frames.extend(batch_scene)
            if accumulator is not None:
                accumulator.add(result, batch_cues)
            top_scores = next_top
            frame_predictions.extend(batch_frames)
            processed += len(batch_cues)
            if checkpoint:
                checkpoint.record(processed, top_scores, batch_frames,
                                  analysis=accumulator.state() if accumulator is not None else None,
                                  scene_frames=batch_scene if retain_scene else None)
    finally:
        # Save the last fully completed batch on Ctrl+C/inference/preparation errors.
        try:
            if checkpoint:
                checkpoint.flush()
        finally:
            batches.close()
    if processed != len(cues) or not len(top_scores):
        raise RuntimeError("Incomplete inference; scene sync was not started.")
    scene_scores = np.sort(top_scores, axis=0)[::-1].mean(axis=0)
    statistics = accumulator.statistics(labels) if accumulator is not None else {}
    if analysis is not None:
        analysis["scene_statistics"] = statistics
        if retain_scene:
            analysis.update(scene_frames=scene_frames, scene_score_floor=scene_floor)
    if accumulator is not None:
        predictions = [(labels[index], float(scene_scores[index])) for index in accumulator.items]
        predictions = [(tag, score) for tag, score in predictions if raw_scene or
                       (tag not in scene_exclusions and scene_reason(score, statistics[tag], settings.scene_threshold, TEMPORAL_POLICY))]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    elif retain_scene:
        candidates = {tag for row in scene_frames for tag, _ in row["tags"]}
        predictions = [(tag, float(scene_scores[i])) for i, tag in enumerate(labels) if tag in candidates]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    else:
        predictions = [(tag, score) for tag, score in select_scores(scene_scores, labels, settings.scene_threshold, 0)
                       if raw_scene or tag not in scene_exclusions]
    if MAX_TAGS and not raw_scene:
        predictions = predictions[:MAX_TAGS]
    if scene_exclusions and not raw_scene:
        frame_predictions = [FramePrediction(frame.cue, tuple((tag, score) for tag, score in frame.tags
                                                            if tag not in scene_exclusions)) for frame in frame_predictions]
    return predictions, frame_predictions


def build_markers(frames: list[FramePrediction], core_exclusions: frozenset[str], *,
                  threshold=None, seeds=None, identity=None, cache=None) -> tuple[list[MarkerSpec], int]:
    """Group eligible cores independently; exclusions may still be extras."""
    if TEMPORAL_POLICY.enabled:
        from stash_temporal import build_markers as temporal_markers
        return temporal_markers(sys.modules[__name__], frames, core_exclusions,
                                threshold=threshold, seeds=seeds, identity=identity, extra_cache=cache)
    if threshold is not None:
        frames = [FramePrediction(frame.cue, tuple((tag, score) for tag, score in frame.tags if score >= threshold)) for frame in frames]
    active: dict[str, dict] = {}
    finished: list[MarkerSpec] = []

    def finish(tag: str) -> None:
        run = active.pop(tag)
        if run["hits"] >= MARKER_MIN_HITS:
            finished.append(MarkerSpec(tag=tag, **run))

    for frame in frames:
        core_tags = [(tag, score) for tag, score in frame.tags if tag not in core_exclusions]
        additional_tags = frame.tags
        if MARKER_MAX_TAGS_PER_FRAME:
            core_tags = core_tags[:MARKER_MAX_TAGS_PER_FRAME]
            additional_tags = additional_tags[:MARKER_MAX_TAGS_PER_FRAME]
        for tag, score in core_tags:
            if tag in active and frame.cue.start - active[tag]["end"] > MARKER_MERGE_GAP_SECONDS + 0.001:
                finish(tag)
            if tag not in active:
                active[tag] = {"start": frame.cue.start, "end": frame.cue.end,
                               "score": score, "hits": 1,
                               "extra_tags": tuple(name for name, _ in additional_tags if name != tag)}
            else:
                run = active[tag]
                run["end"] = max(run["end"], frame.cue.end)
                run["score"] = max(run["score"], score)
                run["hits"] += 1
    for tag in list(active):
        finish(tag)
    total = len(finished)
    finished.sort(key=lambda marker: (-marker.score, marker.start, marker.tag))
    if MAX_MARKERS_PER_SCENE:
        finished = finished[:MAX_MARKERS_PER_SCENE]
    return sorted(finished, key=lambda marker: (marker.start, marker.tag)), total


def owned_marker(marker: dict, source: ModelSource = JOYTAG) -> bool:
    return bool((marker.get("title") or "").startswith(source.title_prefix)
                and any(tag.get("name") == source.owner_tag for tag in marker.get("tags") or []))


def marker_has_excluded_core(marker: dict, core_exclusions: frozenset[str],
                             source: ModelSource = JOYTAG) -> bool:
    name = (marker.get("primary_tag") or {}).get("name") or ""
    return name.startswith(source.prefix) and name[len(source.prefix):] in core_exclusions


def marker_extra_cleanup_payload(marker: dict, scene_id: str, blocked_names: set[str], has_end: bool):
    """Preserve marker identity, primary, title and timing while stripping extras."""
    primary = str(marker["primary_tag"]["id"])
    extras = {str(tag["id"]) for tag in marker.get("tags", []) if str(tag["id"]) != primary}
    kept = {str(tag["id"]) for tag in marker.get("tags", [])
            if str(tag["id"]) != primary and tag["name"] not in blocked_names}
    if extras == kept:
        return None
    payload = dict(id=str(marker["id"]), scene_id=scene_id, title=marker["title"],
                   seconds=marker["seconds"], primary_tag_id=primary, tag_ids=sorted(kept))
    if has_end:
        payload["end_seconds"] = marker.get("end_seconds")
    return payload


def cleanup_scene_marker_cores(stash: StashClient, scene: dict,
                              core_exclusions: frozenset[str], report=None, source: ModelSource = JOYTAG,
                              tag_exclusions: frozenset[str] = frozenset()) -> Counter:
    """Remove excluded cores and global extras; never infer replacement runs."""
    core_exclusions = core_exclusions | tag_exclusions
    excluded = [marker for marker in scene.get("scene_markers") or []
                if (owned_marker(marker, source) and marker_has_excluded_core(marker, core_exclusions, source))
                or marker_has_excluded_core(marker, tag_exclusions, source)]
    stats = Counter(markers_deleted=len(excluded), markers_core_cleaned=len(excluded))
    for marker in excluded:
        if report:
            report(f"{'[DRY] Would delete' if stash.dry_run else 'Deleting'} marker {marker['id']} "
                   f"on scene {scene['id']}: core={marker['primary_tag']['name']!r}, "
                   f"start={format_time(marker['seconds'])}")
        if not stash.dry_run:
            stash.delete_marker(str(marker["id"]))
    excluded_ids = {str(marker["id"]) for marker in excluded}
    blocked_names = {source.prefix + tag for tag in tag_exclusions | MARKER_TAG_BLOCKLIST}
    for marker in scene.get("scene_markers") or []:
        if str(marker["id"]) in excluded_ids:
            continue
        names = blocked_names if owned_marker(marker, source) else {source.prefix + tag for tag in tag_exclusions}
        payload = marker_extra_cleanup_payload(marker, str(scene["id"]), names, stash.has_end_time)
        if payload is not None:
            stats["markers_updated"] += 1
            stats["markers_extras_cleaned"] += 1
            if report:
                report(f"{'[DRY] ' if stash.dry_run else ''}Remove excluded extras from marker {marker['id']} on scene {scene['id']}.")
            if not stash.dry_run:
                stash.update_marker(payload)
    return stats


def marker_payload(spec: MarkerSpec, scene_id: str, ids: dict[str, str | None], has_end: bool,
                   source: ModelSource = JOYTAG) -> dict:
    primary = ids[source.prefix + spec.tag]
    extras = {ids[source.prefix + tag] for tag in spec.extra_tags}
    extras.add(ids[source.owner_tag])
    extras.discard(primary)
    extras.discard(None)
    payload = {"scene_id": scene_id, "title": source.title_prefix + spec.tag,
               "seconds": round(spec.start, 3), "primary_tag_id": primary,
               "tag_ids": sorted(extras)}
    if has_end:
        payload["end_seconds"] = round(spec.end, 3) if MARKER_USE_END_TIME else None
    return payload


def marker_differs(existing: dict, wanted: dict) -> bool:
    if existing["title"] != wanted["title"] or abs(existing["seconds"] - wanted["seconds"]) > 0.0005:
        return True
    if str(existing["primary_tag"]["id"]) != wanted["primary_tag_id"]:
        return True
    # Some API versions include the primary tag in the returned tags list.
    extra_ids = {str(tag["id"]) for tag in existing["tags"]} - {wanted["primary_tag_id"]}
    if extra_ids != set(wanted["tag_ids"]):
        return True
    if "end_seconds" in wanted:
        old_end, new_end = existing.get("end_seconds"), wanted["end_seconds"]
        if old_end is None or new_end is None:
            return old_end != new_end
        return abs(old_end - new_end) > 0.0005
    return False


def plan_marker_changes(existing: list[dict], wanted: list[dict],
                        core_exclusions: frozenset[str], source: ModelSource = JOYTAG) -> tuple[list[dict], list[dict], list[str], int]:
    available = {str(marker["id"]): marker for marker in existing if owned_marker(marker, source)}
    creates, updates = [], []
    unchanged = 0
    for payload in wanted:
        matches = [marker for marker in available.values()
                   if str(marker["primary_tag"]["id"]) == payload["primary_tag_id"]
                   and abs(marker["seconds"] - payload["seconds"]) <= MARKER_MATCH_TOLERANCE_SECONDS]
        if not matches:
            creates.append(payload)
            continue
        current = min(matches, key=lambda marker: (abs(marker["seconds"] - payload["seconds"]), str(marker["id"])))
        marker_id = str(current["id"])
        del available[marker_id]
        if marker_differs(current, payload):
            updates.append({**payload, "id": marker_id})
        else:
            unchanged += 1
    # The exclusion policy always wins over retaining obsolete generated runs.
    deletes = sorted(marker_id for marker_id, marker in available.items()
                     if REPLACE_MARKERS or marker_has_excluded_core(marker, core_exclusions, source))
    return creates, updates, deletes, unchanged


def sync_scene(stash: StashClient, resolver: TagResolver, scene: dict,
               scene_predictions: list[tuple[str, float]], specs: list[MarkerSpec],
               core_exclusions: frozenset[str], source: ModelSource = JOYTAG,
               scene_exclusions: frozenset[str] = frozenset()) -> Counter:
    if resolver.source != source:
        raise ValueError("Resolver/source mismatch; refusing to mix model namespaces.")
    spec_count = len(specs)
    # Enforce again at reconciliation, even for specs supplied outside inference.
    from stash_exclusions import filter_spec
    core_exclusions = core_exclusions | scene_exclusions
    specs = [filter_spec(spec, scene_exclusions) for spec in specs if spec.tag not in core_exclusions]
    scene_id = str(scene["id"])
    existing_tags = scene.get("tags") or []
    scene_names = {source.prefix + tag for tag, _ in scene_predictions if tag not in scene_exclusions} if APPLY_SCENE_TAGS else set()
    names = set(scene_names)
    if CREATE_MARKERS and specs:
        names.add(source.owner_tag)
        for spec in specs:
            names.add(source.prefix + spec.tag)
            names.update(source.prefix + tag for tag in spec.extra_tags)
    # Resolve everything before replacing scene tags or removing old markers.
    ids = {name: resolver.resolve(name) for name in sorted(names)}
    if CREATE_MARKERS and specs and not ids.get(source.owner_tag):
        raise ValueError(f"Missing marker ownership tag {source.owner_tag!r}; enable CREATE_MISSING or create it in Stash.")
    wanted = [marker_payload(spec, scene_id, ids, stash.has_end_time, source) for spec in specs
              if ids.get(source.prefix + spec.tag)] if CREATE_MARKERS else []
    existing_markers = scene.get("scene_markers") or []
    plan = plan_marker_changes(existing_markers, wanted, core_exclusions, source) if CREATE_MARKERS else ([], [], [], 0)
    creates, updates, deletes, unchanged = plan
    if CREATE_MARKERS:
        deletes = sorted(set(deletes) | {str(marker['id']) for marker in existing_markers
                                        if marker_has_excluded_core(marker, scene_exclusions, source)})
    # Retained markers must also lose forbidden extras when replacement is off.
    changed_ids = set(deletes) | {payload["id"] for payload in updates}
    wanted_keys = {(payload["primary_tag_id"], payload["seconds"]) for payload in wanted}
    extra_updates = []
    if CREATE_MARKERS and scene_exclusions:
        blocked_names = {source.prefix + tag for tag in scene_exclusions}
        for marker in existing_markers:
            if str(marker["id"]) in changed_ids:
                continue
            if (str(marker["primary_tag"]["id"]), marker["seconds"]) in wanted_keys:
                continue
            payload = marker_extra_cleanup_payload(marker, scene_id, blocked_names, stash.has_end_time)
            if payload is not None:
                extra_updates.append(payload)
        updates.extend(extra_updates)
    excluded_ids = {str(marker["id"]) for marker in existing_markers
                    if owned_marker(marker, source) and marker_has_excluded_core(marker, core_exclusions, source)}
    stats = Counter(markers_created=len(creates), markers_updated=len(updates),
                    markers_deleted=len(deletes), markers_unchanged=unchanged,
                    markers_skipped=spec_count - len(wanted),
                    markers_extras_cleaned=len(extra_updates),
                    markers_core_cleaned=len(excluded_ids.intersection(deletes)))

    existing_ids = {str(tag["id"]) for tag in existing_tags}
    preserved = {str(tag["id"]) for tag in existing_tags
                 if not (tag["name"].startswith(source.prefix) and (REPLACE_SCENE_TAGS
                     or tag["name"][len(source.prefix):] in scene_exclusions))}
    final_ids = preserved | {ids[name] for name in scene_names if ids[name] is not None}
    scene_changed = APPLY_SCENE_TAGS and final_ids != existing_ids
    stats["scene_tags_updated"] = int(scene_changed)
    stats["scene_tags_cleaned"] = sum(tag["name"].startswith(source.prefix)
        and tag["name"][len(source.prefix):] in scene_exclusions for tag in existing_tags) if APPLY_SCENE_TAGS else 0

    if stash.dry_run:
        return stats
    # No destructive cleanup until all new/replacement marker writes succeed.
    for payload in creates:
        stash.create_marker(payload)
    for payload in updates:
        stash.update_marker(payload)
    if scene_changed:
        stash.update_scene_tags(scene_id, sorted(final_ids))
    for marker_id in deletes:
        stash.delete_marker(marker_id)
    return stats


def validate_config(scene_work: bool = True) -> None:
    parsed = urlparse(STASH_URL)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("STASH_URL must be an http(s) base URL without credentials/query/fragment.")
    _ = parsed.port
    if scene_work and not APPLY_SCENE_TAGS and not CREATE_MARKERS:
        raise ValueError("Enable scene tags, markers, or both.")
    for name, value in (("JOYTAG_DEFAULT_SCENE_THRESHOLD", JOYTAG_DEFAULT_SCENE_THRESHOLD), ("DEFAULT_MARKER_THRESHOLD", DEFAULT_MARKER_THRESHOLD)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    for name, value in (("JOYTAG_DEFAULT_BATCH_SIZE", JOYTAG_DEFAULT_BATCH_SIZE), ("TOP_K_FRAMES", TOP_K_FRAMES),
                        ("MARKER_MIN_HITS", MARKER_MIN_HITS), ("SCENE_PAGE_SIZE", SCENE_PAGE_SIZE)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("MAX_TAGS", MAX_TAGS), ("MAX_FRAMES_PER_SCENE", MAX_FRAMES_PER_SCENE),
                        ("MARKER_MAX_TAGS_PER_FRAME", MARKER_MAX_TAGS_PER_FRAME),
                        ("MAX_MARKERS_PER_SCENE", MAX_MARKERS_PER_SCENE)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0 (0 = unlimited).")
    for value in (MARKER_MERGE_GAP_SECONDS, MARKER_MATCH_TOLERANCE_SECONDS):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Marker time gaps/tolerances must be finite and nonnegative.")
    if AGGREGATION not in {"max", "topk_mean"}:
        raise ValueError("AGGREGATION must be max or topk_mean.")


def print_hierarchy_summary(resolver: TagResolver) -> None:
    if resolver.hierarchy:
        print(resolver.source.title + " hierarchy" + (" (planned)" if resolver.stash.dry_run else "") + ": " + ", ".join(
            f"{name}={resolver.hierarchy_stats[name]}" for name in
            ("categories_created", "tags_created", "tags_reparented", "parents_unchanged", "hierarchy_skipped")
        ))


def cleanup_scene_tags(stash: StashClient, scene: dict, exclusions: frozenset[str],
                       source: ModelSource, report=None) -> Counter:
    existing = scene.get("tags") or []
    removed = [tag for tag in existing if tag["name"].startswith(source.prefix)
               and tag["name"][len(source.prefix):] in exclusions]
    if removed:
        if report:
            report(f"{'[DRY] ' if stash.dry_run else ''}{source.key} scene {scene['id']}: remove "
                   + ", ".join(tag["name"] for tag in removed))
        retained = {str(tag["id"]) for tag in existing} - {str(tag["id"]) for tag in removed}
        if not stash.dry_run:
            stash.update_scene_tags(str(scene["id"]), sorted(retained))
    return Counter(scene_tags_updated=int(bool(removed)), scene_tags_cleaned=len(removed))


def outputs_already_present(scene: dict, source: ModelSource) -> bool:
    tagged = any(tag["name"].startswith(source.prefix) for tag in scene.get("tags") or [])
    marked = any(owned_marker(marker, source) for marker in scene.get("scene_markers") or [])
    return (not APPLY_SCENE_TAGS or tagged) and (not CREATE_MARKERS or marked)


def run_model(stash: StashClient, config: ModelSettings, vocabulary: Vocabulary,
              hierarchy: TagHierarchy | None, args: Any, core_exclusions: frozenset[str],
              scene_exclusions: frozenset[str]) -> Counter:
    from tqdm import tqdm
    from stash_runtime import RamCache
    source = config.source
    report = args.report
    resolver = TagResolver(stash, hierarchy, report=report.detail, source=source, exclusions=scene_exclusions, create_missing=CREATE_MISSING)
    stats = Counter()
    if args.sync_hierarchy_only:
        resolver.preload()
        labels = [label for label in vocabulary.labels if label not in scene_exclusions]
        selected = labels if args.create_all_tags else [label for label in labels if source.prefix + label in resolver.cache]
        print(f"{'[DRY] ' if DRY_RUN else ''}{source.title}: synchronizing hierarchy for {len(selected):,} tags.")
        for label in tqdm(selected, desc=source.key + " hierarchy", unit="tag"):
            resolver.resolve(source.prefix + label)
        print_hierarchy_summary(resolver)
        return stats
    tagger = load_tagger(config, vocabulary, prepare_image)
    try:
        progress = tqdm(stash.iter_scenes(args.scene_id), desc=source.key, unit="scene")
        for scene in progress:
            scene_id = str(scene["id"])
            title = (scene.get("title") or "")[:60]
            paths = scene.get("paths") or {}
            if not paths.get("sprite") or not paths.get("vtt"):
                stats["skipped"] += 1
                report.detail(f"SKIP {source.key} scene {scene_id}: sprite/VTT missing; generate Stash sprites first.")
                continue
            try:
                if SKIP_EXISTING_OUTPUTS and outputs_already_present(scene, source):
                    stats["skipped"] += 1
                    continue
                cues, total_frames = fetch_cues(stash, paths)
                predictions, frame_predictions = predict_scene(tagger, stash, paths, cues, scene_exclusions,
                                                                prefetch=args.prefetch_batches, report=report,
                                                                prepare_workers=getattr(args, 'prepare_workers', 1))
                markers, total_markers = build_markers(frame_predictions, core_exclusions,
                    threshold=config.marker_threshold, cache=RamCache(
                        int(args.ram_cache_gb * 2**30),
                        rss_budget=int(getattr(args, 'ram_budget_gb', 32) * 2**30))) if CREATE_MARKERS else ([], 0)
                current = stash.find_scene(scene_id)
                result = sync_scene(stash, resolver, current, predictions, markers, core_exclusions, source, scene_exclusions)
                stats.update(result)
                stats["processed"] += 1
                report.detail(
                    f"{'[DRY] ' if DRY_RUN else ''}{source.key} scene {scene_id} {title!r}: "
                    f"frames={len(cues)}/{total_frames}, scene_tags={len(predictions)}, "
                    f"markers create={result['markers_created']} update={result['markers_updated']} "
                    f"delete={result['markers_deleted']} keep={result['markers_unchanged']} "
                    f"core_cleanup={result['markers_core_cleaned']} scene_cleanup={result['scene_tags_cleaned']}"
                )
                if total_markers > len(markers):
                    report.detail(f"  Marker cap: kept {len(markers)} of {total_markers} runs by confidence.")
                if DRY_RUN and report.verbose:
                    if APPLY_SCENE_TAGS:
                        tqdm.write("  Scene predictions: " + (", ".join(
                            f"{source.prefix}{tag}={score:.3f}" for tag, score in predictions) or "(none)"))
                    for marker in markers:
                        interval = format_time(marker.start)
                        if MARKER_USE_END_TIME and stash.has_end_time:
                            interval += " -> " + format_time(marker.end)
                        tqdm.write(f"  {interval} | {source.prefix}{marker.tag} | "
                                   f"confidence={marker.score:.3f}, hits={marker.hits} | "
                                   f"extra={', '.join(source.prefix + tag for tag in marker.extra_tags) or '(none)'}")
            except Exception as exc:
                stats["errors"] += 1
                report.issue('inference', source.key, scene_id, exc, safe_error(exc, api_key=STASH_API_KEY))
            finally:
                report.progress(progress, stats)
        if resolver.missing:
            verb = "Would create" if DRY_RUN and CREATE_MISSING else "Missing (skipped)"
            print(f"{source.key}: {verb} tags: {len(resolver.missing):,}")
        print_hierarchy_summary(resolver)
    finally:
        tagger.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    global DRY_RUN, APPLY_SCENE_TAGS, CREATE_MARKERS, TEMPORAL_POLICY
    parser = argparse.ArgumentParser(description="Tag Stash scenes and timestamped markers with JoyTag, WD, DTQ and/or OppaiOracle.")
    parser.add_argument("--models", "--model", nargs="+", metavar="MODEL", help="joytag, wd, dtq, oppai, any subset, or all; commas also accepted.")
    parser.add_argument("--model-config", type=Path, default=MODELS_CONFIG_FILE, metavar="YAML", help="Selection, paths, thresholds and batch sizes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override configured device.")
    parser.add_argument("--offline", action="store_true", help="Require local model assets; never download.")
    parser.add_argument("--temporal-policy", type=Path, default=TEMPORAL_POLICY_FILE, metavar="YAML",
                        help="Automatic scene qualification, marker hysteresis and extras using Stash VTT/sprites only.")
    parser.add_argument("--no-temporal", action="store_true", default=not DEFAULT_TEMPORAL,
                        help="Restore original scene/marker qualification and first-frame extras.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to Stash.")
    parser.add_argument("--scene-id", action="append", metavar="ID", help="Limit to this scene; repeat for multiple scenes.")
    parser.add_argument("--output-mode", choices=("merged", "separate"), default=DEFAULT_OUTPUT_MODE,
                        help="merged (default): common ai: tags and interval union; separate: original model prefixes.")
    parser.add_argument("--state-db", type=Path, default=MERGE_STATE_FILE, metavar="SQLITE",
                        help="Durable per-model evidence for merged outputs; retain this file between runs.")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", default=None,
                        help="Reuse matching completed results and resume checkpoints (default; merged output).")
    resume.add_argument("--restart", action="store_true", help="Request fresh inference for selected scenes/models; resume later without this flag.")
    parser.add_argument("--checkpoint-batches", type=int, default=CHECKPOINT_BATCHES,
                        help="Persist partial inference every N batches; also saves on Ctrl+C/errors. Default: 8.")
    parser.add_argument("--prefetch-batches", type=int, choices=range(17), default=PREFETCH_BATCHES,
                        help="Prepare N next batches in RAM (default: 4); 0 disables. No disk image cache.")
    parser.add_argument("--ram-cache-gb", type=float, default=RAM_CACHE_GB, metavar="GIB",
                        help="Shared result/index/media cache in GiB (default: 24); 0 disables. Capped to 75%% of --ram-budget-gb.")
    parser.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB, metavar="GIB",
                        help="Process RAM target in GiB (default: 32); reserves 25%% for models and active work. RSS pressure evicts caches.")
    parser.add_argument("--prepare-workers", type=int, choices=range(1, 17), default=PREPARE_WORKERS,
                        help="CPU image preparation workers (default: 4); one model at a time runs inference.")
    parser.add_argument("--verbose", action="store_true", help="Include per-scene/tag diagnostics and detailed dry-run changes.")
    parser.add_argument("--profile-merge", type=Path, metavar="NEW.jsonl",
                        help="Stream per-scene wall/CPU, stage/API timings, counts and byte sizes to a new JSONL file; merged output only.")
    parser.add_argument("--force-merge", action="store_true",
                        help="Recompute even unchanged merges without rerunning models; requires --merge-only. Does not clear source state.")
    parser.add_argument("--rethreshold-scenes", action="store_true",
                        help="Apply YAML scene thresholds to retained frame evidence; requires --merge-only. "
                             "May be combined with --rethreshold-markers.")
    parser.add_argument("--rethreshold-markers", action="store_true",
                        help="Apply YAML marker thresholds to complete cached frames; requires --merge-only. "
                             "Never runs inference; rejects changed scene thresholds or insufficient retention.")
    parser.add_argument("--abandon-pending", action="store_true",
                        help="Abandon pending merge/cleanup plans, retaining applied changes and inference. "
                             "Alone: exit after abandonment; with --merge-only or cleanup: use current YAML policies. Honors --dry-run.")
    parser.add_argument("--cleanup-strategy", choices=("auto", "sql", "scenes"), default="auto",
                        help="Cleanup discovery: auto uses read-only SQL when available, else scans scenes; sql requires SQL; scenes scans scenes directly.")
    parser.add_argument("--cleanup-page-size", type=int, default=2000,
                        help="Maximum ID rows per SQL cleanup SELECT (default: 2000; no COUNT or OFFSET).")
    parser.add_argument("--cleanup-write-batch-size", type=int, default=200,
                        help="Maximum marker/scene IDs per cleanup mutation (default: 200). Tags are deleted one at a time.")
    parser.add_argument("--prune-empty-tags", action="store_true",
                        help="After successful merged sync, delete unused leaf tags in the selected prefixes.")
    parser.add_argument("--prune-prefixes", nargs="+", default=None, metavar="PREFIX",
                        help="Prefixes for empty-tag pruning (default: jt: wd: dtq: op: ai: m:). Global scope; m: requires managed group metadata.")
    parser.add_argument("--hierarchy", type=Path, default=JOYTAG_HIERARCHY_FILE, metavar="YAML", help="JoyTag hierarchy.")
    parser.add_argument("--wd-hierarchy", type=Path, default=JOYTAG_HIERARCHY_FILE.with_name("wd_hierarchy.yml"), metavar="YAML", help="WD hierarchy; defaults to bundled wd_hierarchy.yml.")
    parser.add_argument("--dtq-hierarchy", type=Path, default=JOYTAG_HIERARCHY_FILE.with_name("dtq_hierarchy.yml"), metavar="YAML", help="DTQ hierarchy; defaults to bundled dtq_hierarchy.yml.")
    parser.add_argument("--oppai-hierarchy", type=Path, default=JOYTAG_HIERARCHY_FILE.with_name("oppai_hierarchy.yml"), metavar="YAML", help="OppaiOracle hierarchy; defaults to bundled oppai_hierarchy.yml.")
    parser.add_argument("--marker-core-exclusions", type=Path, default=MARKER_CORE_EXCLUSIONS_FILE, metavar="YAML", help="Common/per-model excluded marker core labels.")
    parser.add_argument("--scene-tag-exclusions", "--tag-exclusions", dest="scene_tag_exclusions", type=Path,
                        default=SCENE_TAG_EXCLUSIONS_FILE, metavar="YAML",
                        help="Global blacklist for scene tags, marker cores/extras and hierarchy creation (legacy filename supported).")
    parser.add_argument("--no-hierarchy", action="store_true", help="Disable model/category parent synchronization; marker groups still use ai_markers.")
    parser.add_argument("--create-all-tags", action="store_true", help="With --sync-hierarchy-only, create full vocabularies.")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--markers-only", action="store_true", help="Leave scene tags unchanged; also works with --merge-only.")
    outputs.add_argument("--scene-tags-only", action="store_true", help="Leave markers unchanged; also works with --merge-only.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sync-hierarchy-only", action="store_true", help="Organize tags without inference or scene/marker edits.")
    modes.add_argument("--validate-hierarchy", action="store_true", help="Validate vocabularies/hierarchies without Stash or weights.")
    modes.add_argument("--cleanup-markers-only", "--sync-marker-core-only", dest="cleanup_markers_only", action="store_true", help="Remove excluded marker cores and global extras without inference or model files.")
    modes.add_argument("--cleanup-scene-tags-only", action="store_true", help="Apply scene blacklist without inference or model files.")
    modes.add_argument("--cleanup-only", action="store_true", help="Apply both blacklists without inference; full all-model cleanup removes globally banned tags in batches.")
    modes.add_argument("--download-models", action="store_true", help="Download selected model assets from Hugging Face without Stash or inference.")
    modes.add_argument("--merge-only", action="store_true", help="Merge saved evidence and existing source-prefixed results without models/inference.")
    modes.add_argument("--sync-marker-groups-only", action="store_true",
                       help="Repair m:<core> / m:<core>:occurance groups on existing owned ai: markers; no inference or time changes.")
    modes.add_argument("--explain-tag", metavar="TAG", help="Show saved sources, scores and intervals for a raw/ai: tag; requires --scene-id; no Stash access.")
    modes.add_argument("--prune-empty-tags-only", action="store_true",
                       help="Remove unused leaf tag definitions without models/inference. Honors --dry-run.")
    modes.add_argument("--db-report", action="store_true",
                       help="Report state DB health, size and aggregate evidence/cache counts without Stash or model access.")
    modes.add_argument("--db-compact", action="store_true",
                       help="Safely VACUUM the state DB while idle; refuses pending publication/cleanup plans. Honors --dry-run.")
    modes.add_argument("--anonymous-diagnostics", type=Path, metavar="NEW.json",
                       help="Write a local content-free developer diagnostics JSON from the state DB; never uploads it.")
    modes.add_argument("--model-report", type=Path, metavar="NEW.json",
                       help="Write an offline per-model utility and leave-one-model-out ablation report from saved evidence.")
    args = parser.parse_args(argv)
    database_mode = args.db_report or args.db_compact or args.anonymous_diagnostics is not None
    if args.model_report is not None:
        if args.output_mode != "merged":
            parser.error("--model-report requires --output-mode merged.")
        if args.models or args.device or args.offline or args.resume is not None or args.restart:
            parser.error("--model-report uses all saved model evidence; model/inference options do not apply.")
        if (args.markers_only or args.scene_tags_only or args.profile_merge or args.force_merge
                or args.rethreshold_scenes or args.rethreshold_markers or args.abandon_pending
                or args.prune_empty_tags or args.prune_prefixes or args.create_all_tags or args.no_hierarchy
                or args.dry_run):
            parser.error("--model-report cannot be combined with tagging, threshold, pruning or dry-run modifiers.")
        if args.model_report.suffix.lower() != ".json" or args.model_report.exists():
            parser.error("--model-report requires a new .json file; existing files are never overwritten.")
    if database_mode:
        if args.output_mode != "merged":
            parser.error("--db-report/--db-compact require --output-mode merged.")
        if args.scene_id:
            parser.error("Database maintenance covers the state DB; --scene-id does not apply.")
        if args.models or args.device or args.offline or args.no_temporal or args.resume is not None or args.restart:
            parser.error("Database maintenance does not use model/inference options.")
        if (args.markers_only or args.scene_tags_only or args.profile_merge or args.force_merge
                or args.rethreshold_scenes or args.rethreshold_markers or args.abandon_pending
                or args.prune_empty_tags or args.prune_prefixes or args.create_all_tags or args.no_hierarchy):
            parser.error("Database maintenance cannot be combined with tagging, threshold, cleanup or pruning modifiers.")
        if args.anonymous_diagnostics is not None:
            if args.dry_run:
                parser.error("--anonymous-diagnostics is already read-only for the state DB; --dry-run does not apply.")
            from stash_developer_report import run as run_anonymous_diagnostics
            return run_anonymous_diagnostics(args)
        from stash_db import run_mode as run_database_mode
        return run_database_mode(args, STASH_URL)
    if args.output_mode != "merged" and (args.resume or args.restart):
        parser.error("--resume/--restart require --output-mode merged; separate retains its original full-inference behavior.")
    args.resume = DEFAULT_RESUME if args.resume is None else args.resume
    if args.checkpoint_batches < 1:
        parser.error("--checkpoint-batches must be >= 1.")
    if not 1 <= args.cleanup_page_size <= 10000 or not 1 <= args.cleanup_write_batch_size <= 1000:
        parser.error("--cleanup-page-size must be 1..10000 and --cleanup-write-batch-size must be 1..1000.")
    if not math.isfinite(args.ram_cache_gb) or args.ram_cache_gb < 0:
        parser.error("--ram-cache-gb must be a finite number >= 0.")
    if not math.isfinite(args.ram_budget_gb) or args.ram_budget_gb <= 0:
        parser.error("--ram-budget-gb must be a finite number > 0.")
    args.ram_cache_gb = min(args.ram_cache_gb, args.ram_budget_gb * .75)
    from stash_runtime import RunReport
    from tqdm import tqdm
    args.report = RunReport(args.verbose, tqdm.write)
    DRY_RUN = DRY_RUN or args.dry_run
    if args.markers_only or args.cleanup_markers_only or args.sync_marker_groups_only:
        APPLY_SCENE_TAGS, CREATE_MARKERS = False, True
    elif args.scene_tags_only or args.cleanup_scene_tags_only:
        APPLY_SCENE_TAGS, CREATE_MARKERS = True, False
    elif args.cleanup_only:
        APPLY_SCENE_TAGS, CREATE_MARKERS = True, True
    hierarchy_only = args.sync_hierarchy_only or args.validate_hierarchy
    cleanup_only = args.cleanup_only or args.cleanup_markers_only or args.cleanup_scene_tags_only
    if args.abandon_pending:
        if (args.output_mode != 'merged' or hierarchy_only or args.download_models or args.explain_tag
                or args.prune_empty_tags_only or args.sync_marker_groups_only or args.restart):
            parser.error("--abandon-pending is standalone or accompanies merged --merge-only/cleanup modes.")
        if not args.merge_only and not cleanup_only and (args.markers_only or args.scene_tags_only or args.prune_empty_tags or args.profile_merge):
            parser.error("Standalone --abandon-pending does not publish outputs or prune tags.")
    if args.rethreshold_scenes and (not args.merge_only or args.markers_only):
        parser.error("--rethreshold-scenes requires --merge-only and scene output.")
    if args.rethreshold_markers and (not args.merge_only or args.scene_tags_only):
        parser.error("--rethreshold-markers requires --merge-only and marker output.")
    if args.force_merge and not args.merge_only:
        parser.error("--force-merge requires --merge-only.")
    if args.profile_merge:
        if args.output_mode != 'merged' or hierarchy_only or cleanup_only or args.download_models or args.explain_tag or args.prune_empty_tags_only or args.sync_marker_groups_only:
            parser.error("--profile-merge applies to merged tagging or --merge-only.")
        if args.profile_merge.suffix.lower() != '.jsonl' or args.profile_merge.exists():
            parser.error("--profile-merge requires a new .jsonl file; existing files are never overwritten.")
    if (args.markers_only or args.scene_tags_only) and (hierarchy_only or cleanup_only or args.download_models or args.explain_tag or args.prune_empty_tags_only or args.sync_marker_groups_only):
        parser.error("--markers-only/--scene-tags-only apply to tagging or --merge-only.")
    if hierarchy_only and args.no_hierarchy:
        parser.error("Hierarchy-only modes cannot be combined with --no-hierarchy.")
    if (hierarchy_only or args.download_models) and args.scene_id:
        parser.error("This mode operates on global/model files; --scene-id does not apply.")
    if args.create_all_tags and not args.sync_hierarchy_only:
        parser.error("--create-all-tags requires --sync-hierarchy-only.")
    if (args.merge_only or args.explain_tag or args.sync_marker_groups_only) and args.output_mode != "merged":
        parser.error("--merge-only, --explain-tag and --sync-marker-groups-only require --output-mode merged.")
    if args.explain_tag and not args.scene_id:
        parser.error("--explain-tag requires at least one --scene-id.")
    if args.prune_empty_tags_only and args.scene_id:
        parser.error("Empty-tag definitions have global scope; --scene-id does not apply to --prune-empty-tags-only.")
    if args.prune_empty_tags and (hierarchy_only or args.download_models or args.explain_tag or args.output_mode != "merged" or args.sync_marker_groups_only):
        parser.error("--prune-empty-tags follows merged tagging, --merge-only or merged cleanup; use --prune-empty-tags-only otherwise.")
    if args.prune_prefixes and not (args.prune_empty_tags or args.prune_empty_tags_only):
        parser.error("--prune-prefixes requires an empty-tag pruning mode.")
    if args.restart and (hierarchy_only or cleanup_only or args.download_models or args.explain_tag or args.merge_only or args.prune_empty_tags_only or args.sync_marker_groups_only):
        parser.error("--restart applies to model inference.")
    if args.abandon_pending and not args.merge_only and not cleanup_only:
        from stash_recovery import run_only
        if args.scene_id:
            parser.error("Standalone abandonment covers pending plans for this Stash/state DB; --scene-id does not apply.")
        validate_config(scene_work=False)
        return run_only(sys.modules[__name__], args)
    if args.sync_marker_groups_only:
        from stash_marker_groups import run_backfill
        validate_config(scene_work=False)
        return run_backfill(sys.modules[__name__], args)
    if args.prune_empty_tags_only:
        from stash_prune import run_prune
        validate_config(scene_work=False)
        return run_prune(sys.modules[__name__], args)
    if args.model_report is not None:
        from stash_model_report import run as run_model_report
        return run_model_report(sys.modules[__name__], args)
    settings = load_model_settings(args.model_config, args.models, args.device, args.offline)
    print("Selected models: " + ", ".join(f"{item.source.title} ({item.source.prefix})" for item in settings))
    if args.download_models:
        for config in settings:
            load_vocabulary(config)
            ensure_files(config, weight_files(config))
            if config.source.key == "joytag":
                from stash_joytag import loader_path
                loader_path(config.download_missing)
            print(f"{config.source.title}: files ready in {config.model_dir}.")
        return 0
    from stash_temporal import load_policy
    TEMPORAL_POLICY = load_policy(sys.modules[__name__], args.temporal_policy, not args.no_temporal) if not hierarchy_only else TemporalPolicy(enabled=False)
    validate_config(scene_work=not hierarchy_only and not cleanup_only and not args.explain_tag)
    empty_policies = {name: frozenset() for name in SOURCES}
    core_policies = load_exclusion_policy(args.marker_core_exclusions, "excluded_core_tags") if CREATE_MARKERS and not hierarchy_only else empty_policies
    scene_policies = load_exclusion_policy(args.scene_tag_exclusions, "excluded_scene_tags")
    if args.output_mode == "merged":
        import stash_merge
        app = sys.modules[__name__]
        if args.explain_tag:
            return stash_merge.explain(app, args, core_policies, scene_policies)
        if cleanup_only:
            return stash_merge.run_cleanup(app, settings, args, core_policies, scene_policies)
        if args.merge_only:
            return stash_merge.run_merged(app, settings, args, {}, stash_merge.combined_hierarchy(app, args, scene_policies), core_policies, scene_policies)
    if cleanup_only:
        from stash_merge import run_cleanup as run_full_cleanup
        return run_full_cleanup(sys.modules[__name__], settings, args, core_policies, scene_policies)
    catalogs, hierarchies = {}, {}
    # Validate all selected vocabularies and hierarchies before any Stash writes.
    for config in settings:
        source = config.source
        vocabulary = load_vocabulary(config)
        catalogs[source.key] = vocabulary
        hierarchy_path = args.hierarchy if source.key == "joytag" else getattr(args, source.key + "_hierarchy")
        if args.no_hierarchy:
            hierarchy = None
        elif hierarchy_path is not None:
            hierarchy = load_tag_hierarchy(hierarchy_path, vocabulary.labels, source)
        else:
            hierarchy = metadata_hierarchy(vocabulary, source)
        hierarchies[source.key] = hierarchy
        print(f"{source.title}: {len(vocabulary.labels):,} labels; "
              f"hierarchy={'off' if hierarchy is None else str(len(hierarchy.category_parents)) + ' categories'}; "
              f"scene threshold={config.scene_threshold}, marker threshold={config.marker_threshold}.")
    merged_hierarchy = stash_merge.combined_hierarchy(app, args, scene_policies) if args.output_mode == "merged" else None
    if merged_hierarchy:
        print(f"AI: {len(merged_hierarchy.leaf_parents):,} unique labels; {len(merged_hierarchy.category_parents)} shared categories.")
    if args.validate_hierarchy:
        print("Vocabularies/hierarchies validated. No Stash requests or model weights loaded.")
        return 0
    if args.output_mode == "merged":
        if args.sync_hierarchy_only:
            return stash_merge.run_hierarchy(app, args, catalogs, merged_hierarchy, scene_policies)
        return stash_merge.run_merged(app, settings, args, catalogs, merged_hierarchy, core_policies, scene_policies)
    stash = create_stash_client(DRY_RUN)
    stats = Counter()
    try:
        if any(hierarchies.values()):
            stash.detect_tag_hierarchy()
        if CREATE_MARKERS and not hierarchy_only:
            stash.detect_marker_capabilities()
            if MARKER_USE_END_TIME and not stash.has_end_time:
                print("Stash has no marker end-time field; creating point markers.")
        print("DRY RUN: all Stash changes below are planned." if DRY_RUN else "Synchronizing selected models.")
        for config in settings:
            try:
                stats.update(run_model(stash, config, catalogs[config.source.key], hierarchies[config.source.key],
                                       args, core_policies[config.source.key], scene_policies[config.source.key]))
            except Exception as exc:
                stats["errors"] += 1
                args.report.issue('load model', config.source.key, None, exc, safe_error(exc, api_key=STASH_API_KEY))
        if not hierarchy_only and not stats["errors"]:
            from stash_exclusions import prune_excluded_definitions, selected_policies
            stats.update(prune_excluded_definitions(sys.modules[__name__], stash,
                selected_policies(scene_policies, {config.source.key for config in settings}), args.report.detail))
    finally:
        stash.close()
    args.report.finish('Done' + (' (planned)' if DRY_RUN else ''), stats)
    if stats["errors"] and not DRY_RUN:
        print("Failed scenes may have partial writes. Correct the error and rerun to reconcile them.")
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {safe_error(exc, api_key=STASH_API_KEY)}", file=sys.stderr)
        raise SystemExit(1)
