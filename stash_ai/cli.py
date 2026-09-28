#!/usr/bin/env python
r"""
Stash AI: multi-model scene tags, source evidence and timestamped markers.
Python 3.14.7. Launch with the repository-root stash_ai.py entry point.
See docs/README_multimodel.md for configuration, migration and all options.

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

import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

INSTALL_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_MODE = "merged"
DEFAULT_RESUME = True
CHECKPOINT_BATCHES = 8
PREFETCH_BATCHES = 4  # Current batch plus up to four CPU-prepared batches; 0 disables.
RAM_CACHE_GB = 24.0  # Shared result/index/media cache; allocated on demand.
RAM_BUDGET_GB = 32.0  # Process target; reserves 25% for models and active work.
PREPARE_WORKERS = 4  # CPU transforms only. One model/inference call at a time.
MERGE_STATE_FILE = (INSTALL_ROOT / "tagger_sources.sqlite3")

from stash_ai.taggers import (JOYTAG, SOURCES, ModelSource, ModelSettings, Vocabulary,
                           select_models, load_vocabulary, load_tagger, ensure_files, weight_files)
from stash_ai.temporal import TemporalPolicy
from stash_ai.domain import Cue, FramePrediction, MarkerSpec, TagHierarchy
from stash_ai.client import StashClient, safe_error
from stash_ai.tags import TagResolver
from stash_ai.config import (
    vocabulary_digest, load_unique_yaml, load_exclusion_policy,
    load_model_settings as _load_model_settings, metadata_hierarchy, load_tag_hierarchy,
)
from stash_ai.vtt import TIMESTAMP_RE, RECT_RE, timestamp_seconds, format_time, parse_vtt_cues

DEFAULT_TEMPORAL = True
TEMPORAL_POLICY = TemporalPolicy()
TEMPORAL_POLICY_FILE = (INSTALL_ROOT / "temporal_policy.yml")


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

# Marker ownership identifiers are defined in stash_ai/taggers.py and stash_ai/merge.py.
MARKER_MATCH_TOLERANCE_SECONDS = 0.25

CREATE_MISSING = True
DRY_RUN = False
SKIP_EXISTING_OUTPUTS = False  # Checked per model and enabled output type.
MAX_FRAMES_PER_SCENE = 0  # 0 = all VTT cues. A cap can miss short events.
JOYTAG_DEFAULT_BATCH_SIZE = 16
SCENE_PAGE_SIZE = 100
REQUEST_TIMEOUT = (5, 600)
HTTP_RETRIES = 3  # Media GETs and read-only GraphQL retry transient failures; GraphQL writes never retry.

JOYTAG_HIERARCHY_FILE = (INSTALL_ROOT / "joytag_hierarchy.yml")
MARKER_CORE_EXCLUSIONS_FILE = (INSTALL_ROOT / "marker_core_exclusions.yml")
if not MARKER_CORE_EXCLUSIONS_FILE.exists():
    # Existing installations can retain the old filename or pass an explicit path.
    MARKER_CORE_EXCLUSIONS_FILE = MARKER_CORE_EXCLUSIONS_FILE.with_name("joytag_marker_core_exclusions.yml")
SCENE_TAG_EXCLUSIONS_FILE = (INSTALL_ROOT / "scene_tag_exclusions.yml")
MODELS_CONFIG_FILE = (INSTALL_ROOT / "tagger_models.yml")

def load_model_settings(path: Path, chosen: list[str] | None, device: str | None,
                        offline: bool = False) -> list[ModelSettings]:
    """Load model settings using CLI defaults while keeping configuration parsing isolated."""
    return _load_model_settings(
        path, chosen, device, offline,
        joytag_scene_threshold=JOYTAG_DEFAULT_SCENE_THRESHOLD,
        default_marker_threshold=DEFAULT_MARKER_THRESHOLD,
        joytag_batch_size=JOYTAG_DEFAULT_BATCH_SIZE,
    )


JOYTAG_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
JOYTAG_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def evenly_sample(items: list, count: int) -> list:
    """Compatibility wrapper for deterministic cue sampling."""
    from stash_ai.inference import evenly_sample as _evenly_sample
    return _evenly_sample(items, count)


def create_stash_client(dry_run: bool) -> StashClient:
    """Bind the current CLI settings once for each client instance."""
    return StashClient(dry_run, url=STASH_URL, api_key=STASH_API_KEY,
                       request_timeout=REQUEST_TIMEOUT, http_retries=HTTP_RETRIES,
                       scene_page_size=SCENE_PAGE_SIZE, create_markers=CREATE_MARKERS)


def fetch_cues(stash: StashClient, paths: dict) -> tuple[list[Cue], int]:
    """Compatibility wrapper for VTT cue loading."""
    from stash_ai.inference import fetch_cues as _fetch_cues
    return _fetch_cues(sys.modules[__name__], stash, paths)


def iter_sprite_frames(stash: StashClient, paths: dict,
                       cues: list[Cue]) -> Iterator[tuple[Cue, Any]]:
    """Compatibility wrapper for sprite-frame iteration."""
    from stash_ai.inference import iter_sprite_frames as _iter_sprite_frames
    return _iter_sprite_frames(stash, paths, cues)


def prepare_image(image: Any, target_size: int) -> Any:
    """Compatibility wrapper for JoyTag image preparation."""
    from stash_ai.inference import prepare_image as _prepare_image
    return _prepare_image(sys.modules[__name__], image, target_size)


def select_scores(scores: list[float], labels: list[str], threshold: float,
                  limit: int) -> list[tuple[str, float]]:
    """Compatibility wrapper for score filtering."""
    from stash_ai.inference import select_scores as _select_scores
    return _select_scores(scores, labels, threshold, limit)


def predict_scene(tagger: Any, stash: StashClient, paths: dict, cues: list[Cue],
                  scene_exclusions: frozenset[str] = frozenset(), *, raw_scene: bool = False,
                  checkpoint=None, prefetch: int = 0, analysis: dict | None = None, report=None,
                  prepare_workers: int = 1
                  ) -> tuple[list[tuple[str, float]], list[FramePrediction]]:
    """Compatibility wrapper around the local inference pipeline."""
    from stash_ai.inference import predict_scene as _predict_scene
    return _predict_scene(
        sys.modules[__name__], tagger, stash, paths, cues, scene_exclusions,
        raw_scene=raw_scene, checkpoint=checkpoint, prefetch=prefetch,
        analysis=analysis, report=report, prepare_workers=prepare_workers,
    )


def build_markers(frames: list[FramePrediction], core_exclusions: frozenset[str], *,
                  threshold=None, seeds=None, identity=None, cache=None) -> tuple[list[MarkerSpec], int]:
    """Group eligible cores independently; exclusions may still be extras."""
    if TEMPORAL_POLICY.enabled:
        from stash_ai.temporal import build_markers as temporal_markers
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
    from stash_ai.exclusions import filter_spec
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
    """Compatibility wrapper around separate-output model orchestration."""
    from stash_ai.separate_run import run_model as _run_model
    return _run_model(
        sys.modules[__name__],
        stash,
        config,
        vocabulary,
        hierarchy,
        args,
        core_exclusions,
        scene_exclusions,
    )

def main(argv: list[str] | None = None) -> int:
    global DRY_RUN, APPLY_SCENE_TAGS, CREATE_MARKERS, TEMPORAL_POLICY
    from stash_ai.cli_args import mode_flags, parse_args
    from stash_ai.cli_modes import run_database_command, run_standalone_command

    app = sys.modules[__name__]
    args = parse_args(app, argv)
    result = run_database_command(app, args)
    if result is not None:
        return result

    from stash_ai.runtime import RunReport
    from tqdm import tqdm
    args.report = RunReport(args.verbose, tqdm.write)
    DRY_RUN = DRY_RUN or args.dry_run
    if args.markers_only or args.cleanup_markers_only or args.sync_marker_groups_only:
        APPLY_SCENE_TAGS, CREATE_MARKERS = False, True
    elif args.scene_tags_only or args.cleanup_scene_tags_only:
        APPLY_SCENE_TAGS, CREATE_MARKERS = True, False
    elif args.cleanup_only:
        APPLY_SCENE_TAGS, CREATE_MARKERS = True, True
    hierarchy_only, cleanup_only = mode_flags(args)
    result = run_standalone_command(app, args, cleanup_only=cleanup_only)
    if result is not None:
        return result
    settings = load_model_settings(args.model_config, args.models, args.device, args.offline)
    print("Selected models: " + ", ".join(f"{item.source.title} ({item.source.prefix})" for item in settings))
    if args.download_models:
        for config in settings:
            load_vocabulary(config)
            ensure_files(config, weight_files(config))
            if config.source.key == "joytag":
                from stash_ai.joytag import loader_path
                loader_path(config.download_missing)
            print(f"{config.source.title}: files ready in {config.model_dir}.")
        return 0
    from stash_ai.temporal import load_policy
    TEMPORAL_POLICY = load_policy(sys.modules[__name__], args.temporal_policy, not args.no_temporal) if not hierarchy_only else TemporalPolicy(enabled=False)
    validate_config(scene_work=not hierarchy_only and not cleanup_only and not args.explain_tag)
    empty_policies = {name: frozenset() for name in SOURCES}
    core_policies = load_exclusion_policy(args.marker_core_exclusions, "excluded_core_tags") if CREATE_MARKERS and not hierarchy_only else empty_policies
    scene_policies = load_exclusion_policy(args.scene_tag_exclusions, "excluded_scene_tags")
    if args.output_mode == "merged":
        from stash_ai import merge as stash_merge
        if args.explain_tag:
            return stash_merge.explain(app, args, core_policies, scene_policies)
        if cleanup_only:
            return stash_merge.run_cleanup(app, settings, args, core_policies, scene_policies)
        if args.merge_only:
            return stash_merge.run_merged(app, settings, args, {}, stash_merge.combined_hierarchy(app, args, scene_policies), core_policies, scene_policies)
    if cleanup_only:
        from stash_ai.merge import run_cleanup as run_full_cleanup
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
            from stash_ai.exclusions import prune_excluded_definitions, selected_policies
            stats.update(prune_excluded_definitions(sys.modules[__name__], stash,
                selected_policies(scene_policies, {config.source.key for config in settings}), args.report.detail))
    finally:
        stash.close()
    args.report.finish('Done' + (' (planned)' if DRY_RUN else ''), stats)
    if stats["errors"] and not DRY_RUN:
        print("Failed scenes may have partial writes. Correct the error and rerun to reconcile them.")
    return 1 if stats["errors"] else 0


def run() -> int:
    """CLI exit handling shared by the root launcher."""
    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {safe_error(exc, api_key=STASH_API_KEY)}", file=sys.stderr)
        return 1
