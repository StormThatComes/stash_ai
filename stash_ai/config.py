"""Configuration, exclusions and hierarchy loading for the CLI."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from stash_ai.domain import TagHierarchy
from stash_ai.models import SOURCES, ModelSource, ModelSettings, select_models


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
                    or label.startswith("ai:") or any(label.startswith(source.code + ':') for source in SOURCES.values())):
                raise ValueError(f"Invalid excluded tag {label!r}; use a raw label without jt:/wd:/dtq:/op:/ai:.")
            from stash_ai.evidence.labels import raw_label
            raw_label(label)
        if len(entries) != len(set(entries)):
            raise ValueError(f"Duplicate label in {context}.")
        return frozenset(entries)
    common = checked(data[key], key)
    additions = data.get("by_model", {})
    if not isinstance(additions, dict) or set(additions) - set(SOURCES):
        raise ValueError("by_model must contain only joytag, wd, dtq, and/or oppai keys.")
    return {name: common | checked(additions.get(name, []), f"by_model.{name}") for name in SOURCES}

def load_model_configuration(path: Path):
    data = load_unique_yaml(path, "Model configuration")
    allowed = {"version", "enabled_models", "device", "download_missing", "prefetch_batches", "result_prefetch_batches",
               "prepare_workers", "merge_workers", "model_report_workers", *SOURCES}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Unknown model configuration keys.")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Model configuration version must be 1.")
    if 'prefetch_batches' in data and (type(data['prefetch_batches']) is not int or
                                      not 0 <= data['prefetch_batches'] <= 128):
        raise ValueError('prefetch_batches must be an integer in 0..128.')
    if 'result_prefetch_batches' in data and (type(data['result_prefetch_batches']) is not int or
                                             not 1 <= data['result_prefetch_batches'] <= 128):
        raise ValueError('result_prefetch_batches must be an integer in 1..128.')
    for name in ('prepare_workers', 'merge_workers', 'model_report_workers'):
        if name in data and (type(data[name]) is not int or not 1 <= data[name] <= 16):
            raise ValueError(f'{name} must be an integer in 1..16.')
    return data


def load_model_settings(path: Path, chosen: list[str] | None, device: str | None,
                        offline: bool = False) -> list[ModelSettings]:
    data = load_model_configuration(path)
    if chosen is None and "enabled_models" not in data:
        raise ValueError("enabled_models is required when --models is omitted.")
    selected = select_models(chosen if chosen is not None else data["enabled_models"])
    settings = []
    for name in SOURCES:
        values = data.get(name, {})
        if not isinstance(values, dict) or set(values) - {
                "model_dir", "scene_threshold", "marker_threshold", "batch_size",
                "scene_cache_floor", "marker_cache_floor"}:
            raise ValueError(f"Invalid {name} settings in {path.name}.")
        if name not in selected:
            continue
        required = {"scene_threshold", "marker_threshold", "batch_size"}
        if missing := required - values.keys():
            raise ValueError(f"Missing {name} settings: {', '.join(sorted(missing))}.")
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
            scene_threshold=values["scene_threshold"],
            marker_threshold=values["marker_threshold"],
            batch_size=values["batch_size"],
            scene_cache_floor=values.get("scene_cache_floor"),
            marker_cache_floor=values.get("marker_cache_floor"),
            device=device or data.get("device", "auto"),
            download_missing=False if offline else data.get("download_missing", True))
        config.validate()
        settings.append(config)
    return sorted(settings, key=lambda item: selected.index(item.source.key))

def load_tag_hierarchy(path: Path) -> TagHierarchy:
    if not path.is_file():
        raise FileNotFoundError(f"Hierarchy YAML missing: {path}. Place it beside the script or use --no-hierarchy.")
    data = load_unique_yaml(path, "Hierarchy")
    return hierarchy_from_data(data)


def hierarchy_from_data(data):
    """Validate the complete shared AI taxonomy, independent of model selection."""
    from stash_ai.evidence.labels import AI, raw_label
    from stash_ai.tools import MIGRATION_BRIDGE_HINT
    allowed = {"version", "source", "groups"}
    if isinstance(data, dict) and type(data.get('version')) is int and data['version'] == 1:
        raise ValueError('Hierarchy YAML requires version: 2. ' + MIGRATION_BRIDGE_HINT)
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Hierarchy YAML must be a mapping using only: " + ", ".join(sorted(allowed)))
    if type(data.get("version")) is not int or data["version"] != 2:
        raise ValueError('Hierarchy YAML requires version: 2. ' + MIGRATION_BRIDGE_HINT)
    prefix, root_name = 'ai:category:', 'ai:category:AI'
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Hierarchy groups must be a non-empty mapping of category paths to label lists.")
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
            raw_label(label)
            leaves.setdefault(label, set()).add(parent)
    collisions = set(categories) & ({AI.prefix + label for label in leaves} | {AI.owner_tag})
    if collisions:
        raise ValueError(f"Hierarchy categories collide with model/marker tags: {sorted(collisions)!r}")
    source_metadata = data.get("source", {})
    if not isinstance(source_metadata, dict):
        raise ValueError("Hierarchy source metadata must be a mapping.")
    labels = sorted(leaves)
    if source_metadata.get("tag_count", len(labels)) != len(labels):
        raise ValueError("Hierarchy source.tag_count does not match the shared vocabulary.")
    if source_metadata.get("vocabulary_sha256", vocabulary_digest(labels)) != vocabulary_digest(labels):
        raise ValueError("Hierarchy source.vocabulary_sha256 does not match the shared vocabulary.")
    return TagHierarchy({tag: tuple(sorted(parents)) for tag, parents in leaves.items()}, categories, prefix, AI.hierarchy_description)


def validate_hierarchy_vocabulary(hierarchy: TagHierarchy, labels: list[str], source: ModelSource) -> None:
    """Every selected model label needs a parent; other models' labels are allowed."""
    missing = set(labels) - hierarchy.leaf_parents.keys()
    if missing:
        raise ValueError(
            f"Hierarchy/{source.key} vocabulary mismatch: {len(missing)} unmapped labels "
            f"({', '.join(sorted(missing)[:12])}). Add these labels to the shared hierarchy YAML."
        )
