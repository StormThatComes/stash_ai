"""Model asset management and vocabulary parsing."""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from stash_ai import models


CATEGORY_NAMES = {0: "General", 1: "Artist", 3: "Copyright", 4: "Character", 5: "Meta", 9: "Rating"}


def metadata_files(source: models.ModelSource) -> tuple[str, ...]:
    if source.key == "joytag":
        return ("top_tags.txt",)
    if source.key == "wd":
        return ("selected_tags.csv",)
    if source.key == "oppai":
        return ("vocabulary.json", "selected_tags.csv", "config.json", "preprocessing.json")
    return ("tag_to_id.json", "tag_category.json", "config.json")


def weight_files(settings: models.ModelSettings) -> tuple[str, ...]:
    if settings.source.key == "joytag":
        return ("config.json", "model.safetensors")
    return ("model.onnx",)


def ensure_files(settings: models.ModelSettings, filenames: tuple[str, ...]) -> None:
    """Download only the chosen model's missing assets, at a pinned revision."""
    missing = [name for name in filenames if not (settings.model_dir / name).is_file()]
    if not missing:
        return
    if not settings.download_missing:
        raise FileNotFoundError(f"{settings.source.title}: missing {', '.join(missing)} in {settings.model_dir}. "
                                "Configure model_dir or enable download_missing.")
    from huggingface_hub import hf_hub_download
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        remote = "/".join(part for part in (settings.source.subdirectory, name) if part)
        print(f"Downloading {settings.source.key}/{name}...")
        cached = hf_hub_download(repo_id=settings.source.repo, filename=remote, revision=settings.source.revision)
        fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=settings.model_dir)
        os.close(fd)
        try:
            shutil.copyfile(cached, temporary)
            os.replace(temporary, settings.model_dir / name)
        finally:
            Path(temporary).unlink(missing_ok=True)


def validate_labels(labels: list[str], source: models.ModelSource) -> None:
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"{source.title}: vocabulary is empty or contains duplicate labels.")
    if any(not isinstance(label, str) or not label or label != label.strip() or "\n" in label or "\r" in label
           for label in labels):
        raise ValueError(f"{source.title}: invalid raw label in vocabulary.")
    if source.owner_tag in {source.prefix + label for label in labels}:
        raise ValueError(f"{source.title}: a model label collides with the marker ownership tag.")


def load_vocabulary(settings: models.ModelSettings) -> models.Vocabulary:
    source = settings.source
    ensure_files(settings, metadata_files(source))
    if source.key == "oppai":
        return load_oppai_vocabulary(settings)
    categories = {}
    image_size = 448
    if source.key == "joytag":
        labels = [line.strip() for line in (settings.model_dir / "top_tags.txt").read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
    elif source.key == "wd":
        with (settings.model_dir / "selected_tags.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"name", "category"} <= set(reader.fieldnames or []):
                raise ValueError("WD selected_tags.csv requires name and category columns.")
            rows = list(reader)
        # CSV row position is the output index; tag_id is a Danbooru identifier.
        labels = [row["name"] for row in rows]
        categories = {row["name"]: CATEGORY_NAMES.get(int(row["category"]), f"Category {row['category']}")
                      for row in rows}
    else:
        mapping = json.loads((settings.model_dir / "tag_to_id.json").read_text(encoding="utf-8-sig"))
        if (not isinstance(mapping, dict) or not mapping or any(type(index) is not int for index in mapping.values())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError("DTQ tag_to_id.json must contain unique contiguous integer indices starting at zero.")
        labels = sorted(mapping, key=mapping.get)
        config = json.loads((settings.model_dir / "config.json").read_text(encoding="utf-8-sig"))
        if type(config.get("num_classes")) is not int or config["num_classes"] != len(labels):
            raise ValueError("DTQ config.num_classes does not match tag_to_id.json.")
        image_size = config.get("image_size")
        if type(image_size) is not int or image_size < 1:
            raise ValueError("DTQ config.image_size must be a positive integer.")
        raw_categories = json.loads((settings.model_dir / "tag_category.json").read_text(encoding="utf-8-sig"))
        if not isinstance(raw_categories, dict):
            raise ValueError("DTQ tag_category.json must be a mapping.")
        for label in labels:
            categories[label] = CATEGORY_NAMES.get(raw_categories.get(label), "Uncategorized")
    validate_labels(labels, source)
    return models.Vocabulary(labels, categories, image_size)


def load_oppai_vocabulary(settings: models.ModelSettings) -> models.Vocabulary:
    """Keep the original ONNX indices while removing only PAD/UNK output slots."""
    def read(name):
        return json.loads((settings.model_dir / name).read_text(encoding="utf-8-sig"))
    config, preprocessing, vocabulary = read("config.json"), read("preprocessing.json"), read("vocabulary.json")
    if not all(isinstance(value, dict) for value in (config, preprocessing, vocabulary)):
        raise ValueError("OppaiOracle metadata must contain JSON objects.")
    mapping = vocabulary.get("tag_to_index")
    if (not isinstance(mapping, dict) or not mapping
            or any(type(index) is not int for index in mapping.values())
            or set(mapping.values()) != set(range(len(mapping)))):
        raise ValueError("OppaiOracle tag_to_index must contain unique contiguous integer indices.")
    all_labels = sorted(mapping, key=mapping.get)
    validate_labels(all_labels, settings.source)
    if vocabulary.get("index_to_tag") != {str(i): label for i, label in enumerate(all_labels)}:
        raise ValueError("OppaiOracle index_to_tag disagrees with tag_to_index.")
    if all_labels[:2] != ["<PAD>", "<UNK>"]:
        raise ValueError("OppaiOracle expected PAD/UNK at output indices 0 and 1.")
    if (config.get("format") != "onnx" or type(config.get("num_labels")) is not int
            or config["num_labels"] != len(all_labels) or config.get("image_size") != 448):
        raise ValueError("OppaiOracle requires the V1.1 ONNX configuration and matching vocabulary.")
    expected = {"image_size": 448, "patch_size": 16, "num_channels": 3,
                "color_order": "RGB", "resize_mode": "letterbox", "pad_color_rgb": [114, 114, 114],
                "normalize_mean": [0.5, 0.5, 0.5], "normalize_std": [0.5, 0.5, 0.5],
                "input_dtype": "float32", "input_layout": "BCHW"}
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise ValueError(f"Unsupported OppaiOracle preprocessing.{key}; use the V1.1_onnx files.")
    with (settings.model_dir / "selected_tags.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"name", "category"} <= set(reader.fieldnames or []):
            raise ValueError("OppaiOracle selected_tags.csv requires name and category columns.")
        rows = list(reader)
    if [row["name"] for row in rows] != all_labels:
        raise ValueError("OppaiOracle selected_tags.csv does not match the vocabulary output order.")
    # Do not discard emoticon tags such as <o>_<o>, ratings, or any category.
    indices = tuple(i for i, label in enumerate(all_labels) if label not in {"<PAD>", "<UNK>"})
    labels = [all_labels[i] for i in indices]
    categories = {all_labels[i]: CATEGORY_NAMES.get(int(rows[i]["category"]), f"Category {rows[i]['category']}")
                  for i in indices}
    return models.Vocabulary(labels, categories, 448, len(all_labels), indices)
