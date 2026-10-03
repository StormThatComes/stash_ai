"""Model sources, settings, vocabularies and shared prediction validation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSource:
    key: str
    title: str
    prefix: str
    owner_tag: str
    title_prefix: str
    hierarchy_description: str
    repo: str = ""
    revision: str = ""
    subdirectory: str = ""


JOYTAG = ModelSource("joytag", "JoyTag", "jt:", "jt:marker:joytag", "[JoyTag] ",
                     "[JoyTag hierarchy v1]", "fancyfeast/joytag",
                     "6b7f16331a6ccf0fdce37d5a9564715f6e772b22")
WD = ModelSource("wd", "WD EVA02-Large v3", "wd:", "wd:marker:wd", "[WD] ",
                 "[WD hierarchy v1]", "SmilingWolf/wd-eva02-large-tagger-v3",
                 "b25b82a03f7282e41aa2f257a52c7583b710bd1c")
DTQ = ModelSource("dtq", "DanbooruTagQuery L/16", "dtq:", "dtq:marker:dtq", "[DTQ] ",
                  "[DTQ hierarchy v1]", "realphongha/danbooru-tag-query",
                  "d2f8da27a23db4adf95c3e663af183b92784ad3e", "models/DanbooruTagQuery_l16_448x448")
OPPAI = ModelSource("oppai", "OppaiOracle V1.1", "op:", "op:marker:oppai", "[OppaiOracle] ",
                    "[OppaiOracle hierarchy v1]", "Grio43/OppaiOracle",
                    "96992fa30568c386e9fe7c8a1a68f798a3202c09", "V1.1_onnx")
SOURCES = {source.key: source for source in (JOYTAG, WD, DTQ, OPPAI)}


@dataclass(frozen=True)
class ModelSettings:
    source: ModelSource
    model_dir: Path
    scene_threshold: float = 0.4
    marker_threshold: float = 0.65
    batch_size: int = 4
    device: str = "auto"
    download_missing: bool = True
    scene_cache_floor: float | None = None
    marker_cache_floor: float | None = None

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        for name in ("scene_threshold", "marker_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{self.source.key}.{name} must be a number between 0 and 1.")
        for name in ("scene_cache_floor", "marker_cache_floor"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{self.source.key}.{name} must be null or a number between 0 and 1.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError(f"{self.source.key}.batch_size must be a positive integer.")
        if type(self.download_missing) is not bool:
            raise ValueError("download_missing must be true or false.")


@dataclass(frozen=True)
class Vocabulary:
    labels: list[str]
    categories: dict[str, str]
    image_size: int = 448
    # Some classifiers have technical output slots that must never become tags.
    output_count: int | None = None
    output_indices: tuple[int, ...] | None = None


def select_models(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise ValueError("Select joytag, wd, dtq, oppai, or all.")
    names = [name.strip().lower() for value in values for name in value.split(",")]
    if "all" in names:
        if names != ["all"]:
            raise ValueError("Use all on its own, or list individual model names.")
        return list(SOURCES)
    unknown = set(names) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}. Choose joytag, wd, dtq, oppai, or all.")
    return list(dict.fromkeys(names))


def validate_scores(scores: Any, count: int, labels: list[str], title: str) -> Any:
    import numpy as np
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (count, len(labels)):
        raise RuntimeError(f"{title}: output shape {values.shape} != {(count, len(labels))}.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise RuntimeError(f"{title}: non-finite or invalid probabilities; scene sync was not started.")
    return values
