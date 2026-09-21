"""Local inference adapters. No Stash access and no tag/category allowlist."""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import shutil
import tempfile
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
CATEGORY_NAMES = {0: "General", 1: "Artist", 3: "Copyright", 4: "Character", 5: "Meta", 9: "Rating"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSettings:
    source: ModelSource
    model_dir: Path
    scene_threshold: float = 0.4
    marker_threshold: float = 0.65
    batch_size: int = 4
    device: str = "auto"
    download_missing: bool = True

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        for name in ("scene_threshold", "marker_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{self.source.key}.{name} must be a number between 0 and 1.")
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


def metadata_files(source: ModelSource) -> tuple[str, ...]:
    if source.key == "joytag":
        return ("top_tags.txt",)
    if source.key == "wd":
        return ("selected_tags.csv",)
    if source.key == "oppai":
        return ("vocabulary.json", "selected_tags.csv", "config.json", "preprocessing.json")
    return ("tag_to_id.json", "tag_category.json", "config.json")


def weight_files(settings: ModelSettings) -> tuple[str, ...]:
    if settings.source.key == "joytag":
        return ("config.json", "model.safetensors")
    return ("model.onnx",)


def ensure_files(settings: ModelSettings, filenames: tuple[str, ...]) -> None:
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


def validate_labels(labels: list[str], source: ModelSource) -> None:
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"{source.title}: vocabulary is empty or contains duplicate labels.")
    if any(not isinstance(label, str) or not label or label != label.strip() or "\n" in label or "\r" in label for label in labels):
        raise ValueError(f"{source.title}: invalid raw label in vocabulary.")
    if source.owner_tag in {source.prefix + label for label in labels}:
        raise ValueError(f"{source.title}: a model label collides with the marker ownership tag.")


def load_vocabulary(settings: ModelSettings) -> Vocabulary:
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
        categories = {row["name"]: CATEGORY_NAMES.get(int(row["category"]), f"Category {row['category']}") for row in rows}
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
    return Vocabulary(labels, categories, image_size)


def load_oppai_vocabulary(settings: ModelSettings) -> Vocabulary:
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
    return Vocabulary(labels, categories, 448, len(all_labels), indices)


def prepare_oppai(image: Any, size: int = 448) -> tuple[Any, Any]:
    """Author's V1.1 letterbox: no upscaling; True in mask means padding."""
    import numpy as np
    from PIL import Image, ImageOps
    transposed = ImageOps.exif_transpose(image)
    if transposed.mode in ("RGBA", "LA") or "transparency" in transposed.info:
        rgb = Image.new("RGB", transposed.size, (114, 114, 114))
        rgba = transposed.convert("RGBA")
        alpha = rgba.getchannel("A")
        foreground = rgba.convert("RGB")
        rgb.paste(foreground, mask=alpha)
        foreground.close()
        alpha.close()
        rgba.close()
    else:
        rgb = transposed.convert("RGB")
    scale = min(size / rgb.width, size / rgb.height, 1.0)
    width, height = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
    resized = rgb.resize((width, height), Image.Resampling.BILINEAR) if (width, height) != rgb.size else rgb
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top:top + height, left:left + width] = np.asarray(resized, dtype=np.uint8)
    mask = np.ones((size, size), dtype=np.bool_)
    mask[top:top + height, left:left + width] = False
    pixels = (canvas.astype(np.float32) / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
    if resized is not rgb:
        resized.close()
    rgb.close()
    transposed.close()
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)), mask


def prepare_wd(image: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    rgb = background.convert("RGB")
    side = max(rgb.size)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    resized = padded.resize((size, size), Image.Resampling.BICUBIC) if side != size else padded
    result = np.ascontiguousarray(np.asarray(resized, dtype=np.float32)[:, :, ::-1])
    for opened in (rgba, background, rgb, padded, resized):
        opened.close()
    return result


def prepare_dtq(image: Any, size: int) -> Any:
    import numpy as np
    from PIL import Image
    rgb = image.convert("RGB")
    scale = size / max(rgb.size)
    width, height = max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))
    resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(resized, ((size - width) // 2, (size - height) // 2))
    data = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    data = (data - np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]) / np.asarray(
        IMAGENET_STD, dtype=np.float32)[:, None, None]
    for opened in (rgb, resized, padded):
        opened.close()
    return np.ascontiguousarray(data, dtype=np.float32)


def validate_scores(scores: Any, count: int, labels: list[str], title: str) -> Any:
    import numpy as np
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (count, len(labels)):
        raise RuntimeError(f"{title}: output shape {values.shape} != {(count, len(labels))}.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise RuntimeError(f"{title}: non-finite or invalid probabilities; scene sync was not started.")
    return values


class OnnxTagger:
    def __init__(self, settings: ModelSettings, vocabulary: Vocabulary):
        import onnxruntime as ort
        self.settings, self.vocabulary = settings, vocabulary
        self.labels = vocabulary.labels
        ensure_files(settings, ("model.onnx",))
        use_cuda = settings.device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
        if settings.device == "cuda" and not use_cuda:
            raise RuntimeError("CUDA provider unavailable. Install onnxruntime-gpu (not both ORT packages).")
        if use_cuda:
            try:
                import torch
            except ImportError:
                pass
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        def create_session(selected_providers):
            path = str(settings.model_dir / "model.onnx")
            if settings.source.key != "oppai":
                return ort.InferenceSession(path, providers=selected_providers)
            # Author's workaround: full optimization can insert unsupported
            # MemcpyToHost nodes on the bool padding-mask path in some ORT builds.
            last_error = None
            for level in (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, ort.GraphOptimizationLevel.ORT_DISABLE_ALL):
                options = ort.SessionOptions()
                options.graph_optimization_level = level
                options.intra_op_num_threads = min(os.cpu_count() or 2, 8)
                try:
                    return ort.InferenceSession(path, sess_options=options, providers=selected_providers)
                except Exception as exc:
                    last_error = exc
            raise RuntimeError("OppaiOracle ONNX initialization failed at both supported optimization levels.") from last_error
        try:
            self.session = create_session(providers)
        except Exception:
            if settings.device != "auto" or not use_cuda:
                raise
            print(f"{settings.source.title}: CUDA initialization failed; using CPU.")
            self.session = create_session(["CPUExecutionProvider"])
        active = self.session.get_providers()
        if settings.device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError("Requested CUDA, but ONNX Runtime initialized only CPU. Check CUDA/cuDNN DLLs.")
        self.device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        self.mask_name = None
        if settings.source.key == "oppai":
            by_name = {value.name: value for value in inputs}
            if len(inputs) != 2 or set(by_name) != {"pixel_values", "padding_mask"}:
                raise ValueError("OppaiOracle requires pixel_values and padding_mask ONNX inputs.")
            inp, mask = by_name["pixel_values"], by_name["padding_mask"]
            if mask.type != "tensor(bool)" or list(mask.shape[1:]) != [448, 448] or mask.shape[0] != inp.shape[0]:
                raise ValueError("OppaiOracle padding_mask must be bool [batch, 448, 448], matching the image batch.")
            self.mask_name = mask.name
        elif len(inputs) == 1:
            inp = inputs[0]
        else:
            raise ValueError(f"{settings.source.title}: expected one image input.")
        if len(outputs) != 1 or inp.type != "tensor(float)" or outputs[0].type != "tensor(float)":
            raise ValueError(f"{settings.source.title}: expected one float32 image input and one tag output.")
        out = outputs[0]
        if settings.source.key == "oppai" and out.name != "probabilities":
            raise ValueError("OppaiOracle requires the probabilities output (sigmoid already in the graph).")
        if len(inp.shape) != 4 or len(out.shape) != 2:
            raise ValueError("ONNX dimensions do not match this tagger adapter.")
        expected = [448, 448, 3] if settings.source.key == "wd" else [3, vocabulary.image_size, vocabulary.image_size]
        if list(inp.shape[1:]) != expected:
            raise ValueError(f"Unexpected ONNX input shape {inp.shape}; expected [batch, {expected}].")
        self.output_count = vocabulary.output_count if vocabulary.output_count is not None else len(self.labels)
        self.output_indices = vocabulary.output_indices
        if self.output_indices is not None and (
                len(self.output_indices) != len(self.labels) or len(set(self.output_indices)) != len(self.labels)
                or any(type(i) is not int or not 0 <= i < self.output_count for i in self.output_indices)):
            raise ValueError("Invalid vocabulary-to-output index mapping.")
        if isinstance(out.shape[1], int) and out.shape[1] != self.output_count:
            raise ValueError("ONNX output size does not match the vocabulary.")
        self.size = 448 if settings.source.key == "wd" else vocabulary.image_size
        self.input_name, self.output_name = inp.name, out.name
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) and inp.shape[0] > 0 else None
        self.batch_size = min(settings.batch_size, self.fixed_batch) if self.fixed_batch else settings.batch_size
        print(f"Loaded {settings.source.title} on {self.device}; {len(self.labels):,} labels, batch={self.batch_size}.")

    def prepare(self, image: Any) -> Any:
        if self.settings.source.key == "oppai":
            return prepare_oppai(image, self.size)
        return prepare_wd(image, self.size) if self.settings.source.key == "wd" else prepare_dtq(image, self.size)

    def predict_batch(self, images: list[Any]) -> Any:
        import numpy as np
        count = len(images)
        if not count or count > self.batch_size:
            raise ValueError("Invalid inference batch length.")
        feed = {self.input_name: np.stack([pair[0] for pair in images] if self.mask_name else images)}
        if self.mask_name:
            feed[self.mask_name] = np.stack([pair[1] for pair in images])
        if self.fixed_batch and count < self.fixed_batch:
            feed = {name: np.concatenate([values, np.repeat(values[-1:], self.fixed_batch - count, axis=0)])
                    for name, values in feed.items()}
        raw = np.asarray(self.session.run([self.output_name], feed)[0], dtype=np.float32)
        if raw.shape != (len(feed[self.input_name]), self.output_count) or not np.isfinite(raw).all():
            raise RuntimeError(f"{self.settings.source.title}: invalid raw model output.")
        if self.settings.source.key == "dtq":
            exp = np.exp(-np.abs(raw))
            raw = np.where(raw >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))
        if self.output_indices is not None:
            raw = raw[:, self.output_indices]
        return validate_scores(raw[:count], count, self.labels, self.settings.source.title)

    def close(self) -> None:
        self.session = None
        gc.collect()


class JoyTagger:
    def __init__(self, settings: ModelSettings, vocabulary: Vocabulary, prepare_image):
        import torch
        self.settings, self.vocabulary = settings, vocabulary
        self.labels, self.batch_size = vocabulary.labels, settings.batch_size
        self.prepare_image = prepare_image
        ensure_files(settings, weight_files(settings))
        from stash_joytag import load_model
        selected = "cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device
        self.device = torch.device("cpu" if selected == "auto" else selected)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA for JoyTag, but torch.cuda.is_available() is false.")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.model = load_model(settings.model_dir, len(self.labels), settings.download_missing)
        self.model.eval().to(self.device)
        print(f"Loaded JoyTag on {self.device}; {len(self.labels):,} labels, batch={self.batch_size}.")

    def prepare(self, image: Any) -> Any:
        return self.prepare_image(image, self.model.image_size)

    def predict_batch(self, images: list[Any]) -> Any:
        import torch
        from contextlib import nullcontext
        batch = torch.stack(images).to(self.device, non_blocking=self.device.type == "cuda")
        amp = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), amp:
            scores = self.model({"image": batch})["tags"].sigmoid().float().cpu().numpy()
        return validate_scores(scores, len(images), self.labels, "JoyTag")

    def close(self) -> None:
        import torch
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_tagger(settings: ModelSettings, vocabulary: Vocabulary, prepare_joytag):
    settings.validate()
    return JoyTagger(settings, vocabulary, prepare_joytag) if settings.source.key == "joytag" else OnnxTagger(settings, vocabulary)
