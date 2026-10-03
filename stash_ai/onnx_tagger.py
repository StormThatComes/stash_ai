"""ONNX model preprocessing, session setup and inference adapter."""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from stash_ai.vocabulary import ensure_files

if TYPE_CHECKING:
    from stash_ai.models import ModelSettings, Vocabulary


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    resized = (
        rgb.resize((width, height), Image.Resampling.BILINEAR)
        if (width, height) != rgb.size else rgb
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top:top + height, left:left + width] = np.asarray(resized, dtype=np.uint8)
    mask = np.ones((size, size), dtype=np.bool_)
    mask[top:top + height, left:left + width] = False
    pixels = (
        canvas.astype(np.float32) / np.float32(255.0) - np.float32(0.5)
    ) / np.float32(0.5)

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
    data = (
        data - np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    ) / np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    for opened in (rgb, resized, padded):
        opened.close()
    return np.ascontiguousarray(data, dtype=np.float32)


@dataclass(frozen=True)
class _OnnxModelIO:
    input_name: str
    output_name: str
    mask_name: str | None
    fixed_batch: int | None
    output_count: int
    output_indices: tuple[int, ...] | None
    size: int
    batch_size: int


def _execution_providers(ort, settings):
    available = ort.get_available_providers()
    use_cuda = settings.device != "cpu" and "CUDAExecutionProvider" in available
    if settings.device == "cuda" and not use_cuda:
        raise RuntimeError(
            "CUDA provider unavailable. Install onnxruntime-gpu (not both ORT packages)."
        )
    if use_cuda:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda else ["CPUExecutionProvider"]
    )
    return providers, use_cuda


def _create_session(ort, settings, providers):
    path = str(settings.model_dir / "model.onnx")
    if settings.source.key != "oppai":
        return ort.InferenceSession(path, providers=providers)

    # Author's workaround: full optimization can insert unsupported
    # MemcpyToHost nodes on the bool padding-mask path in some ORT builds.
    last_error = None
    for level in (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = min(os.cpu_count() or 2, 8)
        try:
            return ort.InferenceSession(path, sess_options=options, providers=providers)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "OppaiOracle ONNX initialization failed at both supported optimization levels."
    ) from last_error


def _open_session(ort, settings):
    providers, use_cuda = _execution_providers(ort, settings)
    try:
        session = _create_session(ort, settings, providers)
    except Exception:
        if settings.device != "auto" or not use_cuda:
            raise
        print(f"{settings.source.title}: CUDA initialization failed; using CPU.")
        session = _create_session(ort, settings, ["CPUExecutionProvider"])

    active = session.get_providers()
    if settings.device == "cuda" and "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "Requested CUDA, but ONNX Runtime initialized only CPU. Check CUDA/cuDNN DLLs."
        )
    device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
    return session, device


def _image_input(inputs, settings):
    if settings.source.key == "oppai":
        by_name = {value.name: value for value in inputs}
        if len(inputs) != 2 or set(by_name) != {"pixel_values", "padding_mask"}:
            raise ValueError(
                "OppaiOracle requires pixel_values and padding_mask ONNX inputs."
            )
        inp, mask = by_name["pixel_values"], by_name["padding_mask"]
        if (
            mask.type != "tensor(bool)"
            or list(mask.shape[1:]) != [448, 448]
            or mask.shape[0] != inp.shape[0]
        ):
            raise ValueError(
                "OppaiOracle padding_mask must be bool [batch, 448, 448], "
                "matching the image batch."
            )
        return inp, mask.name
    if len(inputs) != 1:
        raise ValueError(f"{settings.source.title}: expected one image input.")
    return inputs[0], None


def _output_mapping(vocabulary, out):
    output_count = (
        vocabulary.output_count
        if vocabulary.output_count is not None
        else len(vocabulary.labels)
    )
    output_indices = vocabulary.output_indices
    if output_indices is not None and (
        len(output_indices) != len(vocabulary.labels)
        or len(set(output_indices)) != len(vocabulary.labels)
        or any(type(index) is not int or not 0 <= index < output_count for index in output_indices)
    ):
        raise ValueError("Invalid vocabulary-to-output index mapping.")
    if isinstance(out.shape[1], int) and out.shape[1] != output_count:
        raise ValueError("ONNX output size does not match the vocabulary.")
    return output_count, output_indices


def _inspect_model(session, settings, vocabulary):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    inp, mask_name = _image_input(inputs, settings)
    if len(outputs) != 1 or inp.type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError(
            f"{settings.source.title}: expected one float32 image input and one tag output."
        )

    out = outputs[0]
    if settings.source.key == "oppai" and out.name != "probabilities":
        raise ValueError(
            "OppaiOracle requires the probabilities output (sigmoid already in the graph)."
        )
    if len(inp.shape) != 4 or len(out.shape) != 2:
        raise ValueError("ONNX dimensions do not match this tagger adapter.")

    expected = (
        [448, 448, 3]
        if settings.source.key == "wd"
        else [3, vocabulary.image_size, vocabulary.image_size]
    )
    if list(inp.shape[1:]) != expected:
        raise ValueError(
            f"Unexpected ONNX input shape {inp.shape}; expected [batch, {expected}]."
        )

    output_count, output_indices = _output_mapping(vocabulary, out)
    fixed_batch = (
        inp.shape[0]
        if isinstance(inp.shape[0], int) and inp.shape[0] > 0
        else None
    )
    size = 448 if settings.source.key == "wd" else vocabulary.image_size
    batch_size = (
        min(settings.batch_size, fixed_batch)
        if fixed_batch else settings.batch_size
    )
    return _OnnxModelIO(
        inp.name,
        out.name,
        mask_name,
        fixed_batch,
        output_count,
        output_indices,
        size,
        batch_size,
    )


class OnnxTagger:
    def __init__(self, settings: ModelSettings, vocabulary: Vocabulary):
        import onnxruntime as ort

        self.settings, self.vocabulary = settings, vocabulary
        self.labels = vocabulary.labels
        ensure_files(settings, ("model.onnx",))
        self.session, self.device = _open_session(ort, settings)
        model_io = _inspect_model(self.session, settings, vocabulary)
        self.input_name = model_io.input_name
        self.output_name = model_io.output_name
        self.mask_name = model_io.mask_name
        self.fixed_batch = model_io.fixed_batch
        self.output_count = model_io.output_count
        self.output_indices = model_io.output_indices
        self.size = model_io.size
        self.batch_size = model_io.batch_size
        print(
            f"Loaded {settings.source.title} on {self.device}; "
            f"{len(self.labels):,} labels, batch={self.batch_size}."
        )

    def prepare(self, image: Any) -> Any:
        if self.settings.source.key == "oppai":
            return prepare_oppai(image, self.size)
        return (
            prepare_wd(image, self.size)
            if self.settings.source.key == "wd"
            else prepare_dtq(image, self.size)
        )

    def predict_batch(self, images: list[Any]) -> Any:
        import numpy as np
        from stash_ai.models import validate_scores

        count = len(images)
        if not count or count > self.batch_size:
            raise ValueError("Invalid inference batch length.")

        feed = {
            self.input_name: np.stack(
                [pair[0] for pair in images] if self.mask_name else images
            )
        }
        if self.mask_name:
            feed[self.mask_name] = np.stack([pair[1] for pair in images])
        if self.fixed_batch and count < self.fixed_batch:
            feed = {
                name: np.concatenate(
                    [
                        values,
                        np.repeat(
                            values[-1:],
                            self.fixed_batch - count,
                            axis=0,
                        ),
                    ]
                )
                for name, values in feed.items()
            }

        raw = np.asarray(
            self.session.run([self.output_name], feed)[0],
            dtype=np.float32,
        )
        if (
            raw.shape != (len(feed[self.input_name]), self.output_count)
            or not np.isfinite(raw).all()
        ):
            raise RuntimeError(
                f"{self.settings.source.title}: invalid raw model output."
            )
        if self.settings.source.key == "dtq":
            exp = np.exp(-np.abs(raw))
            raw = np.where(raw >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))
        if self.output_indices is not None:
            raw = raw[:, self.output_indices]
        return validate_scores(
            raw[:count],
            count,
            self.labels,
            self.settings.source.title,
        )

    def close(self) -> None:
        self.session = None
        gc.collect()
