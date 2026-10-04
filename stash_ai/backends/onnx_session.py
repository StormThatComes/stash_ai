"""ONNX provider selection, session setup and input/output inspection."""
from __future__ import annotations

import os
from dataclasses import dataclass


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
