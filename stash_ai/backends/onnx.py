"""ONNX inference adapter over explicit preprocessing and session helpers."""
from __future__ import annotations

import gc
from typing import Any, TYPE_CHECKING

from stash_ai.backends.vocabulary import ensure_files
from stash_ai.backends.onnx_preprocessing import prepare_oppai, prepare_wd, prepare_dtq
from stash_ai.backends.onnx_session import _open_session, _inspect_model

if TYPE_CHECKING:
    from stash_ai.models import ModelSettings, Vocabulary


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
