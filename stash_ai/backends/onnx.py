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
        from stash_ai.backends.profiling import active
        from tempfile import TemporaryDirectory
        self.report = active()
        self.profile_directory = TemporaryDirectory(prefix='stash-ort-profile-') if self.report is not None else None
        try:
            if self.profile_directory is None:
                self.session, self.device = _open_session(ort, settings)
            else:
                from pathlib import Path
                try:
                    self.session, self.device = _open_session(ort, settings,
                        profile_prefix=str(Path(self.profile_directory.name)/'trace'))
                except Exception as error:
                    def failed():
                        raise error
                    self.report.optional('backend_onnx_profile_setup', failed)
                    self.profile_directory.cleanup()
                    self.profile_directory = None
                    self.session, self.device = _open_session(ort, settings)
        except BaseException:
            if self.profile_directory is not None:
                from stash_ai.runtime.cleanup import cleanup
                cleanup([self.profile_directory.cleanup], report=self.report)
            raise
        try:
            model_io = _inspect_model(self.session, settings, vocabulary)
        except BaseException:
            # Once opened, the session and tracing directory belong to us even
            # when model IO inspection fails before construction completes.
            from stash_ai.runtime.cleanup import cleanup
            session, self.session = self.session, None
            directory, self.profile_directory = self.profile_directory, None
            callbacks = []
            if callable(getattr(session, 'close', None)):
                callbacks.append(session.close)
            if directory is not None:
                callbacks.append(directory.cleanup)
            cleanup(callbacks, report=self.report)
            raise
        self.node_profile_enabled = self.profile_directory is not None
        self._node_profile_batches = 0
        self._node_profile_limit = 64
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
        return self.predict_collated_batch(self.collate_batch(images), len(images))

    def collate_batch(self, images: list[Any]) -> dict[str, Any]:
        import numpy as np

        count = len(images)
        if not count or count > self.batch_size:
            raise ValueError("Invalid inference batch length.")

        def collate(rows):
            if self.fixed_batch and count < self.fixed_batch:
                # One final-size allocation rather than stack/repeat/concatenate.
                first = np.asarray(rows[0])
                values = np.empty((self.fixed_batch, *first.shape), dtype=first.dtype)
                for index, row in enumerate(rows):
                    values[index] = row
                values[count:] = first
                return values
            return np.stack(rows)

        feed = {self.input_name: collate(
            [pair[0] for pair in images] if self.mask_name else images)}
        if self.mask_name:
            feed[self.mask_name] = collate([pair[1] for pair in images])
        return feed

    def predict_collated_batch(self, feed: dict[str, Any], count: int) -> Any:
        from stash_ai.models import validate_scores
        return validate_scores(self._predict_collated_scores(feed, count), count,
                               self.labels, self.settings.source.title)

    def _predict_collated_scores(self, feed: dict[str, Any], count: int) -> Any:
        import numpy as np

        from stash_ai.runtime.timing import stage
        report = getattr(self, 'report', None)
        with stage(report, 'ort_session_run'):
            output = self.session.run([self.output_name], feed)[0]
        # Node tracing is opt-in and bounded to the first 64 backend batches.
        if getattr(self, 'node_profile_enabled', False):
            self._node_profile_batches += 1
            if self._node_profile_batches >= self._node_profile_limit:
                from stash_ai.backends.profiling import record_onnx
                directory, self.profile_directory = self.profile_directory, None
                self.node_profile_enabled = False
                def sample():
                    record_onnx(report, self.session, self.settings.source.key)
                    if report is not None and report.profile is not None:
                        row = report.profile.models.setdefault(self.settings.source.key, {})
                        row['node_profile_sample_batches'] = self._node_profile_batches
                try:
                    if report is not None:
                        report.optional('backend_node_sampling', sample)
                finally:
                    if directory is not None:
                        from stash_ai.runtime.cleanup import cleanup
                        cleanup([directory.cleanup], report=report)
        with stage(report, 'ort_output_copy'):
            raw = np.asarray(output, dtype=np.float32)
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
        return raw[:count]

    def close(self) -> None:
        from stash_ai.backends.profiling import record_onnx
        from stash_ai.runtime.cleanup import cleanup
        report, session = getattr(self, 'report', None), self.session
        directory = getattr(self, 'profile_directory', None)
        def release():
            self.session = None
            gc.collect()
        cleanup([lambda: record_onnx(report, session, self.settings.source.key) if session is not None and directory is not None else None, release,
                 *([directory.cleanup] if directory is not None else [])], report=report)
