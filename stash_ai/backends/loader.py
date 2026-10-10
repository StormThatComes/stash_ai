"""Local inference adapters. No Stash access and no tag/category allowlist."""
from __future__ import annotations

import gc
from typing import Any

from stash_ai import models
from stash_ai.backends import onnx as onnx_tagger
from stash_ai.backends import vocabulary as assets


class JoyTagger:
    def __init__(self, settings: models.ModelSettings, vocabulary: models.Vocabulary, prepare_image):
        import torch
        from stash_ai.backends.profiling import active
        self.report = active()
        self.settings, self.vocabulary = settings, vocabulary
        self.labels, self.batch_size = vocabulary.labels, settings.batch_size
        self.prepare_image = prepare_image
        assets.ensure_files(settings, assets.weight_files(settings))
        from stash_ai.backends.joytag import load_model
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
        return self.predict_collated_batch(self.collate_batch(images), len(images))

    def collate_batch(self, images: list[Any]) -> Any:
        import torch
        return torch.stack(images)

    def predict_collated_batch(self, prepared: Any, count: int) -> Any:
        return models.validate_scores(self._predict_collated_scores(prepared, count), count, self.labels, 'JoyTag')

    def _predict_collated_scores(self, prepared: Any, count: int) -> Any:
        import torch
        from contextlib import nullcontext
        from stash_ai.backends.profiling import gpu_events, record_gpu
        from stash_ai.runtime.timing import stage
        report = getattr(self, 'report', None)
        events = gpu_events(report, torch) if self.device.type == 'cuda' else None
        if events is not None:
            report.optional('backend_event_record', events[0].record)
        with stage(report, 'backend_input_transfer_enqueue'):
            batch = prepared.to(self.device, non_blocking=self.device.type == "cuda")
        if events is not None:
            report.optional('backend_event_record', events[1].record)
        amp = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), amp:
            with stage(report, 'backend_model_compute_enqueue'):
                device_scores = self.model({"image": batch})["tags"].sigmoid().float()
            if events is not None:
                report.optional('backend_event_record', events[2].record)
            with stage(report, 'backend_output_copy_and_wait'):
                scores = device_scores.cpu().numpy()
        if events is not None:
            record_gpu(report, events, self.settings.source.key, report.work(), count)
        return scores

    def close(self) -> None:
        import torch
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_tagger(settings: models.ModelSettings, vocabulary: models.Vocabulary, prepare_joytag):
    settings.validate()
    if settings.source.key == "joytag":
        return JoyTagger(settings, vocabulary, prepare_joytag)
    return onnx_tagger.OnnxTagger(settings, vocabulary)
