"""Lazy mask backend selection. All backends share pose ownership validation."""
from contextlib import contextmanager

BACKENDS = ('sam_pose2seg', 'sam3', 'hq_sam2_large', 'rfdetr_seg_large')


def model_contract(backend):
    if backend == 'hq_sam2_large':
        from . import hq_sam2_registry as r
        return dict(backend=backend, revision=r.SOURCE_REVISION, weights=r.SHA256, adapter=1)
    if backend == 'sam3':
        from .sam3_masks import REVISION, FILES
        return dict(backend=backend, revision=REVISION, files=FILES, adapter=1)
    if backend == 'rfdetr_seg_large':
        from .rfdetr_masks import SHA256, VERSION
        return dict(backend=backend, weights=SHA256, version=VERSION, adapter=1)
    raise ValueError('Unknown mask backend: ' + str(backend))


def load(config, device, backend):
    from .inference_precision import validate
    validate(device, config.data['execution']['sam_precision'])
    if backend == 'hq_sam2_large':
        from .hq_sam2_masks import load
    elif backend == 'sam3':
        from .sam3_masks import load
    elif backend == 'rfdetr_seg_large':
        from .rfdetr_masks import load
    else:
        raise ValueError('Unknown mask backend: ' + str(backend))
    return load(config, device)


def asset_directory(config, backend, revision):
    from .sapiensid_loading import local_path
    selected = config.data['sapiensid']['pose']['refinement']['directory']
    return local_path(config, selected) if selected else config.model_dir / backend / revision


def checkpoint(config, backend, revision, name, url, size, checksum):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_v1 import verify_checkpoint
    path = asset_directory(config, backend, revision) / name
    if not path.exists():
        _require_download(config, path)
        install(url, path, size, checksum)
    verify_checkpoint(path, expected_size=size, expected_sha256=checksum)
    return path


class FrameSegmenter:
    """Bounded single-image features, reused for all people and decoder retries.

    These adapters deliberately keep only one encoded frame resident. The
    shared refinement queue still batches masked pose passes across frames.
    """
    def __init__(self, model, device, precision):
        self.model, self.device, self.precision = model, device, precision
        self.size = None

    def resident_modules(self):
        return [self.model]

    def prepare_residency(self):
        self.clear()

    def clear(self):
        self.size = None

    @contextmanager
    def frame(self, image):
        import torch
        from .inference_precision import autocast
        from .profiling import pipeline_timing
        try:
            with torch.inference_mode(), autocast(self.device, self.precision):
                self.size = image.size
                with pipeline_timing('sam_image_embedding'):
                    self.encode(image)
                yield self
        finally:
            self.clear()

    def predict(self, positive, negative=()):
        return self.options(positive, negative, False)[0]

    def predict_options(self, positive, negative):
        return self.options(positive, negative, True)

    def predict_frames(self, frames, *, process=None):
        result = []
        for image, prompts in frames:
            with self.frame(image):
                initial = [self.predict(p, n) for p, n in prompts]
                result.append(process(prompts, initial, self.options) if process else initial)
        return result

    def refine_frames(self, frames, process):
        return self.predict_frames(frames, process=process)

    def checked(self, masks, scores):
        import numpy as np
        from .sam_pose2seg import Segmenter
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (self.size is None or masks.ndim != 3 or masks.shape[1:] != self.size[::-1]
                or not 1 <= len(masks) <= 3 or scores.shape != (len(masks),)):
            raise ValueError('Mask backend returned invalid full-frame candidate shapes.')
        return [Segmenter._checked(m[None], np.asarray([s])) for m, s in zip(masks, scores)]
