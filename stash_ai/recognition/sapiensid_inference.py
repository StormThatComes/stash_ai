"""Synchronous PyTorch SapiensID v1 inference with validated person geometry."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from threading import get_ident

from .models import normalized_embedding
from .sapiensid_v1 import (
    EMBEDDING_DIMENSIONS, INPUT_SIZE, WB4M_BYTES, WB4M_SHA256, preprocess,
)


def load_verified_weights(model, path, *, expected_size=WB4M_BYTES,
                          expected_sha256=WB4M_SHA256):
    """Load a locally supplied SapiensID state dict into an existing module.

    Hash and deserialize the SAME open file descriptor, to avoid TOCTOU
    changes. `weights_only=True` is mandatory. Refuse incomplete or
    incompatible state dictionaries; no unsafe pickle fallback. Constructing
    the upstream model/configuration is the caller's responsibility.
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError('SapiensID model must be a torch.nn.Module.')
    if (type(expected_size) is not int or expected_size < 1
            or type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Invalid SapiensID checkpoint integrity metadata.')
    with Path(path).open('rb') as stream:
        initial = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial.st_mode) or initial.st_size != expected_size:
            raise ValueError('SapiensID checkpoint size mismatch.')
        digest = hashlib.sha256()
        consumed = 0
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            consumed += len(piece)
            if consumed > expected_size:
                raise ValueError('SapiensID checkpoint changed during verification.')
            digest.update(piece)
        final = os.fstat(stream.fileno())
        if (consumed != expected_size or final.st_size != expected_size
                or digest.hexdigest() != expected_sha256):
            raise ValueError('SapiensID checkpoint SHA256/size mismatch.')
        stream.seek(0)
        try:
            checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        except Exception as error:
            raise ValueError('SapiensID checkpoint cannot be loaded safely with weights_only=True.') from error

    if not isinstance(checkpoint, dict):
        raise ValueError('SapiensID checkpoint must contain a mapping of tensor weights.')
    for key in ('state_dict', 'model_state_dict', 'model'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(type(k) is str and isinstance(v, torch.Tensor)
                                 for k, v in checkpoint.items()):
        raise ValueError('SapiensID state_dict must contain only named tensors.')
    # Support an ordinary DDP checkpoint but never silently drop arbitrary
    # prefixes, tensor weights, or extra keys.
    if all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
    # "assign=False" by default; keep existing module tensor ownership.
    model.load_state_dict(checkpoint, strict=True)
    return model


def resolve_device(device):
    import torch
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('SapiensID device must be auto, cpu, or cuda.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('SapiensID CUDA was requested but is unavailable.')
    resolved = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    return torch.device('cuda:0' if resolved == 'cuda' else resolved)


class SapiensIDV1Encoder:
    """Synchronous, one-owner GPU/CPU bridge for fully constructed v1 modules.

    `aligner` must match upstream yolo_dfa contract:
        aligner(N,3,384,384) -> (N,22,2) keypoints, (N,384,384) bool mask.
    The internal upstream KP19Preprocessor selects 19 from those 22; callers
    must NOT pre-reduce to 19 or substitute SCRFD's five face landmarks.
    `model` must return (N,4096) numeric torch features, not class logits.
    """

    def __init__(self, model, aligner, *, device='auto', max_batch_size=4):
        import torch

        if not isinstance(model, torch.nn.Module) or (aligner is not None and not isinstance(aligner, torch.nn.Module)):
            raise TypeError('SapiensID model and aligner must be torch.nn.Module objects.')
        if type(max_batch_size) is not int or not 1 <= max_batch_size <= 16:
            raise ValueError('SapiensID max_batch_size must be in [1,16].')
        self.torch = torch
        self.device = resolve_device(device)
        self.max_batch_size = max_batch_size
        self.owner = get_ident()
        # Do not assign `aligner.eval()`: upstream KeypointPredictor.eval()
        # overrides nn.Module.eval() without returning self.
        self.model = model.to(self.device)
        self.aligner = aligner.to(self.device) if aligner is not None else None
        self.model.eval()
        if self.aligner is not None:
            self.aligner.eval()

    def embed(self, image):
        """Return one normalized 4096D NumPy vector for one person crop."""
        return self.embed_batch([image])[0]

    def embed_person(self, image, points):
        """Encode one associated person using 17 pose + 5 face pixel landmarks.

        Geometry follows the same square padding as the image. Missing body
        points stay (-1,-1); the model performs its own 22 -> 19 selection.
        """
        return self.embed_person_batch([(image,points)])[0]

    def embed_person_batch(self, persons):
        """Prepare bounded CPU inputs, then transfer one batch on the model owner."""
        from .sapiensid_v1 import prepare_person_input
        self._check_owner()
        if not isinstance(persons,(list,tuple)) or not 1<=len(persons)<=self.max_batch_size:
            raise ValueError('SapiensID expects 1..max_batch_size associated persons.')
        prepared=[prepare_person_input(image,points) for image,points in persons]
        tensors=[self.torch.cat(parts,dim=0).to(self.device) for parts in zip(*prepared)]
        return self._encode(*tensors)

    def _check_owner(self):
        if get_ident() != self.owner:
            raise RuntimeError('SapiensID GPU/model inference must remain on its owning thread.')
        if self.model is None:
            raise RuntimeError('SapiensID encoder is closed.')

    def embed_batch(self, images):
        """Infer bounded batches; invalid geometry or logits fail the whole call."""
        torch = self.torch
        if get_ident() != self.owner:
            raise RuntimeError('SapiensID GPU/model inference must remain on its owning thread.')
        if self.model is None:
            raise RuntimeError('SapiensID encoder is closed.')
        if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= self.max_batch_size:
            raise ValueError('SapiensID expects 1..max_batch_size person image crops.')
        batch = torch.cat([preprocess(image) for image in images], dim=0).to(self.device)
        if batch.shape != (len(images), 3, *INPUT_SIZE) or batch.dtype != torch.float32:
            raise ValueError('Invalid SapiensID input tensor.')
        if self.aligner is None:
            raise ValueError('SapiensID needs explicitly associated person keypoints.')
        with torch.inference_mode():
            keypoints, masks = self.aligner(batch)
        return self._encode(batch, keypoints, masks)

    def _encode(self, batch, keypoints, masks):
        torch = self.torch
        count = len(batch)
        with torch.inference_mode():
            if (not isinstance(keypoints, torch.Tensor)
                    or keypoints.shape != (count, 22, 2)
                    or keypoints.device != self.device
                    or keypoints.dtype not in (torch.float16, torch.float32, torch.float64)
                    or not torch.isfinite(keypoints).all().item()):
                raise ValueError('SapiensID aligner must return finite (N,22,2) keypoints.')
            missing = (keypoints == -1).all(dim=-1)
            valid = ((keypoints >= 0) & (keypoints <= 1)).all(dim=-1)
            if not torch.all(missing | valid).item():
                raise ValueError('SapiensID keypoints must be normalized or (-1,-1).')
            if (not isinstance(masks, torch.Tensor)
                    or masks.shape != (count, *INPUT_SIZE)
                    or masks.device != self.device or masks.dtype != torch.bool):
                raise ValueError('SapiensID aligner must return (N,384,384) boolean masks.')
            if not masks.flatten(1).any(dim=1).all().item():
                raise ValueError('SapiensID person mask must contain foreground pixels.')
            features = self.model(batch, foreground_masks=masks, ldmks=keypoints)
            if (not isinstance(features, torch.Tensor)
                    or features.shape != (count, EMBEDDING_DIMENSIONS)
                    or features.device != self.device or features.dtype not in
                    (torch.float16, torch.float32, torch.float64)
                    or not torch.isfinite(features).all().item()):
                raise ValueError('SapiensID model must return finite (N,4096) embeddings.')
            values = features.detach().float().cpu().numpy()
        # Double precision L2 normalization matches the existing embedding
        # namespace implementation; threshold calibration is still required.
        return [normalized_embedding(row, EMBEDDING_DIMENSIONS) for row in values]

    def close(self):
        """Release held modules; caller owns any shared CUDA cache."""
        if get_ident() != self.owner:
            raise RuntimeError('SapiensID must be closed by its inference owner.')
        self.model = None
        self.aligner = None
