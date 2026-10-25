"""SapiensID v1 PyTorch inference bridge; no external downloads or GPUs required."""
import hashlib
from threading import Thread

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from PIL import Image

from stash_ai.recognition.sapiensid_inference import (
    SapiensIDV1Encoder, load_verified_weights,
)


class Aligner(torch.nn.Module):
    def __init__(self, *, malformed=False, missing=False, mask_invalid=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.malformed = malformed
        self.missing = missing
        self.mask_invalid = mask_invalid
        self.last_shape = None

    def eval(self):
        # Intentionally match upstream KeypointPredictor, which returns None.
        super().eval()

    def forward(self, images):
        self.last_shape = tuple(images.shape)
        n = images.shape[0]
        kps = torch.full((n, 22, 2), 0.5, device=images.device)
        if self.missing:
            kps[:, 1] = -1
        if self.malformed:
            kps[:, 1, 0] = -1
        mask = torch.ones((n, 384, 384), dtype=torch.bool, device=images.device)
        if self.mask_invalid:
            mask[:] = False
        return kps, mask


class PersonModel(torch.nn.Module):
    def __init__(self, *, wrong_shape=False, nonfinite=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.wrong_shape = wrong_shape
        self.nonfinite = nonfinite
        self.call_record = None

    def forward(self, images, *, foreground_masks, ldmks):
        self.call_record = (bool(torch.is_grad_enabled()), tuple(ldmks.shape),
                            tuple(foreground_masks.shape), images.device.type,
                            tuple(images.shape))
        dim = 512 if self.wrong_shape else 4096
        embeddings = torch.zeros((len(images), dim), device=images.device)
        embeddings[:, 0] = self.scale
        if self.nonfinite:
            embeddings[:, 1] = float('nan')
        return embeddings


def sample(color=(65, 85, 125), size=(80, 145)):
    return Image.new('RGB', size, color=color)


def test_cpu_inference_uses_existing_torch_with_official_22_keypoint_contract():
    model, aligner = PersonModel(), Aligner(missing=True)
    adapter = SapiensIDV1Encoder(model, aligner, device='cpu', max_batch_size=3)
    embeddings = adapter.embed_batch([sample(), sample((150, 80, 60), (151, 80))])
    assert len(embeddings) == 2
    for vector in embeddings:
        assert vector.shape == (4096,)
        assert vector.dtype == np.float32
        assert np.linalg.norm(vector) == pytest.approx(1.)
        assert vector[0] == pytest.approx(1.)
    assert aligner.last_shape == (2, 3, 384, 384)
    assert model.call_record == (False, (2, 22, 2), (2, 384, 384), 'cpu', (2, 3, 384, 384))
    assert adapter.embed(sample()).shape == (4096,)
    adapter.close()
    with pytest.raises(RuntimeError, match='closed'):
        adapter.embed(sample())


@pytest.mark.parametrize('kwargs,match', [
    ({'malformed': True}, 'keypoints must be normalized'),
    ({'mask_invalid': True}, 'must contain foreground'),
])
def test_aligner_contract_fails_closed(kwargs, match):
    adapter = SapiensIDV1Encoder(PersonModel(), Aligner(**kwargs), device='cpu')
    with pytest.raises(ValueError, match=match):
        adapter.embed(sample())


@pytest.mark.parametrize('kwargs', [{'wrong_shape': True}, {'nonfinite': True}])
def test_wrong_features_never_become_recognition_evidence(kwargs):
    adapter = SapiensIDV1Encoder(PersonModel(**kwargs), Aligner(), device='cpu')
    with pytest.raises(ValueError, match='must return finite'):
        adapter.embed(sample())


def test_batch_budget_and_thread_ownership():
    adapter = SapiensIDV1Encoder(PersonModel(), Aligner(), device='cpu', max_batch_size=1)
    with pytest.raises(ValueError, match='expects 1..'):
        adapter.embed_batch([])
    with pytest.raises(ValueError, match='expects 1..'):
        adapter.embed_batch([sample(), sample()])
    errors = []
    def other_thread():
        try:
            adapter.embed(sample())
        except Exception as error:
            errors.append(error)
    thread = Thread(target=other_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1 and 'owning thread' in str(errors[0])


def test_cuda_request_fails_if_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='unavailable'):
        SapiensIDV1Encoder(PersonModel(), Aligner(), device='cuda')


def test_safe_verified_checkpoint_loading(tmp_path):
    source = PersonModel()
    target = PersonModel()
    with torch.no_grad():
        source.scale.fill_(2.)
    weights = {'state_dict': {name: value.clone() for name, value in source.state_dict().items()}}
    path = tmp_path / 'model.pth'
    torch.save(weights, path)
    data = path.read_bytes()
    expected = dict(expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest())
    result = load_verified_weights(target, path, **expected)
    assert result is target and target.scale.item() == pytest.approx(2.)
    with pytest.raises(ValueError, match='SHA256'):
        load_verified_weights(PersonModel(), path, expected_size=len(data),
                              expected_sha256='0' * 64)
    with pytest.raises(ValueError, match='size mismatch'):
        load_verified_weights(PersonModel(), path, expected_size=len(data) + 1,
                              expected_sha256=expected['expected_sha256'])


def test_verified_checkpoint_rejects_wrong_state_dict(tmp_path):
    path = tmp_path / 'model.pth'
    torch.save({'state_dict': {'unexpected': torch.ones(1)}}, path)
    data = path.read_bytes()
    kwargs = dict(expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(RuntimeError):
        load_verified_weights(PersonModel(), path, **kwargs)


def test_verified_checkpoint_rejects_nonweights_without_unsafe_pickle(tmp_path):
    path = tmp_path / 'model.pth'
    torch.save({'other': 'not a state dict'}, path)
    data = path.read_bytes()
    kwargs = dict(expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match='only named tensors'):
        load_verified_weights(PersonModel(), path, **kwargs)
