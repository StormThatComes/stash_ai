"""Offline SapiensID v1 source/checkpoint bootstrap tests with fake upstream.

No upstream Python import, real model checkpoint, Ultralytics install, or
network/expensive GPU inference is required for these tests.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from PIL import Image

from stash_ai.recognition import sapiensid_bootstrap as module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value


def nested(value):
    return Config({key: nested(item) for key, item in value.items()}) if isinstance(value, dict) else value


class Omega:
    @staticmethod
    def create(value):
        return nested(value)

    @staticmethod
    def load(stream):
        assert 'mrm' in stream.read()
        return nested(dict(name='mrm', input_size=[384, 384], output_dim=4096,
                           color_space='RGB', kp_preprocessor=dict(name='kp19')))


class FakeModel(torch.nn.Module):
    @classmethod
    def from_config(cls, config):
        assert config.name == 'mrm' and config.start_from == ''
        return cls()

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, images, *, foreground_masks, ldmks):
        assert ldmks.shape[1:] == (22, 2) and foreground_masks.shape[1:] == (384, 384)
        output = torch.zeros((len(images), 4096), device=images.device)
        output[:, 0] = self.scale
        return output


class FakeAligner(torch.nn.Module):
    @classmethod
    def from_config(cls, config):
        assert config.use_yolo is False and config.face_detector is None
        return cls(config)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch):
        assert self.config.use_yolo is True and self.config.face_detector == 'dfa_mobilenetv4'
        assert self.yolo_net.loaded and self.face_predictor is not None
        return (torch.full((len(batch), 22, 2), .5, device=batch.device),
                torch.ones((len(batch), 384, 384), dtype=torch.bool, device=batch.device))

    def eval(self):
        # Upstream intentionally violates nn.Module.eval's return contract.
        super().eval()


class DFA:
    @staticmethod
    def get_landmark_predictor(**kwargs):
        assert kwargs == dict(network='mobilenetv4', use_aggregator=True, input_size=160)
        return torch.nn.Linear(2, 2), object()

    @staticmethod
    def get_preprocessor(**kwargs):
        return object()

    @staticmethod
    def FaceKPPredictor(net, prior, preprocessing):
        return net


def bundle(tmp_path, monkeypatch, *, bad_yolo=False):
    checkout = tmp_path / 'trusted'
    checkout.mkdir()
    monkeypatch.setattr(module, '_trusted_checkout', lambda path: checkout)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    cfg = checkpoint / 'model.yaml'
    cfg.write_text('name: mrm\n', encoding='utf-8')
    model = checkpoint / 'model.pth'
    torch.save({'state_dict': FakeModel().state_dict()}, model)
    dfa = tmp_path / 'dfa.pth'
    torch.save(torch.nn.Linear(2, 2).state_dict(), dfa)
    pose = tmp_path / 'pose.pt'
    pose.write_bytes(b'verified-pose-fixture')
    observed = []

    class YOLO:
        def __init__(self, path):
            observed.append(Path(path))
            assert Path(path).name == 'yolov8n-pose.pt'
            assert Path(path).read_bytes() == pose.read_bytes()
            assert Path(path) != pose
            self.loaded = True

    def imports(_):
        observed.append('upstream import')
        return FakeModel, FakeAligner, DFA, YOLO, Omega

    monkeypatch.setattr(module, '_upstream_modules', imports)
    arguments = dict(
        checkout=checkout, checkpoint_dir=checkpoint,
        model_yaml_sha256=sha(cfg), yolo_pose_path=pose,
        yolo_pose_sha256=('0'*64 if bad_yolo else sha(pose)),
        dfa_path=dfa, dfa_sha256=sha(dfa),
        model_size=model.stat().st_size, model_sha256=sha(model), device='cpu',
    )
    return arguments, observed


def test_local_offline_bootstrap_builds_encoder_without_unsafe_dfa_loading(tmp_path, monkeypatch):
    kwargs, observed = bundle(tmp_path, monkeypatch)
    encoder = module.load_local_encoder(**kwargs)
    try:
        result = encoder.embed(Image.new('RGB', (120, 220), (30, 40, 50)))
        assert result.shape == (4096,)
        assert result[0] == pytest.approx(1.)
        assert observed[0] == 'upstream import'
        assert len(observed) == 2
        assert not observed[1].exists()  # YOLO got a verified temporary copy.
    finally:
        encoder.close()


def test_corrupt_pose_fails_before_any_upstream_import(tmp_path, monkeypatch):
    kwargs, observed = bundle(tmp_path, monkeypatch, bad_yolo=True)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module.load_local_encoder(**kwargs)
    assert not observed


def test_corrupt_config_fails_before_any_upstream_import(tmp_path, monkeypatch):
    kwargs, observed = bundle(tmp_path, monkeypatch)
    kwargs['model_yaml_sha256'] = '0'*64
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module.load_local_encoder(**kwargs)
    assert not observed


def test_source_provenance_must_be_pinned_and_clean(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    for name in (
        'tasks/sapiensID/src/models/sapiens_id/__init__.py',
        'tasks/sapiensID/src/aligners/keypoint_predictor/__init__.py',
        'tasks/sapiensID/src/aligners/keypoint_predictor/face_detector/dfa/__init__.py',
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# fixture\n', encoding='utf-8')

    def process(head, status):
        def run(args, **kwargs):
            if 'rev-parse' in args:
                return SimpleNamespace(stdout=head + '\n')
            return SimpleNamespace(stdout=status)
        return run

    monkeypatch.setattr(module.subprocess, 'run', process('a'*40, ''))
    with pytest.raises(ValueError, match='pinned'):
        module._trusted_checkout(root)
    monkeypatch.setattr(module.subprocess, 'run', process(module.UPSTREAM_COMMIT, '?? modified.py\n'))
    with pytest.raises(ValueError, match='clean'):
        module._trusted_checkout(root)
    monkeypatch.setattr(module.subprocess, 'run', process(module.UPSTREAM_COMMIT, ''))
    assert module._trusted_checkout(root) == root


def test_dfa_checkpoint_refuses_pickle_objects(tmp_path):
    class_holder = {'unsafe': 'not all tensor weights'}
    file = tmp_path / 'dfa.pth'
    torch.save(class_holder, file)
    with pytest.raises(ValueError, match='named tensors'):
        module._load_dfa_weights(DFA, file, sha256=sha(file))


def test_dfa_checkpoint_requires_correct_digest(tmp_path):
    file = tmp_path / 'dfa.pth'
    torch.save(torch.nn.Linear(2,2).state_dict(), file)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module._load_dfa_weights(DFA, file, sha256='0'*64)
    assert isinstance(module._load_dfa_weights(DFA, file, sha256=sha(file)), torch.nn.Module)


def test_asset_validation_rejects_symlink_and_oversize(tmp_path):
    original = tmp_path / 'weights.pt'
    original.write_bytes(b'abc')
    symlink = tmp_path / 'alias.pt'
    try:
        symlink.symlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip('Symlinks are not supported in this environment.')
    with pytest.raises(ValueError, match='symbolic links'):
        module._read_checked(symlink, sha(original), max_bytes=10, name='alias')
    with pytest.raises(ValueError, match='invalid asset type/size'):
        module._read_checked(original, sha(original), max_bytes=2, name='weights')
