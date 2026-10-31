"""Person-only D-FINE geometry, integrity, evidence contracts and pose integration."""
import copy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, load, plain, validate
from stash_ai.recognition import dfine_detector as dfine, dfine_registry as registry


def configuration(tmp_path, **options):
    path = tmp_path / 'recognition.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nsapiensid:\n'
                    '  pose: {backend: sapiens, detector: dfine_l}\n')
    config = load(path)
    data = plain(config.data)
    data['sapiensid']['pose'].update(options)
    return replace(config, data=data)


@pytest.mark.parametrize('key,value', [
    ('detector', 'dfine'), ('detector', None), ('detector', []), ('detector', True),
    ('dfine_directory', None), ('dfine_directory', ' '), ('dfine_directory', '\0'),
])
def test_invalid_options(tmp_path, key, value):
    config = configuration(tmp_path, **{key: value})
    with pytest.raises(ValueError, match='sapiensid.pose'):
        validate(config.data)


@pytest.mark.parametrize('detector', ['dfine_l', 'dfine_x'])
def test_box_only_detector_requires_top_down_pose(detector):
    data = defaults()
    data['sapiensid']['pose']['detector'] = detector
    with pytest.raises(ValueError, match='requires pose.backend'):
        validate(data)
    for backend in ('sapiens', 'vitpose_plus_large'):
        data['sapiensid']['pose']['backend'] = backend
        validate(data)


def fingerprints(config):
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract
    return contract(config), body_contract(config), identity_contract(config)


@pytest.mark.parametrize('backend', ['yolo', 'sapiens', 'vitpose_plus_large'])
def test_old_and_current_yolo_contracts_are_unchanged(tmp_path, backend):
    config = configuration(tmp_path, detector='yolo', backend=backend)
    expected = fingerprints(config)
    config.data['sapiensid']['pose']['dfine_directory'] = 'unused/weights'
    assert fingerprints(config) == expected
    for field in ('detector', 'dfine_directory'):
        config.data['sapiensid']['pose'].pop(field)
    assert fingerprints(config) == expected


def test_detector_weights_and_processing_separate_evidence(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    expected = fingerprints(config)
    config.data['sapiensid']['pose'].update(dfine_directory='elsewhere', batch_size=1,
                                          path='unused.pt', sha256='1' * 64)
    assert fingerprints(config) == expected
    for detector in ('yolo', 'dfine_x'):
        config.data['sapiensid']['pose']['detector'] = detector
        assert all(a != b for a, b in zip(fingerprints(config), expected))
    config.data['sapiensid']['pose']['detector'] = 'dfine_l'
    monkeypatch.setattr(registry, 'INFERENCE', 'future-preprocessing')
    assert all(a != b for a, b in zip(fingerprints(config), expected))


@pytest.mark.parametrize('field,value', [('score_threshold', .3), ('nms_threshold', .5),
                                        ('max_people_per_frame', 1)])
def test_person_selection_settings_change_evidence(tmp_path, field, value):
    config = configuration(tmp_path)
    expected = fingerprints(config)
    config.data['sapiensid']['pose'][field] = value
    assert all(a != b for a, b in zip(fingerprints(config), expected))


def fake_assets(monkeypatch):
    models = copy.deepcopy(registry.MODELS)
    payloads = {name: name.encode() for name in models['dfine_l']['assets']}
    for model in models.values():
        model['assets'] = {name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in payloads.items()}
    monkeypatch.setattr(registry, 'MODELS', models)
    return payloads


@pytest.mark.parametrize('detector', ['dfine_l', 'dfine_x'])
def test_only_selected_pinned_assets_download_and_corrupt_cache_is_rejected(tmp_path, monkeypatch, detector):
    from stash_ai.recognition import sapiensid_downloads
    config = configuration(tmp_path, detector=detector)
    payloads = fake_assets(monkeypatch)
    calls = []
    def install(url, path, size, checksum):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[path.name])
    monkeypatch.setattr(sapiensid_downloads, 'install', install)
    model = registry.MODELS[detector]
    directory = dfine.assets(config)
    assert directory == config.model_dir / detector / model['revision']
    assert calls == [f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}"
                     for name in payloads]
    assert dfine.assets(config) == directory and len(calls) == 3
    path = directory / 'model.safetensors'
    path.write_bytes(b'x' * len(payloads[path.name]))
    with pytest.raises(ValueError, match='SHA256'):
        dfine.assets(config)
    assert len(calls) == 3


def test_offline_relative_directory(tmp_path, monkeypatch):
    config = configuration(tmp_path, dfine_directory='offline/dfine')
    config.data['models']['download_missing'] = False
    payloads = fake_assets(monkeypatch)
    with pytest.raises(FileNotFoundError, match='download_missing'):
        dfine.assets(config)
    directory = tmp_path / 'offline/dfine'
    directory.mkdir(parents=True)
    for name, raw in payloads.items():
        (directory / name).write_bytes(raw)
    assert dfine.assets(config) == directory


def test_exported_yaml_preserves_detector_and_relative_directory(tmp_path):
    import yaml
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_report_policy import proposal
    config = configuration(tmp_path, detector='dfine_x', dfine_directory='custom/dfine')
    exported = yaml.safe_load(configuration_document(config, proposal(config, {}), {}))
    assert exported['sapiensid']['pose']['detector'] == 'dfine_x'
    assert exported['sapiensid']['pose']['dfine_directory'] == str((tmp_path / 'custom/dfine').resolve())


def test_missing_runtime_fails_before_download(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'transformers':raise ImportError('missing DFineForObjectDetection')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    monkeypatch.setattr(dfine, 'assets', lambda _: pytest.fail('download before runtime check'))
    with pytest.raises(RuntimeError, match='D-FINE requires.*requirements-sapiensid'):
        dfine.load(configuration(tmp_path))


@pytest.mark.parametrize('mismatch', [False, True])
def test_loader_uses_verified_local_config_and_rejects_partial_weights(tmp_path, monkeypatch, mismatch):
    import json
    torch = pytest.importorskip('torch')
    from transformers import DFineForObjectDetection, DFineConfig
    config = configuration(tmp_path, dfine_directory='local')
    directory = tmp_path / 'local'
    directory.mkdir()
    documents = {'config.json': dict(model_type='d_fine', use_pretrained_backbone=False,
        use_focal_loss=True, id2label={str(i): 'person' if i == 0 else str(i) for i in range(80)}),
        'preprocessor_config.json': dict(size={'height': 640, 'width': 640}, do_normalize=False)}
    model_assets = {}
    for name, document in documents.items():
        raw = json.dumps(document).encode()
        (directory / name).write_bytes(raw)
        model_assets[name] = len(raw), hashlib.sha256(raw).hexdigest()
    models = copy.deepcopy(registry.MODELS)
    models['dfine_l']['assets'] = model_assets
    monkeypatch.setattr(registry, 'MODELS', models)
    calls = []
    def pretrained(path, **kwargs):
        calls.append(path)
        assert path == str(directory) and isinstance(kwargs['config'], DFineConfig)
        assert kwargs['local_files_only'] and kwargs['use_safetensors']
        assert kwargs['trust_remote_code'] is False and kwargs['output_loading_info']
        return torch.nn.Linear(1, 1), dict(missing_keys=['backbone.weight'] if mismatch else [])
    monkeypatch.setattr(DFineForObjectDetection, 'from_pretrained', pretrained)
    if mismatch:
        with pytest.raises(ValueError, match='checkpoint does not match'):
            dfine.load(config)
    else:
        detector = dfine.load(config)
        assert not detector.model.training and detector.model.weight.dtype == torch.float32
    assert len(calls) == 1


def detections():
    torch = pytest.importorskip('torch')
    logits = torch.full((1, 6, 80), -20.)
    logits[0, :, 0] = torch.logit(torch.tensor([.95, .90, .85, .2, .8, .5]))
    logits[0, 3, 1:] = 20.  # Other classes must not fill the person budget.
    boxes = torch.tensor([[[.5, .5, .8, .8], [.5, .5, .79, .79], [.1, .1, .4, .4],
                           [.7, .7, .1, .1], [1.2, .5, .1, .2], [.8, .8, .2, .2]]])
    return SimpleNamespace(logits=logits, pred_boxes=boxes)


def test_person_filter_clipping_nms_limit_and_original_coordinates():
    result = dfine.DFineDetector._decode(detections(), [(100, 200)], .5, .7, 2, True)[0].boxes
    np.testing.assert_allclose(result.xyxy.numpy(), [[10, 20, 90, 180], [0, 0, 30, 60]], atol=1e-5)
    np.testing.assert_allclose(result.conf.numpy(), [.95, .85])
    assert len(dfine.DFineDetector._decode(detections(), [(100, 200)], .5, .7, 1, True)[0].boxes.conf) == 1
    # At IoU=1 the second overlapping candidate survives.
    result = dfine.DFineDetector._decode(detections(), [(100, 200)], .5, 1., 2, True)[0].boxes
    np.testing.assert_allclose(result.conf.numpy(), [.95, .90])
    empty = dfine.DFineDetector._decode(detections(), [(100, 200)], 1., .7, 100, True)[0].boxes
    assert empty.xyxy.shape == (0, 4) and empty.conf.shape == (0,)


@pytest.mark.parametrize('failure', ['nan', 'infinite_score', 'negative_width', 'classes', 'frames', 'boxes'])
def test_malformed_outputs_fail_before_pose(failure):
    output = detections()
    if failure == 'nan':output.pred_boxes[0, 0, 0] = float('nan')
    elif failure == 'infinite_score':output.logits[0, 0, 0] = float('inf')
    elif failure == 'negative_width':output.pred_boxes[0, 0, 2] = -.2
    elif failure == 'classes':output.logits = output.logits[:, :, :79]
    elif failure == 'frames':output.logits = output.logits[:0]
    else:output.pred_boxes = output.pred_boxes[:, :, :3]
    with pytest.raises(ValueError, match='D-FINE returned malformed'):
        dfine.DFineDetector._decode(output, [(100, 200)], .5, .7, 100, True)


def test_real_processor_rgb_and_bounded_mixed_size_batches():
    torch = pytest.importorskip('torch')
    from transformers import RTDetrImageProcessor
    processor = RTDetrImageProcessor(size={'height': 64, 'width': 64}, do_normalize=False, do_pad=False)
    calls = []
    def model(**inputs):
        pixels = inputs['pixel_values']
        assert not torch.is_grad_enabled() and pixels.shape[1:] == (3, 64, 64)
        np.testing.assert_allclose(pixels[:, 0].numpy(), 1., atol=1e-6)
        np.testing.assert_allclose(pixels[:, 1:].numpy(), 0., atol=1e-6)
        calls.append(len(pixels))
        logits = torch.full((len(pixels), 1, 80), -20.)
        logits[:, :, 0] = 10.
        return SimpleNamespace(logits=logits, pred_boxes=torch.tensor([[[.5, .5, 1., 1.]]]).repeat(len(pixels), 1, 1))
    detector = dfine.DFineDetector(model, processor, device='cpu', batch_size=2)
    with Image.new('RGBA', (40, 80), 'red') as a, Image.new('RGB', (90, 50), 'red') as b:
        results = detector.predict([a, b, a], conf=.5, iou=.7, max_det=100)
        assert a.getpixel((0, 0)) == (255, 0, 0, 255)
    assert calls == [2, 1]
    for result, bounds in zip(results, ([0, 0, 40, 80], [0, 0, 90, 50], [0, 0, 40, 80])):
        np.testing.assert_allclose(result.boxes.xyxy.numpy(), [bounds])
    assert detector.predict([], conf=.5, iou=.7, max_det=100) == []


@pytest.mark.parametrize('selected', ['sapiens', 'vitpose_plus_large'])
def test_dfine_replaces_yolo_in_single_and_batched_pose_pipeline(tmp_path, monkeypatch, selected):
    torch = pytest.importorskip('torch')
    from stash_ai.recognition import sapiensid_backend, sapiensid_assets, sapiens_pose, vitpose_pose
    config = configuration(tmp_path, backend=selected)
    config.data['sapiensid']['collection']['enabled'] = True
    monkeypatch.setattr(sapiensid_assets, 'pose_asset', lambda _: pytest.fail('YOLO asset was requested'))
    class Processor:
        def __call__(self, *, images, return_tensors):
            from transformers import BatchFeature
            return BatchFeature({'pixel_values': torch.zeros(len(images), 3, 8, 8)})
    def model(**inputs):
        count = len(inputs['pixel_values'])
        logits = torch.full((count, 1, 80), -20.);logits[:, :, 0] = 10.
        return SimpleNamespace(logits=logits, pred_boxes=torch.tensor([[[.5, .5, .8, .8]]]).repeat(count, 1, 1))
    monkeypatch.setattr(dfine, 'load', lambda _: dfine.DFineDetector(model, Processor(), device='cpu', batch_size=2))
    seen = []
    class Refiner:
        def predict(self, image, bounds):
            seen.append(bounds)
            return np.full((len(bounds), 17, 2), 30., dtype=np.float32), np.ones((len(bounds), 17), dtype=np.float32)
        def predict_many(self, frames):
            return [self.predict(image, bounds) for image, bounds in frames]
    chosen = sapiens_pose if selected == 'sapiens' else vitpose_pose
    other = vitpose_pose if selected == 'sapiens' else sapiens_pose
    monkeypatch.setattr(chosen, 'load', lambda *args: Refiner())
    monkeypatch.setattr(other, 'load', lambda *args: pytest.fail('unselected pose loaded'))
    detector = sapiensid_backend.SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    try:
        with Image.new('RGB', (100, 200)) as image:
            detector.detect(image, faces=[], force_pose=True)
            before = copy.deepcopy(detector.frame_people)
            detector.detect_many([(image, [])])
            detector.detect(image, faces=[])
            assert detector.frame_people == before and len(before) == 1
            assert len(detector.person_geometry(image, before[0])[1]) == 22
        assert len(seen) == 2
        assert detector.provenance['recognizer']['person_detector']['detector'] == 'dfine_l'
        assert 'pose_sha256' not in detector.provenance['recognizer']
    finally:
        detector.close()
    assert detector.pose is None and detector.pose_refiner is None
