"""Pinned RF-DETR assets, cold configuration, geometry and model admission."""
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import rfdetr_detector as rf, rfdetr_registry as registry
from stash_ai.recognition.config import defaults, validate
from tests.test_recognition_dfine import configuration as base_config, fingerprints


def configuration(tmp_path, **kwargs):
    return base_config(tmp_path, detector='rfdetr_xl', **kwargs)


@pytest.mark.parametrize('backend', ['sapiens', 'vitpose_plus_large', 'pmpose'])
def test_requires_separate_pose_backend_and_accepts_all_top_down_backends(backend):
    data = defaults()
    data['sapiensid']['pose']['detector'] = 'rfdetr_xl'
    with pytest.raises(ValueError, match='requires pose.backend'):
        validate(data)
    data['sapiensid']['pose']['backend'] = backend
    validate(data)


@pytest.mark.parametrize('value', [None, True, 2, [], ' ', '\0'])
def test_directory_validation(tmp_path, value):
    with pytest.raises(ValueError, match='rfdetr_directory'):
        validate(configuration(tmp_path, rfdetr_directory=value).data)


@pytest.mark.parametrize('detector', ['yolo', 'dfine_l', 'dfine_x'])
def test_existing_detector_contracts_are_unchanged(tmp_path, detector):
    config = base_config(tmp_path, detector=detector)
    expected = fingerprints(config)
    config.data['sapiensid']['pose']['rfdetr_directory'] = 'unused/local'
    assert fingerprints(config) == expected
    config.data['sapiensid']['pose'].pop('rfdetr_directory')
    assert fingerprints(config) == expected


def test_rfdetr_contract_versions_weights_and_runtime_but_not_paths_or_batches(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    expected = fingerprints(config)
    config.data['sapiensid']['pose'].update(rfdetr_directory='elsewhere', batch_size=1,
                                         path='unused.pt', sha256='1'*64, dfine_directory='unused')
    assert fingerprints(config) == expected
    for field, value in [('SHA256', '2'*64), ('VERSION', 'future'), ('PLUS_VERSION', 'future'),
                         ('INFERENCE', 'future'), ('PERSON_LABEL', 0)]:
        with monkeypatch.context() as patch:
            patch.setattr(registry, field, value)
            assert all(a != b for a, b in zip(expected, fingerprints(config)))
    for detector in ('yolo', 'dfine_l', 'dfine_x'):
        config.data['sapiensid']['pose']['detector'] = detector
        assert all(a != b for a, b in zip(expected, fingerprints(config)))


def fake_integrity(monkeypatch, raw):
    monkeypatch.setattr(registry, 'SIZE', len(raw))
    monkeypatch.setattr(registry, 'SHA256', hashlib.sha256(raw).hexdigest())


def test_lazy_verified_download_cache_and_corrupt_file_rejection(tmp_path, monkeypatch):
    from stash_ai.recognition import sapiensid_downloads
    raw = b'test checkpoint'
    fake_integrity(monkeypatch, raw)
    calls = []
    def install(url, path, size, checksum):
        calls.append(url)
        assert size == len(raw) and checksum == hashlib.sha256(raw).hexdigest()
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
    monkeypatch.setattr(sapiensid_downloads, 'install', install)
    config = configuration(tmp_path)
    path = rf.assets(config)
    assert path == config.model_dir / 'rfdetr_xl' / registry.SHA256 / registry.NAME
    assert rf.assets(config) == path and calls == [registry.URL]
    assert 'generation=1768355998186184' in calls[0]
    path.write_bytes(b'X'*len(raw))
    with pytest.raises(ValueError, match='SHA256'):
        rf.assets(config)
    assert len(calls) == 1  # Never overwrite a corrupt/custom checkpoint.


def test_offline_relative_directory_and_exported_configuration(tmp_path, monkeypatch):
    import yaml
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_report_policy import proposal
    config = configuration(tmp_path, rfdetr_directory='offline/rfdetr')
    config.data['models']['download_missing'] = False
    fake_integrity(monkeypatch, b'weights')
    with pytest.raises(FileNotFoundError, match='download_missing'):
        rf.assets(config)
    directory = tmp_path / 'offline/rfdetr'
    directory.mkdir(parents=True)
    (directory / registry.NAME).write_bytes(b'weights')
    assert rf.assets(config) == directory / registry.NAME
    exported = yaml.safe_load(configuration_document(config, proposal(config, {}), {}))
    assert exported['sapiensid']['pose']['rfdetr_directory'] == str(directory.resolve())
    assert exported['sapiensid']['pose']['detector'] == 'rfdetr_xl'


@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_unvalidated_runtime_fails_before_download(tmp_path, monkeypatch, missing):
    def version(name):
        if missing: raise rf.PackageNotFoundError(name)
        return 'unvalidated-version'
    monkeypatch.setattr(rf, 'version', version)
    monkeypatch.setattr(rf, 'assets', lambda _: pytest.fail('download before runtime validation'))
    with pytest.raises(RuntimeError, match='RF-DETR-XL requires.*requirements-cpu'):
        rf.load(configuration(tmp_path))


@pytest.mark.parametrize('incomplete', [False, True])
def test_loader_constructs_locally_and_strictly_loads_verified_weights(tmp_path, monkeypatch, incomplete):
    import sys
    torch = pytest.importorskip('torch')
    config = configuration(tmp_path, rfdetr_directory='weights')
    model = torch.nn.Linear(2, 1)
    model.register_buffer('_kp_active_mask', torch.empty(0, 0, dtype=torch.bool))
    checkpoint = copy.deepcopy(model.state_dict())
    del checkpoint['_kp_active_mask']
    if incomplete: del checkpoint['bias']
    path = tmp_path / 'weights' / registry.NAME
    path.parent.mkdir()
    torch.save({'model': checkpoint, 'license': 'PML-1.0'}, path)
    fake_integrity(monkeypatch, path.read_bytes())
    monkeypatch.setattr(rf, 'version', lambda name: registry.VERSION if name == 'rfdetr' else registry.PLUS_VERSION)
    calls = []
    def build(architecture):
        assert architecture.pretrain_weights == str(path.resolve())
        assert architecture.num_classes == 90 and architecture.device == 'cpu'
        calls.append(architecture)
        return model
    monkeypatch.setitem(sys.modules, 'rfdetr_plus.models.detection', SimpleNamespace(RFDETRXLargeConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'rfdetr.models', SimpleNamespace(build_model_from_config=build))
    if incomplete:
        with pytest.raises(RuntimeError, match='Missing key'):
            rf.load(config)
    else:
        detector = rf.load(config)
        assert detector.model is model and not model.training and model.weight.dtype == torch.float32
        assert '_kp_active_mask' not in model.state_dict()
    assert len(calls) == 1


def detections():
    torch = pytest.importorskip('torch')
    logits = torch.full((1, 8, 91), -20.)
    logits[0, :, 1] = torch.logit(torch.tensor([.95, .90, .85, .2, .8, .5, .99, .99]))
    logits[0, 3, 0] = logits[0, 3, 2] = 20.  # Neither slot zero nor other objects are people.
    boxes = torch.tensor([[[.5, .5, .8, .8], [.5, .5, .79, .79], [.1, .1, .4, .4],
                          [.7, .7, .1, .1], [1.2, .5, .1, .2], [.8, .8, .2, .2],
                          [.5, .5, -.2, .2], [.5, .5, .2, 0.]]])
    return dict(pred_logits=logits, pred_boxes=boxes)


def test_person_category_original_coordinates_clipping_nms_limit_and_invalid_geometry():
    result = rf.RFDETRDetector._decode(detections(), [(100, 200)], .5, .7, 2, True)[0].boxes
    np.testing.assert_allclose(result.xyxy.numpy(), [[10, 20, 90, 180], [0, 0, 30, 60]], atol=1e-5)
    np.testing.assert_allclose(result.conf.numpy(), [.95, .85])
    assert len(rf.RFDETRDetector._decode(detections(), [(100, 200)], .5, .7, 1, True)[0].boxes.conf) == 1
    for suppress, iou in ((False, .7), (True, 1.)):
        result = rf.RFDETRDetector._decode(detections(), [(100, 200)], .5, iou, 2, suppress)[0].boxes
        np.testing.assert_allclose(result.conf.numpy(), [.95, .90])
    empty = rf.RFDETRDetector._decode(detections(), [(100, 200)], 1., .7, 100, True)[0].boxes
    assert empty.xyxy.shape == (0, 4) and empty.conf.shape == (0,)


@pytest.mark.parametrize('field', ['pred_logits', 'pred_boxes'])
def test_nonfinite_filtered_proposal_is_an_error_with_frame_location(field):
    output = detections()
    output[field][0, 3, 0] = float('nan')
    with pytest.raises(ValueError, match='non-finite.*input_frame=4, query=3, component=0'):
        rf.RFDETRDetector._decode(output, [(100, 200)], .5, .7, 100, True, frame_offset=4)


@pytest.mark.parametrize('field', ['pred_logits', 'pred_boxes'])
def test_malformed_shapes_fail(field):
    output = detections()
    output[field] = output[field][..., :-1]
    with pytest.raises(ValueError, match='malformed COCO91'):
        rf.RFDETRDetector._decode(output, [(100, 200)], .5, .7, 100, True)


def test_rgb_normalization_batching_and_autocast_isolation():
    torch = pytest.importorskip('torch')
    calls = []
    def model(pixels):
        assert pixels.shape[1:] == (3, 700, 700) and pixels.dtype == torch.float32
        assert not torch.is_grad_enabled() and not torch.is_autocast_enabled('cpu')
        for channel, (mean, std) in enumerate(zip(registry.MEAN, registry.STD)):
            np.testing.assert_allclose(pixels[:, channel].numpy(), ((1. if channel == 0 else 0.)-mean)/std, atol=1e-6)
        calls.append(len(pixels))
        logits = torch.full((len(pixels), 1, 91), -20.); logits[..., 1] = 10.
        return dict(pred_logits=logits, pred_boxes=torch.tensor([[[.5, .5, 1., 1.]]]).repeat(len(pixels), 1, 1))
    detector = rf.RFDETRDetector(model, device='cpu', batch_size=2)
    with Image.new('RGBA', (40, 80), 'red') as a, Image.new('RGB', (90, 50), 'red') as b:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            results = detector.predict([a, b, a], conf=.5, iou=.7, max_det=100)
            assert torch.is_autocast_enabled('cpu')
        assert a.getpixel((0, 0)) == (255, 0, 0, 255)
    assert calls == [2, 1]
    for result, bounds in zip(results, ([0, 0, 40, 80], [0, 0, 90, 50], [0, 0, 40, 80])):
        np.testing.assert_allclose(result.boxes.xyxy.numpy(), [bounds])
    assert detector.predict([], conf=.5, iou=.7, max_det=100) == []
