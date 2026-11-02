"""Opt-in actual RF-DETR-XL checkpoint loading and person inference on CPU."""
import os

import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_RFDETR_SMOKE') != '1', reason='explicit 506 MB RF-DETR-XL pretrained smoke')
def test_pretrained_rfdetr_people_and_batched_coordinates(tmp_path, request, monkeypatch):
    import hashlib
    from io import BytesIO
    import numpy as np
    import requests
    import torch
    from PIL import Image
    from stash_ai.recognition.sapiensid_loading import load_pose
    from stash_ai.recognition import sapiensid_assets
    from tests.test_recognition_rfdetr import configuration

    previous = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous))
    torch.set_num_threads(2)
    url = ('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/'
           'lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg')
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        raw = response.content
    assert hashlib.sha256(raw).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    config = configuration(tmp_path, rfdetr_directory=os.getenv('STASH_AI_RFDETR_DIRECTORY', ''), batch_size=2)
    monkeypatch.setattr(sapiensid_assets, 'pose_asset', lambda _: pytest.fail('RF-DETR requested YOLO'))
    # Model construction must never download an independent backbone or use the
    # upstream checkpoint downloader (which can overwrite an invalid file).
    import rfdetr.assets.model_weights
    import huggingface_hub
    monkeypatch.setattr(rfdetr.assets.model_weights, 'download_pretrain_weights',
                        lambda *args, **kwargs: pytest.fail('upstream weight downloader called'))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda *args, **kwargs: pytest.fail('unexpected HF backbone download'))
    model = load_pose(config)
    assert model.task == 'detect' and not model.model.training
    with Image.open(BytesIO(raw)).convert('RGB') as image:
        options = dict(conf=.5, iou=.7, max_det=100)
        single = model.predict(image, **options)[0].boxes
        assert len(single.conf) == 1 and single.conf[0] > .8
        bounds = single.xyxy[0].numpy()
        assert bounds[0] < 180 and bounds[1] < 120 and bounds[2] > 350 and bounds[3] > 450
        with image.resize((image.width//2, image.height//2)) as half:
            batch = model.predict([image, half], **options)
            assert [len(item.boxes.conf) for item in batch] == [1, 1]
            np.testing.assert_allclose(batch[0].boxes.xyxy.numpy(), single.xyxy.numpy(), atol=.5)
            np.testing.assert_allclose(batch[1].boxes.xyxy.numpy()*2, single.xyxy.numpy(), atol=8)
            for result, frame in zip(batch, (image, half)):
                boxes = result.boxes.xyxy.numpy()
                assert np.isfinite(boxes).all() and (boxes >= 0).all()
                assert (boxes[:, 0::2] <= frame.width).all() and (boxes[:, 1::2] <= frame.height).all()
