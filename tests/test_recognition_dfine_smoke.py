"""Opt-in download, strict loading and original-coordinate detection with both models."""
import os

import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_DFINE_SMOKE') != '1', reason='explicit 378 MB D-FINE pretrained smoke')
@pytest.mark.parametrize('detector', ['dfine_l', 'dfine_x'])
def test_pretrained_dfine_people_and_batched_coordinates(tmp_path, request, detector, monkeypatch):
    import hashlib
    from io import BytesIO
    import numpy as np
    import requests
    import torch
    import yaml
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.sapiensid_loading import load_pose
    from stash_ai.recognition import sapiensid_assets

    previous = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous))
    torch.set_num_threads(2)
    url = ('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/'
           'lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg')
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        raw = response.content
    assert hashlib.sha256(raw).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    models = dict(device='cpu', directory=os.getenv('STASH_AI_DFINE_MODEL_DIRECTORY', str(tmp_path/'models')))
    path = tmp_path / 'recognition.yml'
    path.write_text(yaml.safe_dump(dict(version=1, models=models,
        sapiensid=dict(pose=dict(backend='sapiens', detector=detector, batch_size=2)))))
    config = load(path)
    monkeypatch.setattr(sapiensid_assets, 'pose_asset', lambda _: pytest.fail('D-FINE requested YOLO'))
    model = load_pose(config)
    assert model.task == 'detect'
    with Image.open(BytesIO(raw)).convert('RGB') as image:
        options = dict(conf=.5, iou=.7, max_det=100)
        single = model.predict(image, **options)[0].boxes
        assert len(single.conf) == 1 and single.conf[0] > .8
        bounds = single.xyxy[0].numpy()
        # The known portrait's face and shoulders must lie inside the person box.
        assert bounds[0] < 180 and bounds[1] < 120 and bounds[2] > 350 and bounds[3] > 450
        with image.resize((image.width//2, image.height//2)) as half:
            batch = model.predict([image, half], **options)
            assert [len(item.boxes.conf) for item in batch] == [1, 1]
            np.testing.assert_allclose(batch[0].boxes.xyxy.numpy(), single.xyxy.numpy(), atol=.5)
            np.testing.assert_allclose(batch[1].boxes.xyxy.numpy() * 2, single.xyxy.numpy(), atol=8)
            for result, frame in zip(batch, (image, half)):
                boxes = result.boxes.xyxy.numpy()
                assert np.isfinite(boxes).all() and (boxes >= 0).all()
                assert (boxes[:, 0::2] <= frame.width).all() and (boxes[:, 1::2] <= frame.height).all()
