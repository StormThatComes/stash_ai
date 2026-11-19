"""Opt-in real CPU model loading/inference; not a segmentation quality benchmark."""
import os
from dataclasses import replace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import load, plain


@pytest.mark.parametrize('backend', ['hq_sam2_large','rfdetr_seg_large','sam3'])
def test_pretrained_mask_backend(tmp_path, backend):
    if os.getenv('STASH_AI_MASK_BACKEND_SMOKE') != backend:
        pytest.skip('Set STASH_AI_MASK_BACKEND_SMOKE to the backend to download/run real weights')
    torch = pytest.importorskip('torch')
    from stash_ai.recognition.sam_pose2seg import load as load_segmenter
    path = tmp_path/'config.yml'
    path.write_text('version: 2\nexecution:\n  gpu: {device: cpu}\npose:\n'
                    '  estimator: {backend: pmpose}\n  refinement: {enabled: true, backend: '+backend+'}\n')
    config = load(path); config = replace(config, data=plain(config.data))
    directory = os.getenv('STASH_AI_MASK_DIRECTORY')
    if directory:
        config.data['sapiensid']['pose']['refinement']['directory'] = directory
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        model = load_segmenter(config, 'cpu')
        image = Image.new('RGB', (96,64), (90,100,110))
        positive, negative = np.array([[20.,20.],[30.,30.]]), np.array([[70.,30.]])
        with model.frame(image):
            mask,score = model.predict(positive,negative)
            assert mask.shape == (64,96) and mask.dtype == np.bool_ and 0 <= score <= 1
            options = model.predict_options(positive,negative)
            assert 1 <= len(options) <= 3
        batch = model.predict_frames([(image,[(positive,negative),(positive,negative)])])
        assert len(batch)==1 and len(batch[0])==2
        assert all(m.shape == (64,96) and m.dtype == np.bool_ for m,_ in batch[0])
    finally:
        torch.set_num_threads(previous)
