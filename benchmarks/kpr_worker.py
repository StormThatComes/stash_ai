"""Standalone feature extraction in the upstream KPR Python environment."""
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np


def extract(config,weights,directory,batches,device):
    import torch
    from torchreid.scripts.builder import build_config
    from torchreid.tools.feature_extractor import KPRFeatureExtractor
    from yacs.config import CfgNode

    options = CfgNode(dict(model=dict(load_weights=str(weights)),
                           data=dict(save_dir=str(directory))))
    cfg = build_config(config_path=str(config),config=options)
    if device=='cpu':
        cfg.use_gpu = False
    if device=='cuda' and not cfg.use_gpu:
        raise ValueError('KPR cannot use the requested CUDA device.')
    if not cfg.model.kpr.keypoints.use_negative_keypoints or cfg.model.promptable_trans.disable_inference_prompting:
        raise ValueError('KPR config must enable negative keypoints and inference prompting.')
    extractor = KPRFeatureExtractor(cfg,verbose=False)
    (directory/'device.txt').write_text('cuda' if cfg.use_gpu else 'cpu')
    with torch.inference_mode():
        for batch in range(batches):
            with np.load(directory/f'input-{batch}.npz',allow_pickle=False) as raw:
                samples = [{key:raw[f'{key}_{i}'] for key in ('image','keypoints_xyc','negative_kps')}
                           for i in range(len(raw.files)//3)]
            _,features,visible,_ = extractor(samples)
            np.savez(directory/f'output-{batch}.npz',features=features.detach().cpu().numpy(),
                     visible=visible.detach().cpu().numpy())


if __name__=='__main__':
    with redirect_stdout(sys.stderr):
        extract(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
