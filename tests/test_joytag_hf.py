"""Opt-in CPU integration check against the pinned upstream HF architecture."""
import importlib.util
import json
import os

import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_HF_SMOKE') != '1', reason='HF integration runs in CI')
def test_pinned_loader_matches_upstream_safetensors_inference(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from stash_ai.joytag import loader_path, load_model

    torch.set_num_threads(1)
    path = loader_path()
    spec = importlib.util.spec_from_file_location('_upstream_joytag_reference', path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    # Exercise the real ViT/CNN-stem code with tiny dimensions and generated
    # safetensors. No pretrained weights, GPU, Stash or user media are needed.
    config = dict(image_size=32, n_tags=2, num_blocks=1, patch_size=16,
                  d_model=32, mlp_dim=64, num_heads=4, stochdepth_rate=0.0,
                  use_sine=True, loss_type='focal2',
                  cnn_stem='conv:c=8;ln;relu;conv:c=16;ln;relu;conv:c=16;ln;relu;conv:c=32;ln;relu;conv:c=32,s=1,k=1,p=0')
    model = upstream.ViT(**config).eval()
    (tmp_path/'config.json').write_text(json.dumps({'class': 'ViT', **config}))
    save_file(model.state_dict(), str(tmp_path/'model.safetensors'))
    (tmp_path/'model.pt').write_bytes(b'must never load pickle weights')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('model.pt loader was called'))
    reference = upstream.VisionModel.load_model(tmp_path).eval()
    actual = load_model(tmp_path, 2, download_missing=False).eval()
    batch = {'image': torch.zeros(2, 3, 32, 32)}
    with torch.inference_mode():
        expected = reference(batch)['tags'].sigmoid()
        result = actual(batch)['tags'].sigmoid()
    assert result.shape == (2, 2) and torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
