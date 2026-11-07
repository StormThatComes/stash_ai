import numpy as np
import pytest


def test_tiled_dark_matches_reference_including_edges_and_empty_points():
    torch=pytest.importorskip('torch');pytest.importorskip('cv2')
    from stash_ai.recognition.pmpose_geometry import decode
    from stash_ai.recognition.pmpose_decode_batch import decode_batch
    maps=np.zeros((2,17,64,48),np.float32)
    maps[0,:,0,0]=1;maps[1,1:,22,11]=.9
    maps[1,1:,22,12]=.7
    c=np.array([[70,100],[130,200]],np.float32)
    s=np.array([[100,140],[128,170]],np.float32)
    expected=np.stack([decode(m,a,b) for m,a,b in zip(maps,c,s)])
    np.testing.assert_allclose(decode_batch(torch.from_numpy(maps),c,s),expected,atol=2e-3,rtol=1e-5)
