from types import SimpleNamespace
import numpy as np
from PIL import Image
from stash_ai.recognition.pose_masks import PoseMasks
from stash_ai.recognition.pose_refinement_config import defaults


def test_generic_mask_preserves_geometry_and_marks_unknown_visibility():
    xy = np.zeros((1,17,2)); xy[0] = [20,20]
    xy[0,5] = [15,30]; xy[0,6] = [25,30]
    scores = np.full((1,17),.9)
    pose = SimpleNamespace(batch_size=1,predict_many=lambda frames:[(xy,scores)])
    mask = np.ones((50,50),bool)
    segmenter = SimpleNamespace(frame=lambda image: __import__('contextlib').nullcontext(),
                                predict=lambda *args:(mask,.99))
    model = PoseMasks(pose,segmenter,dict(defaults(),enabled=True))
    with Image.new('RGB',(50,50)) as image:
        result = model.predict(image,[[0,0,50,50]])
    np.testing.assert_array_equal(result[0],xy)
    assert result[3][0]['status'] == 'masked'
    assert result[3][0]['visibility_measured'] is False
    assert result[3][0]['overlap_mask']
