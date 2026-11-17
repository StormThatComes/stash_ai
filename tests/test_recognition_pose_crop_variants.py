import numpy as np
from PIL import Image
from types import SimpleNamespace
from stash_ai.recognition.pose_crop_variants import Variant


def test_sideways_variant_restores_source_points():
    def predict(frames):
        image, boxes = frames[0]
        assert image.size == (80,100)
        assert boxes == [[20.,50.,60.,90.]]
        xy=np.tile([30.,79.],(1,17,1))
        return [(xy,np.ones((1,17)),dict(presence=np.ones((1,17)),visibility=np.ones((1,17))))]
    with Image.new('RGB',(100,80)) as image:
        result=Variant(SimpleNamespace(predict_many=predict),'sideways').predict_many([(image,[[10,20,50,60]])])[0]
    np.testing.assert_array_equal(result[0][0,0],[20,30])
