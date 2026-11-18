from types import SimpleNamespace
import numpy as np
from PIL import Image
from stash_ai.recognition.pose_context import contexts
from stash_ai.recognition.perspective_shots import advance
from stash_ai.recognition.pose_refinement_config import defaults
from tests.test_recognition_pose_refinement import prediction, pose_points


def test_neighbour_proof_is_identical_across_batch_and_resume_boundaries():
    class Source:
        cues = range(3)
        def frames_indices(self, indices):
            for index in indices:
                yield index,None,Image.new('RGB',(100,100))
    class Model:
        config=SimpleNamespace(data={'sapiensid':{'pose':{'deduplication':{'enabled':False}}}})
        def detect(self, image, *, faces, force_pose, _defer):
            _defer.append((image,faces,[[0,0,100,100]], [.9]))
    raw=prediction([pose_points()])
    owner=SimpleNamespace(settings=defaults(),pose=SimpleNamespace(predict_many=lambda frames:[raw for _ in frames]))
    images=[Image.new('RGB',(100,100)) for _ in range(3)]
    try:
        previous=None
        for cue,image in enumerate(images):
            previous=advance(image,cue,previous,evidence=True)
            image._recognition_cue_signature=previous
            image._recognition_scan='s'
            image._recognition_pose_source=Source(),Model()
        whole=contexts(owner,[(im,[[0,0,100,100]]) for im in images],[raw]*3)
        resumed=contexts(owner,[(images[1],[[0,0,100,100]])],[raw])
        for cue in range(3):
            assert whole['s',cue][0][:4]==resumed['s',cue][0][:4]
            assert whole['s',cue][1][0][0]==resumed['s',cue][1][0][0]
            for joint,value in whole['s',cue][1][0][1].items():
                np.testing.assert_array_equal(value,resumed['s',cue][1][0][1][joint])
    finally:
        for image in images:image.close()
