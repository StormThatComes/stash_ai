import numpy as np
from PIL import Image
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults
from tests.test_recognition_profile3_batching import BatchedPose, BatchedSegmenter, BOX


class RetainedFeatures(BatchedSegmenter):
    def refine_frames(self, frames, process):
        self.embeddings = 0
        self.retries = []
        outcomes = []
        for image, requests in frames:
            self.embeddings += 1
            blank = np.zeros((image.height,image.width),bool)
            target = blank.copy()
            target[10:90,20:60] = True
            def decode(positive, negative, multi):
                assert multi
                self.retries.append(positive.tolist())
                return [(blank, .99), (target, .90)]
            outcomes.append(process(requests, [(blank,.95) for _ in requests], decode))
        return outcomes

    def frame(self, image):
        raise AssertionError('retry re-encoded an image')


def test_retry_reuses_live_features_and_ranks_mask_ownership_before_score():
    model = RefinedPMPose(BatchedPose(), RetainedFeatures(), dict(defaults(),adaptive_decoder_retries=True))
    with Image.new('RGB',(160,100)) as image:
        a,b = model.predict_many([(image,[BOX]),(image,[BOX])])
    for row in (a,b):
        record = row[3][0]
        assert record['mask_attempts'] == 2 and record['retry_mask_choices'] == 2
        assert record['mask_score'] == .9 and record['status'] == 'refined'
    assert model.segmenter.embeddings == 2 and len(model.segmenter.retries) == 2


def test_identical_multimask_retry_is_not_decoded_twice():
    from tests.test_recognition_pose_refinement import Segmenter
    sam = Segmenter()
    calls = []
    def fail(positive, negative=()):
        calls.append(positive.copy())
        return np.zeros((100,160),bool),.2
    sam.predict = fail
    sam.predict_options = lambda p,n: [fail(p,n)]
    model = RefinedPMPose(BatchedPose(),sam,dict(defaults(),adaptive_decoder_retries=True,
                                                 min_prompt_points=1,max_prompt_points=1))
    with Image.new('RGB',(160,100)) as image:
        record = model.predict(image,[BOX])[3][0]
    assert len(calls) == record['mask_attempts'] == 2
