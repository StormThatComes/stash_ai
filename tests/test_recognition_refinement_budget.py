from types import SimpleNamespace
from PIL import Image
from stash_ai.recognition.refinement_budget import Budget


def test_scene_retry_envelope_is_identical_for_resume_and_split_batches():
    source=SimpleNamespace(cues=range(3))
    settings=dict(max_retry_jobs_per_cue=16,max_retry_jobs_per_scene=5)
    limits=[]
    with Image.new('RGB',(1,1)) as image:
        image._recognition_pose_source=source,None
        for cue in (0,1,2,1):
            image._recognition_cue_signature=(cue,0,'')
            budget=Budget(image,settings);limits.append(budget.remaining)
            assert budget.admit('sam_decoder')
            if cue==2:
                record={}
                assert not budget.admit('masked_pose',record=record)
                assert record['retry_skip_reason']=='shared_retry_budget'
    assert limits==[2,2,1,2]
