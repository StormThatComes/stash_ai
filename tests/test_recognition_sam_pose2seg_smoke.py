"""Opt-in real CPU PMPose -> SAM-pose2seg -> mask-conditioned PMPose test."""
import os
import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_SAM_POSE2SEG_SMOKE') != '1', reason='explicit ~975 MB two-model smoke')
def test_pretrained_mask_refinement_and_expanded_geometry_contract(tmp_path, request):
    import hashlib
    from io import BytesIO
    import json
    from pathlib import Path
    import numpy as np
    import requests
    import torch
    import yaml
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.pmpose_pose import load as load_pose
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces

    previous=torch.get_num_threads()
    request.addfinalizer(lambda:torch.set_num_threads(previous))
    torch.set_num_threads(2)
    reference=json.loads((Path(__file__).parent/'fixtures/pmpose_b_reference.json').read_text())
    with requests.get('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/'
                      'lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg',timeout=30) as response:
        response.raise_for_status();raw=response.content
    assert hashlib.sha256(raw).hexdigest()==reference['provenance']['image_sha256']
    path=tmp_path/'recognition.yml'
    path.write_text(yaml.safe_dump(dict(version=1,models=dict(device='cpu'),sapiensid=dict(
        pose=dict(backend='pmpose',pmpose_directory=os.getenv('STASH_AI_PMPOSE_DIRECTORY',''),
            refinement=dict(enabled=True,directory=os.getenv('STASH_AI_SAM_POSE2SEG_DIRECTORY','')))))))
    config=load(path)
    model=load_pose(config,'cpu')
    bounds=[reference['provenance']['bounds']]
    with Image.open(BytesIO(raw)).convert('RGB') as image:
        points,scores,probabilities,records=model.predict(image,bounds)
        assert points.shape==(1,17,2) and scores.shape==(1,17)
        assert all(np.isfinite(value).all() for value in (points,scores,*probabilities.values()))
        assert records[0]['status']=='refined'
        assert records[0]['mask_pixels']>0 and records[0]['mask_score']>0
        assert model.segmenter.predictor._features is None
        # The production consumer uses the same refined geometry in both audit
        # and collection; the original detector score remains unchanged.
        detector=object.__new__(SapiensIDFaces);detector.config=config
        person=detector._people(bounds,[.99],points,scores,probabilities,records)[0]
        assert person['bounds']==records[0]['bounds'] and person['score']==.99
        assert person['pose_refinement']['detector_bounds']==bounds[0]
