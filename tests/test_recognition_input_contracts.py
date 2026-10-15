import numpy as np
from tests.test_recognition_clustering import setup as cluster_config
import json
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_run import run
from tests.helpers.recognition import Backend
import pytest


def test_alignment_requires_real_source_pixels_with_pose_disabled(tmp_path):
    import cv2
    from stash_ai.recognition.quality import assess,rejection
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition.aura import similarity_matrix
    from tests.helpers.recognition import FACE,textured_image
    geometry=list(FACE.row)
    for index in range(4,14):geometry[index]+=2000
    face=checked_detection(geometry)
    config=cluster_config(tmp_path)
    with textured_image() as image:
        quality=assess(image,face)
        assert rejection(quality,config.data['detection']) == 'face_landmarks_invalid'
        aligned=cv2.warpAffine(np.asarray(image),similarity_matrix(face.landmarks),(112,112))
        assert not aligned.any() and not quality['alignment_valid']


def test_video_portrait_skips_unalignable_geometry_before_embedding(tmp_path,monkeypatch):
    from tests.test_recognition_metadata import setup
    from tests.helpers.recognition import FACE
    from stash_ai.recognition.models import checked_detection
    config,client,services=setup(tmp_path,monkeypatch)
    model=Backend();detected=[0]
    values=list(FACE.row);values[4:14]=[80.]*10
    broken=checked_detection(values)
    def detect(image):
        detected[0]+=1
        return [FACE if detected[0]<=3 else broken]
    model.detect=detect;services.backend=lambda *a:model
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        metadata=json.loads(state.connection.execute("SELECT metadata FROM portraits WHERE status='published'").fetchone()[0])
        assert metadata['source_type']=='vtt'
        assert model.calls == 3


def test_complex_embeddings_are_rejected_before_lossy_conversion():
    from stash_ai.recognition.models import normalized_embedding
    with pytest.raises(ValueError, match='real numeric'):
        normalized_embedding(np.array([1+100j,0]),2)


def test_optional_quality_cache_rejects_oversized_numbers():
    from stash_ai.recognition.visibility import cached, CONTRACT_ID
    from stash_ai.recognition.quality import portrait_components
    from stash_ai.recognition.config import defaults
    huge=10**1000
    assert not cached({'visibility':{'contract':CONTRACT_ID},'visible_face_fraction':huge})
    assert portrait_components({'sharpness':huge},defaults()['portraits'])['sharpness'] == 0.


def test_pair_evaluator_reports_oversized_threshold_as_invalid():
    from stash_ai.recognition.evaluation import evaluate_pairs
    with pytest.raises(ValueError):evaluate_pairs([],10**1000)
