import json
from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.quality import alignment_measurements
from stash_ai.recognition.reference_selection import quality
from stash_ai.recognition.visibility import CONTRACT_ID


def test_visibility_and_alignment_reduce_quality_without_rejecting_old_evidence():
    raw = dict(pixels=200, sharpness=400, pose_deviation=.1)
    baseline = quality(json.dumps(raw))
    assert baseline > 0
    assert quality(json.dumps(dict(raw, alignment_error=.1))) < baseline
    assert quality(json.dumps(dict(raw, visible_face_fraction=.4,
                                  visibility=dict(contract=CONTRACT_ID)))) < baseline
    assert quality(json.dumps(dict(raw, alignment_valid=False))) == 0


def test_alignment_residual_is_invariant_to_source_scale_and_translation():
    landmarks=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                        [41.5493,92.3655],[70.7299,92.2041]])
    first=alignment_measurements(SimpleNamespace(landmarks=landmarks))
    second=alignment_measurements(SimpleNamespace(landmarks=landmarks*2+15))
    assert first['alignment_error'] < 1e-6 and second['alignment_error'] < 1e-6
    assert abs(second['alignment_scale']-.5) < 1e-6
    landmarks[2,0]+=12
    assert alignment_measurements(SimpleNamespace(landmarks=landmarks))['alignment_error'] > .01
