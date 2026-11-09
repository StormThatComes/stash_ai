"""The desktop preview owns its pixels and single-cue inference matches the CLI."""
import json

import pytest
from PIL import Image

from stash_ai.recognition.detection_parameters import apply_edits
from stash_ai.recognition.detection_session import CueDetector, open_scene, read_cue
from tests.test_recognition_detection_audit import Detector, run, setup


def test_browsing_and_rendering_match_single_cue_cli(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.args.cue_index = 1
    run(case)
    record = json.loads((case.destination / 'manifest.json').read_text())['scene']['frames'][0]
    engine = CueDetector(case.services, factory=lambda *_: Detector())
    case.requested.clear()
    with open_scene(case.services, case.cfg, '1') as source:
        for index in (2, 0, 2, 1):
            frame = read_cue(source, '1', index)
            if index != 1:
                frame.image.close()
        assert not case.requested
        with pytest.raises(ValueError, match='valid indices'):
            read_cue(source, '1', -1)
    with frame.image, Image.open(case.destination / record['file']) as expected:
        result, actual = engine.render(frame, case.cfg)
        with result:
            assert result.tobytes() == expected.tobytes()
        assert actual == record
    engine.close()
    assert not case.stash.writes


def test_apply_combined_edits_preserves_shared_overrides(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    edits = {'pose.estimator.backend': 'pmpose', 'pose.refinement.enabled': True,
             'pose.refinement.min_visibility': .7, 'face.detector.score_threshold': .8}
    variant, _ = apply_edits(case.cfg, edits)
    next_config, _ = apply_edits(variant, {'pose.point_quality.min_visibility': .1})
    pose = next_config.data['sapiensid']['pose']
    assert pose['refinement']['min_visibility'] == .7
    assert pose['deduplication']['min_visibility'] == .1
    assert next_config.data['detection']['score_threshold'] == .8
    assert case.cfg.data['detection']['score_threshold'] == .9
