"""Read-only alignment-audit CLI and pixel-accurate diagnostic tests."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.alignment_audit import (
    _coverage, _diagnostics, _sampled_indices, execute,
)
from stash_ai.recognition.aura import similarity_matrix
from stash_ai.recognition.cli import parser
from stash_ai.recognition.config import load
from stash_ai.recognition.quality import (
    MIN_ALIGNMENT_FACE_COVERAGE, MIN_ALIGNMENT_SOURCE_COVERAGE, alignable,
)
from stash_ai.recognition.media import SpriteInput
from tests.helpers.recognition import FACE, Stash, textured_image


def config(tmp_path):
    path=tmp_path/'face.yml'
    path.write_text(
        'version: 1\n'
        'models: {detector: yunet, recognizer: lvface_b, device: cpu, download_missing: false}\n'
        'detection: {min_sharpness: 0, max_pose_deviation: null, '
        'min_visible_face_fraction: 0, require_complete_face: false}\n'
        'execution: {prefetch_frames: 0}\n')
    return load(path)


class Detector:
    def __init__(self):
        self.calls=0
        self.closed=False

    def detect(self, image):
        self.calls+=1
        return [FACE]

    def close(self):
        self.closed=True


def fake_services(detector, *, stash=None):
    stash=stash or Stash()
    calls=[]
    def assets(config, *, keys, write):
        calls.extend(keys)
        return {'yunet':'verified-yunet-only'}
    return SimpleNamespace(
        client=lambda **kwargs: stash, source=SpriteInput, assets=assets,
    ), calls, stash


def invoke(tmp_path, *, scene_ids=('1',), max_samples=2, cue_indices=None):
    cfg=config(tmp_path)
    destination=tmp_path/'recognition_alignment_audit'/'local'
    arguments=['audit-alignment']
    for scene in scene_ids:
        arguments += ['--scene-id',scene]
    arguments += ['--output-dir',str(destination),
                  '--max-samples-per-scene',str(max_samples)]
    if cue_indices is not None:
        for index in cue_indices:
            arguments += ['--cue-index',str(index)]
    parsed=parser().parse_args(arguments)
    detector=Detector()
    services, assets, stash=fake_services(detector)
    return cfg,destination,parsed,detector,services,assets,stash


def test_audit_parser_requires_explicit_scenes_and_directory(tmp_path):
    with pytest.raises(SystemExit):
        parser().parse_args(['audit-alignment','--scene-id','1'])
    with pytest.raises(SystemExit):
        parser().parse_args(['audit-alignment','--output-dir',str(tmp_path/'audit')])
    with pytest.raises(SystemExit):
        parser().parse_args(['audit-alignment','--scene-id','0','--output-dir',str(tmp_path/'audit')])
    got=parser().parse_args(['audit-alignment','--scene-id','1','--scene-id','4',
                             '--cue-index','20','--output-dir',str(tmp_path/'audit')])
    assert got.scene_id==['1','4'] and got.cue_index==[20]


def test_alignment_coverage_matches_quality_gate_exactly():
    with textured_image() as image:
        info,matrix,mask=_diagnostics(image,FACE)
        assert matrix.shape==(2,3)
        assert mask.shape==(112,112)
        assert 0<=info['padding_fraction']<=1
        assert info['source_coverage']+info['padding_fraction']==pytest.approx(1)
        assert info['face_region_coverage']+info['face_region_padding_fraction']==pytest.approx(1)
        assert info['alignment_valid']==alignable(FACE,image)
        assert info['alignment_valid']==(info['source_coverage']>=MIN_ALIGNMENT_SOURCE_COVERAGE
                                         and info['face_region_coverage']>=MIN_ALIGNMENT_FACE_COVERAGE)
        assert info['source_eye_distance_pixels'] == pytest.approx(
            np.linalg.norm(np.asarray(FACE.landmarks[0])-np.asarray(FACE.landmarks[1])))
        assert np.allclose(matrix,similarity_matrix(FACE.landmarks),atol=0,rtol=0)


def test_private_exact_source_aligned_mask_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv('STASH_URL','http://stash/')
    cfg,dst,args,detector,services,assets,stash=invoke(tmp_path)
    assert execute(args,cfg,services=services,detector_factory=lambda *_:detector)==0
    assert assets==['yunet'] and detector.closed
    assert detector.calls==3  # All three cues visited; only two saved.
    assert stash.writes==[] and stash.closed
    assert not cfg.database.exists()  # No SQLite ever created.
    manifest=json.loads((dst/'manifest.json').read_text())
    assert manifest['kind']=='recognition_alignment_audit'
    assert manifest['recognizer_selected_but_not_loaded']=='lvface_b'
    assert len(manifest['embedding_namespace'])==64
    assert len(manifest['expected_scan_policy'])==64
    scene=manifest['scenes'][0]
    assert scene['counters']=={'cues_examined':3,'faces_detected':3}
    assert scene['sampled_faces']==2
    assert scene['requested_sample_indices']==[0,2]
    assert len(scene['vtt_source_fingerprint'])==64
    assert len(scene['expected_recognition_scan_id'])==36
    assert scene['vtt_coverage']['selected_cues']==3
    assert scene['nonvisibility_rejections']=={'passed_nonvisibility_gates':3}
    assert [sample['cue_index'] for sample in scene['samples']]==[0,2]
    with textured_image() as image:
        import cv2
        bgr=np.ascontiguousarray(np.asarray(image)[:,:,::-1])
        aligned=cv2.warpAffine(bgr,similarity_matrix(FACE.landmarks),(112,112),borderValue=0.)
        rgb=cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB)
        sample=scene['samples'][0]
        with Image.open(dst/sample['images'][1]) as actual:
            assert np.array_equal(np.asarray(actual),rgb)
        with Image.open(dst/sample['images'][2]) as coverage:
            assert coverage.size==(112,112) and set(np.unique(coverage)) <= {0,255}
        with Image.open(dst/sample['images'][0]) as source:
            assert source.size==(160,160)
        assert sample['geometry']['face_min_dimension_pixels']==128
    if os.name=='posix':
        assert dst.stat().st_mode & 0o777 == 0o700
        assert (dst/'manifest.json').stat().st_mode & 0o777 == 0o600
        assert (dst/scene['samples'][0]['images'][1]).stat().st_mode & 0o777 == 0o600


def test_explicit_cue_index_does_not_change_sampling_index_base(tmp_path,monkeypatch):
    monkeypatch.setenv('STASH_URL','http://stash/')
    cfg,dst,args,detector,services,_,_=invoke(tmp_path,max_samples=8,cue_indices=[1])
    execute(args,cfg,services=services,detector_factory=lambda *_:detector)
    scene=json.loads((dst/'manifest.json').read_text())['scenes'][0]
    assert [sample['cue_index'] for sample in scene['samples']]==[1]
    assert scene['counters']['faces_detected']==3
    assert scene['sampling']=='explicit_indices'


def test_rejects_out_of_range_cues_and_never_creates_database(tmp_path,monkeypatch):
    monkeypatch.setenv('STASH_URL','http://stash/')
    cfg,dst,args,detector,services,_,_=invoke(tmp_path,cue_indices=[999])
    with pytest.raises(ValueError,match='has no selected cue index 999'):
        execute(args,cfg,services=services,detector_factory=lambda *_:detector)
    assert not cfg.database.exists()
    assert detector.closed


def test_rejects_existing_output_without_scanning_or_loading_models(tmp_path,monkeypatch):
    monkeypatch.setenv('STASH_URL','http://stash/')
    cfg,dst,args,detector,services,assets,_=invoke(tmp_path)
    dst.mkdir(parents=True)
    sentinel=dst/'existing.png'
    sentinel.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        execute(args,cfg,services=services,detector_factory=lambda *_:detector)
    assert sentinel.read_bytes()==b'keep'
    assert detector.calls==0 and assets==[]


def test_caps_and_even_sampling():
    assert _sampled_indices(3,8)=={0,1,2}
    assert _sampled_indices(11,3)=={0,5,10}
    assert _sampled_indices(11,1)=={5}
    cfg=SimpleNamespace()
    for count in (0,129):
        with pytest.raises(ValueError,match='max-samples'):
            execute(SimpleNamespace(scene_id=['1'],max_samples_per_scene=count,
                                    cue_index=None, output_dir='unused'),cfg)


def test_invalid_geometry_is_reported_without_a_fake_aligned_image():
    from stash_ai.recognition.models import checked_detection
    # Five identical landmark locations => a singular source transform.
    invalid=checked_detection([1,1,120,120,20,20,20,20,20,20,20,20,20,20,.99])
    with textured_image() as image:
        value,matrix,mask=_diagnostics(image,invalid)
    assert value['alignment_valid'] is False
    assert value['alignment_error']=='invalid_five_point_geometry'
    assert matrix is None and mask is None


def test_cli_main_audit_alignment_handles_missing_faces_db(tmp_path, monkeypatch):
    """Exercise public CLI dispatch, not execute() alone: audit has no --faces-db."""
    from stash_ai.recognition import cli, alignment_audit

    cfg_path = tmp_path / 'face.yml'
    cfg_path.write_text('version: 1\nmodels: {device: cpu}\n')
    output = tmp_path / 'private-audit'
    seen = []

    def fake_audit(args, settings, *, services):
        seen.append((args, settings, services))
        return 0

    marker = object()
    monkeypatch.setattr(alignment_audit, 'execute', fake_audit)
    code = cli.main(['audit-alignment', '--scene-id', '1',
                     '--face-config', str(cfg_path),
                     '--output-dir', str(output), '--device', 'cpu'],
                    services=marker)
    assert code == 0
    assert len(seen) == 1
    args, settings, services = seen[0]
    assert not hasattr(args, 'faces_db')
    assert args.scene_id == ['1']
    assert args.output_dir == output
    assert settings.source_path == cfg_path.resolve()
    assert settings.data['models']['device'] == 'cpu'
    assert services is marker
    assert not settings.database.exists()  # Audit dispatch creates no SQLite.


def test_cli_main_report_preserves_explicit_faces_db(tmp_path, monkeypatch):
    """The shared getter must still forward the existing DB override."""
    from stash_ai.recognition import cli, fragmentation_report

    file = tmp_path / 'custom.sqlite3'
    received = []

    def fake_report(args, settings):
        received.append(settings.database)
        return 0

    monkeypatch.setattr(fragmentation_report, 'execute', fake_report)
    assert cli.main(['report', '--scene-id', '1', '--faces-db', str(file)]) == 0
    assert received == [file.resolve()]
    assert not file.exists()
