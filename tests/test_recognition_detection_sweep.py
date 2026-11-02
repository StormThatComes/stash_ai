"""Single-cue selection, independent fresh trials and pixel-accurate stacked exports."""
from dataclasses import replace
import json
import os

from PIL import Image
import pytest

from stash_ai.recognition import detection_audit, detection_sweep
from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.config import plain
from tests.helpers.recognition import FACE, textured_image
from tests.test_recognition_detection_audit import Detector, PERSON, setup, run


def table(case, text, suffix='.yml'):
    path = case.destination.parent / ('sweep' + suffix)
    path.write_text(text, encoding='utf-8')
    case.args.sweep = path
    case.args.cue_index = 1
    return path


class ConfiguredDetector(Detector):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def detect(self, image, *, force_pose=False):
        assert force_pose
        self.calls += 1
        with textured_image() as original:
            assert image.tobytes() == original.tobytes()
        self.frame_people = [PERSON]
        faces = [FACE] if self.config.data['detection']['score_threshold'] < FACE.score else []
        self.face_matches = [0] * len(faces)
        return faces


def sweep_runner(case, *, fail_variant=None):
    detectors, configs = [], []
    case.services.assets = lambda cfg, **kwargs: {}

    def factory(cfg, paths):
        assert all(detector.closed for detector in detectors)
        configs.append(cfg)
        detector = ConfiguredDetector(cfg)
        if len(detectors) == fail_variant:
            def fail(*args, **kwargs):
                raise RuntimeError('synthetic trial failure')
            detector.detect = fail
        detectors.append(detector)
        return detector

    def execute():
        return detection_audit.execute(case.args, case.cfg, services=case.services, detector_factory=factory)
    return execute, detectors, configs


@pytest.mark.parametrize('extra', [['--cue-index', '-1'], ['--sweep', 'table.yml']])
def test_cli_rejects_invalid_selection(extra):
    with pytest.raises(SystemExit):
        parser().parse_args(['audit-detection', '--scene-id', '1', '--output-dir', 'out', *extra])


@pytest.mark.parametrize('index', [0, 2])
def test_single_cue_keeps_original_index_and_only_runs_that_cue(tmp_path, monkeypatch, index):
    case = setup(tmp_path, monkeypatch)
    case.args.cue_index = index
    from stash_ai.recognition.media import SpriteInput
    selected = []
    original = SpriteInput._cue_image

    def cue_image(self, cue_index, **kwargs):
        selected.append(cue_index)
        return original(self, cue_index, **kwargs)

    monkeypatch.setattr(SpriteInput, '_cue_image', cue_image)
    assert run(case) == 0
    assert selected == [index]
    assert case.detector.calls == 1
    manifest = json.loads((case.destination / 'manifest.json').read_text())
    assert manifest['scene']['frames'][0]['cue_index'] == index
    assert manifest['execution']['cue_index'] == index
    assert [p.name for p in case.destination.glob('*.png')] == [f'scene-1_cue-{index:06d}.png']


def test_out_of_range_cue_fails_before_models_or_output(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.args.cue_index = 3
    with pytest.raises(ValueError, match=r'valid indices: 0\.\.2'):
        run(case)
    assert not case.requested and not case.destination.exists()
    assert case.stash.closed and case.sources[0].cache == {}


@pytest.mark.parametrize('suffix,text', [
    ('.yml', '- {parameter: detection.score_threshold, value: 0.7}\n'
             '- {parameter: sapiensid.pose.refinement.enabled, value: true}\n'),
    ('.csv', '\ufeffparameter,value\ndetection.score_threshold,0.7\n'
             'sapiensid.pose.refinement.enabled,true\n'),
])
def test_typed_tables_use_independent_baselines(tmp_path, monkeypatch, suffix, text):
    case = setup(tmp_path, monkeypatch)
    data = plain(case.cfg.data)
    data['sapiensid']['pose']['backend'] = 'pmpose'
    case.cfg = replace(case.cfg, data=data)
    original = plain(case.cfg.data)
    variants = detection_sweep.load_variants(table(case, text, suffix), case.cfg)
    assert variants[0][0] is case.cfg
    assert variants[1][0].data['detection']['score_threshold'] == .7
    assert variants[2][0].data['detection']['score_threshold'] == .9
    assert variants[2][0].data['sapiensid']['pose']['refinement']['enabled'] is True
    assert variants[2][1]['baseline_value'] is False
    assert plain(case.cfg.data) == original


@pytest.mark.parametrize('text', [
    '[]', '{}', '- {parameter: detection.typo, value: 0.5}',
    '- {parameter: detection.min_face_pixels, value: 100}',  # Not a detector gate.
    '- {parameter: sapiensid.pose.typo, value: 0.5}',
    '- {parameter: sapiensid.pose.refinement, value: {enabled: true}}',
    '- {parameter: sapiensid.pose.score_threshold.typo, value: 0.5}',
    '- {parameter: detection.score_threshold, value: -0.1}',
    '- {parameter: detection.score_threshold, value: .nan}',
    '- {parameter: detection.score_threshold, value: "0.5"}',
    '- {parameter: detection.max_faces_per_frame, value: true}',
    '- {parameter: sapiensid.pose.refinement.enabled, value: true}',  # Needs PMPose.
    '- {parameter: models.detector, value: typo}',
    '- {parameter: detection.nms_threshold, value: 0.5, typo: 1}',
    '- {parameter: detection.nms_threshold, value: 0.5, value: 0.6}',
    '- {parameter: recognition.match_threshold, value: 0.5}',
    '- {parameter: sapiensid.pose.backend, value: []}',
    '- {parameter: sapiensid.pose.refinement.directory, value: {bad: value}}',
    '- {parameter: detection.nms_threshold, value: 0.5}\n' * 65,
])
def test_invalid_table_fails_before_network_models_and_output(tmp_path, monkeypatch, text):
    case = setup(tmp_path, monkeypatch)
    table(case, text)
    with pytest.raises(ValueError):
        run(case)
    assert not case.stash.reads and not case.requested and not case.destination.exists()


@pytest.mark.parametrize('text', ['value,parameter\n0.5,detection.nms_threshold',
                                'parameter,value\ndetection.nms_threshold,',
                                'parameter,value\ndetection.nms_threshold,0.5,extra'])
def test_invalid_csv_is_rejected(tmp_path, monkeypatch, text):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        detection_sweep.load_variants(table(case, text, '.csv'), case.cfg)


def test_sweep_stacks_fresh_independent_trials_with_headers_and_geometry(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    table(case, 'parameter,value\ndetection.score_threshold,1.0\ndetection.nms_threshold,0.6\n', '.csv')
    before = plain(case.cfg.data)
    execute, detectors, configs = sweep_runner(case)
    assert execute() == 0
    assert len(detectors) == 3
    assert all(d.closed and d.calls == d.releases == 1 for d in detectors)
    assert [c.data['detection']['score_threshold'] for c in configs] == [.9, 1., .9]
    assert [c.data['detection']['nms_threshold'] for c in configs] == [.3, .3, .6]
    assert plain(case.cfg.data) == before
    assert case.stash.closed and not case.stash.writes
    assert case.sources[0].cache == {} and not list(tmp_path.glob('*.sqlite3*'))
    assert len(list(case.destination.glob('*.png'))) == 1
    assert len(list(case.destination.iterdir())) == 2  # No temporary panels left over.
    manifest = json.loads((case.destination / 'manifest.json').read_text())
    assert manifest['kind'] == 'recognition_detection_sweep'
    assert manifest['format'] == 4
    scene = manifest['scene']
    assert scene['counts'] == dict(cues_exported=1, variants_exported=3, faces_detected=2, people_detected=3)
    records = scene['frames']
    assert [r['variant_index'] for r in records] == [0, 1, 2]
    assert [r['parameter'] for r in records] == [None, 'detection.score_threshold', 'detection.nms_threshold']
    assert all(r['cue_index'] == 1 and r['start_seconds'] == 1 for r in records)
    assert records[0]['output_dimensions'][0] > records[1]['output_dimensions'][0]  # Face gutter only when needed.
    assert any('baseline: 0.9' in line for line in records[1]['header_lines'])
    with Image.open(case.destination / scene['file']) as combined:
        assert list(combined.size) == scene['output_dimensions']
        for record, cfg in zip(records, configs):
            # Compare the entire panel byte-for-byte with the ordinary audit renderer.
            with textured_image() as image, detection_audit.render_frame(
                    image, ConfiguredDetector(cfg), '1', 1, case.sources[0].cues[1]) as (expected, _):
                x, y = record['source_offset']
                with combined.crop((x, y, x + expected.width, y + expected.height)) as actual:
                    assert actual.tobytes() == expected.tobytes()
        assert records[1]['panel_offset'][1] > records[0]['source_offset'][1] + 160
    if os.name == 'posix':
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in case.destination.iterdir())


def test_cli_sweep_uses_detector_only_factory(tmp_path, monkeypatch):
    from stash_ai.recognition import sapiensid_backend
    case = setup(tmp_path, monkeypatch)
    path = table(case, '- {parameter: detection.score_threshold, value: 0.8}')
    captured = []
    def factory(cfg, paths, **kwargs):
        captured.append(kwargs)
        return ConfiguredDetector(cfg)
    monkeypatch.setattr(sapiensid_backend, 'SapiensIDFaces', factory)
    case.services.assets = lambda *args, **kwargs: {}
    assert main(['audit-detection', '--scene-id', '1', '--cue-index', '1', '--sweep', str(path),
                 '--face-config', str(tmp_path / 'face.yml'), '--device', 'cpu', '--progress-style', 'off',
                 '--output-dir', str(case.destination)], services=case.services) == 0
    assert captured == [{'load_identity': False}] * 2


def test_failed_trial_closes_resources_removes_panels_and_never_claims_complete(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    table(case, '- {parameter: detection.score_threshold, value: 0.8}')
    execute, detectors, _ = sweep_runner(case, fail_variant=1)
    with pytest.raises(RuntimeError, match='synthetic trial failure'):
        execute()
    assert len(detectors) == 2 and all(d.closed for d in detectors)
    assert case.stash.closed and case.sources[0].cache == {}
    assert list(case.destination.iterdir()) == []


def test_canvas_limit_fails_without_partial_png_or_manifest(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    table(case, '- {parameter: detection.score_threshold, value: 0.8}')
    monkeypatch.setattr(detection_sweep, 'MAX_PIXELS', 100)
    execute, detectors, _ = sweep_runner(case)
    with pytest.raises(ValueError, match='use fewer trials'):
        execute()
    assert all(d.closed for d in detectors) and case.stash.closed
    assert list(case.destination.iterdir()) == []
