from pathlib import Path

import pytest
import yaml

from stash_ai.recognition import body_observations, person_signal
from stash_ai.recognition.cli import main
from stash_ai.recognition.config import defaults, load, plain
from stash_ai.recognition.config_layout import INTERNAL_PATHS, PUBLIC_PATHS, leaf_items, v2_document
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity


def test_layout_has_a_unique_path_for_every_internal_setting():
    paths = {k for k, _ in leaf_items(defaults())} - {'version'}
    assert set(INTERNAL_PATHS) == paths | {'sapiensid.pose.vitpose_batch_size'}
    assert len(INTERNAL_PATHS) == len(PUBLIC_PATHS)


@pytest.mark.parametrize('partial', [False, True])
def test_migration_preserves_effective_policy_and_scan_contracts(tmp_path, partial):
    baseline = defaults()
    baseline['sapiensid']['pose'].update(detector='rfdetr_xl', backend='vitpose_plus_large', batch_size=7)
    baseline['sapiensid']['enabled'] = True
    baseline['sapiensid']['collection']['enabled'] = True
    document = {'version': 1, 'sapiensid': baseline['sapiensid']} if partial else baseline
    if partial:
        document['recognition'] = {'min_support': 2}
    original = tmp_path / 'old.yml'
    original.write_text(yaml.safe_dump(document))
    migrated = tmp_path / 'new.yml'
    assert main(['migrate-config', '--face-config', str(original), '--output', str(migrated)]) == 0
    old, new = load(original), load(migrated)
    assert new.document_version == 2
    assert plain(old.data) == plain(new.data)
    assert namespace(new) == namespace(old)
    assert scan_identity(new, namespace(new)) == scan_identity(old, namespace(old))
    assert person_signal.contract(new) == person_signal.contract(old)
    assert body_observations.contract(new) == body_observations.contract(old)
    assert new.database == old.database and not new.database.exists()
    assert not (tmp_path / 'recognition_models').exists()
    with pytest.raises(FileExistsError):
        main(['migrate-config', '--face-config', str(original), '--output', str(migrated)])


def test_migration_rebases_all_relative_paths_when_exporting_elsewhere(tmp_path):
    source = tmp_path / 'config.yml'
    source.write_text('version: 1\nnames: {file: names.yml}\nsapiensid:\n  pose:\n'
                      '    refinement: {directory: assets/sam}\n    rfdetr_directory: weights\n')
    dest = tmp_path / 'exports' / 'new.yml'
    main(['migrate-config', '--face-config', str(source), '--output', str(dest)])
    before, after = load(source), load(dest)
    assert (after.database, after.model_dir, after.names_path) == (before.database, before.model_dir, before.names_path)
    pose = after.data['sapiensid']['pose']
    assert Path(pose['refinement']['directory']) == tmp_path / 'assets/sam'
    assert Path(pose['rfdetr_directory']) == tmp_path / 'weights'


def test_frame_and_vitpose_batches_are_independent_and_do_not_invalidate_evidence(tmp_path):
    path = tmp_path / 'batch.yml'
    path.write_text('version: 2\npose:\n  detector: {frame_batch_size: 8}\n'
                    '  estimator: {vitpose_plus_large: {batch_size: 2}}\n')
    config = load(path)
    assert config.data['sapiensid']['pose']['batch_size'] == 8
    assert config.data['sapiensid']['pose']['vitpose_batch_size'] == 2
    path.write_text('version: 2\n')
    baseline = load(path)
    assert person_signal.contract(config) == person_signal.contract(baseline)
    exported = v2_document(config.data)
    assert exported['pose']['estimator']['vitpose_plus_large']['batch_size'] == 2


@pytest.mark.parametrize('body', [
    'pose: {foo: {}}', 'pose: {detector: {bogus: 1}}', 'face.detector: {}',
    'pose: {estimator: {vitpose_plus_large: {batch_size: true}}}',
    'matching: {face: {samples: false}}', 'recognition: {min_support: 1}',
])
def test_v2_rejects_unknown_shapes_and_values(tmp_path, body):
    path = tmp_path / 'bad.yml'
    path.write_text('version: 2\n' + body)
    with pytest.raises(ValueError):
        load(path)


def test_partial_v2_pose_defaults_and_explicit_v1_opt_out(tmp_path):
    path = tmp_path / 'policy.yml'
    path.write_text('version: 2\n')
    config = load(path)
    assert body_observations.enabled(config)
    assert config.data['sapiensid']['pose']['detector'] == 'rfdetr_xl'
    assert config.data['sapiensid']['pose']['backend'] == 'vitpose_plus_large'
    path.write_text('version: 2\nbody: {enabled: false}\n')
    assert not body_observations.enabled(load(path))
    path.write_text('version: 1\n')
    legacy = load(path)
    assert not body_observations.enabled(legacy)
    assert legacy.data['sapiensid']['pose']['detector'] == 'yolo'


def test_point_quality_inheritance_and_explicit_local_overrides_survive_export(tmp_path):
    path = tmp_path / 'source.yml'
    path.write_text('version: 2\npose:\n  point_quality: {min_presence: 0.6, min_visibility: 0.9}\n'
                    '  refinement: {min_presence: 0.6, min_visibility: null}\n'
                    '  deduplication: {min_visibility: 0.3}\n')
    exported = tmp_path / 'new.yml'
    main(['migrate-config', '--face-config', str(path), '--output', str(exported)])
    config = load(path)
    assert plain(load(exported).data) == plain(config.data)
    document = yaml.safe_load(exported.read_text())
    pose = document['pose']
    assert pose['refinement']['min_presence'] == .6
    assert pose['refinement']['min_visibility'] is None
    assert pose['deduplication']['min_presence'] is None
    pose['point_quality']['min_presence'] = .2
    exported.write_text(yaml.safe_dump(document))
    effective = load(exported).data['sapiensid']['pose']
    assert effective['refinement']['min_presence'] == .6
    assert effective['deduplication']['min_presence'] == .2


@pytest.mark.parametrize('setting', ['min_presence: 1.1', 'min_presence: null', 'min_score: 0.2'])
def test_shared_thresholds_are_validated_even_when_both_stages_override(tmp_path, setting):
    path = tmp_path / 'bad.yml'
    path.write_text(f'version: 2\npose:\n  point_quality: {{{setting}}}\n'
                    '  deduplication: {min_presence: 0.2}\n  refinement: {min_presence: 0.3}\n')
    with pytest.raises(ValueError, match='pose.point_quality'):
        load(path)


def test_invalid_v2_value_reports_its_public_path(tmp_path):
    path = tmp_path / 'invalid.yml'
    path.write_text('version: 2\nexecution: {scan: {cpu_workers: 0}}\n')
    with pytest.raises(ValueError, match=r'execution\.scan\.cpu_workers'):
        load(path)


def test_shipped_run_collects_faceless_cues_once(tmp_path, monkeypatch):
    from stash_ai.recognition.database import FaceDatabase
    from tests.helpers.recognition import Backend
    from tests.test_recognition_body_collection import Persons
    from tests.test_recognition_run import setup

    _, client, services, _ = setup(tmp_path, monkeypatch)
    template = Path(__file__).resolve().parents[1] / 'facial_recognition.yml'
    settings = tmp_path / 'facial_recognition.yml'
    settings.write_bytes(template.read_bytes())
    face = Backend()
    face.detect = lambda image: []
    services.backend = lambda *args: face
    person = Persons(1)

    def body_model(config, *args, **kwargs):
        pose = config.data['sapiensid']['pose']
        assert (pose['detector'], pose['backend']) == ('rfdetr_xl', 'vitpose_plus_large')
        return person

    monkeypatch.setattr('stash_ai.recognition.sapiensid_backend.SapiensIDFaces', body_model)
    argv = ['run', '--face-config', str(settings), '--scene-id', '1', '--progress-style', 'off']
    assert main(argv, services=services) == 0
    assert person.calls == 3
    assert main(argv, services=services) == 0
    assert person.calls == 3
    with FaceDatabase(load(settings).database) as state:
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0
        assert state.connection.execute('SELECT COUNT(*) FROM body_cues').fetchone()[0] == 3
        assert state.connection.execute('SELECT COUNT(*) FROM body_observations').fetchone()[0] == 3
    assert not client.writes
