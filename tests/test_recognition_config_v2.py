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
    baseline = load(root=tmp_path)
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
