from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.aura import input_blob, runtime, similarity_matrix
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import ASSETS, ensure_assets, namespace
from tests.test_recognition_run import setup, run


def test_alignment_similarity_matches_known_affine_and_rejects_degeneracy():
    reference = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                          [41.5493,92.3655],[70.7299,92.2041]])
    angle = np.deg2rad(17)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = (reference @ rotation.T) * 2 + [17, 23]
    matrix = similarity_matrix(source)
    assert np.allclose(source @ matrix[:, :2].T + matrix[:, 2], reference, atol=1e-4)
    with pytest.raises(ValueError, match='degenerate'):
        similarity_matrix(np.zeros((5, 2)))


def test_rgb_normalization_nchw_is_exact_and_has_no_upstream_normalization_twice():
    bgr = np.empty((112,112,3), dtype='uint8')
    bgr[:] = [0, 127, 255]
    blob = input_blob(bgr)
    assert blob.shape == (1,3,112,112) and blob.dtype == np.float32 and blob.flags.c_contiguous
    assert blob[0,:,0,0] == pytest.approx([1, (127-127.5)/127.5, -1])
    with pytest.raises(ValueError):
        input_blob(np.zeros((128,128,3), dtype='uint8'))


@pytest.mark.parametrize('portraits', [False, True])
def test_only_selected_recognition_and_enabled_visibility_assets_are_downloaded(tmp_path, monkeypatch, portraits):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.runtime as face_runtime
    import stash_ai.recognition.models as models
    path = tmp_path / 'face.yml'
    path.write_text(f'version: 1\nmodels: {{recognizer: auraface}}\ndetection: {{min_visible_face_fraction: 0}}\nportraits: {{enabled: {str(portraits).lower()}}}\n')
    calls = []
    monkeypatch.setattr(face_runtime, 'runtime', lambda config: None)
    monkeypatch.setattr(models, 'verify', lambda *args: None)
    monkeypatch.setattr(models, 'verified_bytes', lambda *args: b'weights')
    monkeypatch.setattr('stash_ai.recognition.aura.verified_bytes', lambda *args: b'weights')
    monkeypatch.setattr(downloads, 'install_hf_asset', lambda repo, filename, revision, destination, **kwargs:
                        calls.append((repo, filename, revision)))
    ensure_assets(load(path), write=lambda text: None)
    expected = [ASSETS['yunet'].filename, 'glintr100.onnx'] + (['model.safetensors'] if portraits else [])
    assert [item[1] for item in calls] == expected
    assert all('gender' not in item[1] and '1k3' not in item[1] for item in calls)


def test_runtime_rejects_duplicate_packages_and_missing_cuda(tmp_path, monkeypatch):
    import stash_ai.recognition.runtime as face_runtime
    monkeypatch.setattr(face_runtime.metadata, 'version', lambda name: '1.26.0')
    with pytest.raises(ValueError, match='exactly one'):
        runtime(load(root=tmp_path))
    def version(name):
        if name == 'onnxruntime-gpu':
            raise face_runtime.metadata.PackageNotFoundError(name)
        return '1.26.0'
    monkeypatch.setattr(face_runtime.metadata, 'version', version)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider']))
    path = tmp_path / 'face.yml'
    path.write_text('version: 1\nmodels: {recognizer: auraface, device: cuda}')
    with pytest.raises(ValueError, match='no CUDA'):
        runtime(load(path))


def test_model_switch_retains_evidence_and_profiles_and_updates_scene_associations(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    old = client.links['1'][-1]
    path = tmp_path / 'aura.yml'
    path.write_text('version: 1\nmodels: {recognizer: auraface, device: cpu}\n'
                    'detection: {min_sharpness: 0, max_pose_deviation: null, min_visible_face_fraction: 0, require_complete_face: false}\nportraits: {enabled: false}\n')
    aura = replace(load(path), database=config.database)
    assert namespace(config) != namespace(aura)
    assert run(aura, services, '1') == 0
    assert client.links['1'][0] == '77' and old not in client.links['1']
    assert len(client.links['1']) == 2 and old in client.profiles
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 2
        assert db.connection.execute('SELECT COUNT(DISTINCT namespace) FROM models').fetchone()[0] == 2
        assert db.connection.execute('SELECT COUNT(*) FROM people WHERE performer IS NOT NULL').fetchone()[0] == 2
        assert not db.connection.execute("SELECT 1 FROM publications WHERE kind='namespace'").fetchone()
        assert all('/1.' in url for url in client.media_calls)


def test_local_pilot_uses_same_crops_without_rewriting_published_membership(tmp_path, monkeypatch):
    from stash_ai.recognition.pilot import evaluate_observations
    from tests.helpers.recognition import Backend
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        before = [tuple(row) for row in db.connection.execute('SELECT * FROM members')]
        row = db.connection.execute('SELECT id,scope FROM runs LIMIT 1').fetchone()
        labels = {r[0]: 'fixture-A' for r in db.connection.execute('SELECT id FROM observations')}
        results = evaluate_observations(db, row['id'], {'sface': config}, labels,
                                        assets=lambda config: {}, backend=lambda *args: Backend())
        assert results['sface']['pair_false_merge'] == results['sface']['pair_false_miss_including_withheld'] == 0
        assert results['sface']['accepted_coverage'] == 3
        assert [tuple(row) for row in db.connection.execute('SELECT * FROM members')] == before
        assert db.connection.execute("SELECT data FROM progress WHERE phase='pilot_comparison'").fetchone()


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('limit,legacy', [(1024, False), (1048576, False), (1048576, True)])
@pytest.mark.parametrize('recognizer', ['sface', 'auraface'])
def test_pilot_scales_landmarks_for_downscaled_retained_crops(tmp_path, monkeypatch, limit, legacy, recognizer):
    from tests.test_recognition_vtt_portraits import setup
    from tests.test_recognition_run import run
    from tests.test_recognition_rebuild import changed
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.pilot import evaluate_observations
    from stash_ai.recognition.portrait_availability import retained_image
    from tests.helpers.recognition import Backend
    config, client, services, _ = setup(tmp_path,monkeypatch,enabled=False)
    config = changed(config,storage={'max_crop_bytes':limit})
    assert run(config,services,'1') == 0
    seen = []
    class Inspect(Backend):
        def embed(self,image,face):
            seen.append((image.size,face.row))
            return super().embed(image,face)
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT * FROM observations ORDER BY id LIMIT 1').fetchone()
        if legacy:
            quality = json.loads(row['quality'])
            quality.pop('crop_encoding')
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
            db.connection.commit()
            row = db.connection.execute('SELECT * FROM observations WHERE id=?', (row['id'],)).fetchone()
        with retained_image(row)[0] as image:
            expected_size = image.size
        expected_image,expected_face = retained_image(row)
        expected_image.close()
        run_id = db.connection.execute('SELECT id FROM runs').fetchone()[0]
        evaluate_observations(db,run_id,{recognizer:changed(config, models={'recognizer': recognizer})},{row['id']:'A'},assets=lambda c:{},backend=lambda *a:Inspect())
        assert seen[0][0] == expected_size
        assert seen[0][1] == expected_face.row



import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_pilot_scene_order_matches_public_numeric_replay(tmp_path):
    from tests.test_recognition_clustering import setup,scan
    from tests.test_recognition_identity_merges import changed
    from tests.helpers.recognition import Backend
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.pilot import evaluate_observations
    from stash_ai.recognition.rebuild import rebuild_selected
    config = changed(setup(tmp_path),clustering={'max_samples_per_person':2})
    vectors = {'10':np.array([.8,.6,0,0]),'2':np.array([1.,0,0,0]),'3':np.array([.28,.96,0,0])}
    with FaceDatabase(config.database) as db:
        selected,labels,by_id = [],{},{}
        for position,(scene,vector) in enumerate(vectors.items(),1):
            identifier,_ = scan(db,config,scene,[vector]*3)
            selected.append({'scene':scene,'scan':identifier})
            for offset,row in enumerate(db.connection.execute('SELECT id FROM observations WHERE scan=? ORDER BY cue',(identifier,)).fetchall()):
                new_id = str(uuid.UUID(int=position*100+offset))
                db.connection.execute('UPDATE observations SET id=? WHERE id=?',(new_id,row['id']))
                labels[new_id],by_id[new_id] = scene,vector
            db.connection.commit()
        run_id = db.start_run('scope',list(vectors),{})
        embeddings = iter(by_id[x] for x in sorted(by_id))
        class Adapter(Backend):
            def embed(self,*a):return next(embeddings)
        measured = evaluate_observations(db,run_id,{'sface':config},labels,assets=lambda c:{},backend=lambda *a:Adapter())['sface']
        actual = rebuild_selected(db,selected,config)
        accepted = sum(r['observations'] for rows in actual.values() for r in rows if r['status']=='accepted')
        assert measured['accepted_coverage'] == accepted == 6
        assert measured['pair_false_merge'] == 0



import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_pilot_missing_optional_asset_retains_comparison(tmp_path, monkeypatch):
    from tests.test_recognition_run import setup,run
    from tests.test_recognition_rebuild import changed
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.pilot import evaluate_observations
    from tests.helpers.recognition import Backend
    config,client,services,_ = setup(tmp_path,monkeypatch)
    assert run(config,services,'1') == 0
    pilot_config = changed(config,portraits={'enabled':True})
    with FaceDatabase(config.database) as db:
        run_id = db.connection.execute('SELECT id FROM runs').fetchone()[0]
        labels = {r[0]:'A' for r in db.connection.execute('SELECT id FROM observations')}
        result = evaluate_observations(db,run_id,{'sface':pilot_config},labels,
                                  assets=lambda c:{'faceocc':tmp_path/'missing-optional-asset'},backend=lambda *a:Backend())['sface']
        assert result['embedding_coverage'] == 3 and result['artifact_bytes'] == 0
        assert result['unavailable_optional_artifacts'] == {'faceocc': 'FileNotFoundError'}
        assert db.connection.execute("SELECT 1 FROM progress WHERE phase='pilot_comparison'").fetchone() is not None



@pytest.mark.parametrize('asset', ['faceocc', 'expression_mesh', 'expression_blendshapes', 'yunet'])
@pytest.mark.parametrize('present', [False, True])
def test_pilot_accounts_for_unavailable_optional_and_fatal_required_assets(tmp_path, monkeypatch, asset, present):
    from tests.test_recognition_run import setup, run
    from tests.test_recognition_rebuild import changed
    from stash_ai.recognition.pilot import evaluate_observations
    from tests.helpers.recognition import Backend
    config, _, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    path = tmp_path / 'asset'
    if present:
        path.write_bytes(b'corrupt optional asset')
    with FaceDatabase(config.database) as db:
        run_id = db.connection.execute('SELECT id FROM runs').fetchone()[0]
        labels = {r[0]: 'A' for r in db.connection.execute('SELECT id FROM observations')}
        if asset == 'yunet':
            with pytest.raises((FileNotFoundError, ValueError)):
                evaluate_observations(db, run_id, {'sface': config}, labels,
                                      assets=lambda c: {asset: path}, backend=lambda *a: Backend())
        elif asset != 'yunet':
            configs = {'sface': config, 'auraface': changed(config, models={'recognizer': 'auraface'})}
            results = evaluate_observations(db, run_id, configs, labels,
                                           assets=lambda c: {asset: path}, backend=lambda *a: Backend())
            assert set(results) == set(configs)
            for result in results.values():
                assert result['artifact_bytes'] == 0
                assert asset in result['unavailable_optional_artifacts']
                assert result['accepted_coverage'] == 3


def test_pilot_rejects_inconsistent_retained_dimensions_without_changing_members(tmp_path, monkeypatch):
    from tests.test_recognition_run import setup, run
    from stash_ai.recognition.pilot import evaluate_observations
    from tests.helpers.recognition import Backend
    config, _, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        run_id = db.connection.execute('SELECT id FROM runs').fetchone()[0]
        row = db.connection.execute('SELECT id,quality FROM observations LIMIT 1').fetchone()
        quality = json.loads(row['quality'])
        quality['crop_encoding']['pixels'] = [1, 1]
        db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        db.connection.commit()
        before = [tuple(r) for r in db.connection.execute('SELECT * FROM members')]
        with pytest.raises(ValueError, match='dimensions'):
            evaluate_observations(db, run_id, {'sface': config}, {row['id']: 'A'}, assets=lambda c: {}, backend=lambda *a: Backend())
        assert before == [tuple(r) for r in db.connection.execute('SELECT * FROM members')]



def test_pilot_complete_artifact_accounting_verifies_all_supplied_bytes(tmp_path, monkeypatch):
    import hashlib
    import stash_ai.recognition.pilot as pilot
    from tests.test_recognition_run import setup, run
    from tests.helpers.recognition import Backend
    config, _, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    paths, pinned = {}, dict(pilot.ASSETS)
    for key in ('yunet', 'sface', 'faceocc', 'expression_mesh', 'expression_blendshapes'):
        data = key.encode()
        path = tmp_path / key
        path.write_bytes(data)
        paths[key] = path
        pinned[key] = replace(pinned[key], size=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(pilot, 'ASSETS', pinned)
    with FaceDatabase(config.database) as db:
        run_id = db.connection.execute('SELECT id FROM runs').fetchone()[0]
        labels = {r[0]: 'A' for r in db.connection.execute('SELECT id FROM observations')}
        result = pilot.evaluate_observations(db, run_id, {'sface': config}, labels,
            assets=lambda c: paths, backend=lambda *a: Backend())['sface']
        assert result['artifact_sizes'] == {key: len(key.encode()) for key in paths}
        assert result['artifact_bytes'] == sum(result['artifact_sizes'].values())
        assert result['unavailable_optional_artifacts'] == {}
