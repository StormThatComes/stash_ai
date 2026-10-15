"""Unreliable faces cannot seed identities, references or merge evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.config import defaults, load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.models import checked_detection, namespace
from stash_ai.recognition.quality import assess, rejection
from stash_ai.recognition.scanning import scan_identity, scan_scene
from stash_ai.recognition.visibility import CONTRACT_ID
from tests.helpers.recognition import Backend, FACE, MediaClient, Visibility, textured_image
from tests.test_recognition_run import run
from tests.test_recognition_vtt_portraits import setup


def settings(tmp_path, **detection):
    path = tmp_path / 'face.yml'
    path.write_text(yaml.safe_dump({'version': 1, 'detection': {'min_sharpness': 0, **detection}}))
    return load(path)


def enabled(config, **changes):
    data = plain(config.data)
    data['detection'].update(defaults()['detection'], min_sharpness=0, **changes)
    return replace(config, data=data)


@pytest.mark.parametrize('kind,reason', [
    ('pose', 'face_pose'), ('visibility', 'face_occluded'),
    ('incomplete', 'face_incomplete'), ('landmarks', 'face_landmarks_invalid'),
])
def test_problematic_sequence_cannot_create_a_fragment_or_supply_references(tmp_path, kind, reason):
    config, client = settings(tmp_path), MediaClient(count=6)
    geometry = list(FACE.row)
    if kind == 'pose':
        geometry[8] = 120  # nose displaced from the eye midpoint
    elif kind == 'incomplete':
        geometry[0] = -4
    elif kind == 'landmarks':
        geometry[8] = 159  # nose outside the detected face box
    poor_face = checked_detection(geometry)

    class Mixed(Backend):
        frame = 0
        def detect(self, image):
            self.frame += 1
            return [FACE if self.frame <= 3 else poor_face]
        def embed(self, image, face):
            self.calls += 1
            return np.array([1., 0, 0, 0] if self.frame <= 3 else [0., 1, 0, 0])

    class Covered(Visibility):
        def measure(self, *args):
            value = super().measure(*args)
            if kind == 'visibility' and self.calls > 3:
                value['visible_face_fraction'] = .5
            return value

    backend, visibility = Mixed(), Covered()
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        scan, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source,
                                  lambda: backend, visibility=visibility)
        assert not reused and backend.calls == 3 and backend.closed
        rows = db.connection.execute('SELECT * FROM observations ORDER BY cue').fetchall()
        assert len(rows) == 6 and all(r['rejected'] is None for r in rows[:3])
        assert all(r['rejected'] == reason and r['embedding'] is None and r['crop'] is None for r in rows[3:])
        assert visibility.calls == (6 if kind == 'visibility' else 3)
        groups = cluster_scene(db, scan, config)
        assert len(groups) == 1 and groups[0]['status'] == 'accepted' and groups[0]['observations'] == 3
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 3
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0
        repeated, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source,
            lambda: pytest.fail('completed scan loaded backend'), visibility=None)
        assert reused and repeated == scan


def test_disabled_filters_preserve_difficult_observations_without_segmentation(tmp_path):
    config = settings(tmp_path, max_pose_deviation=None, min_visible_face_fraction=0, require_complete_face=False)
    backend, client = Backend(), MediaClient()
    geometry = list(FACE.row)
    geometry[0], geometry[8] = -4, 159
    backend.detect = lambda image: [checked_detection(geometry)]
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        scan, _ = scan_scene(db, 'scope', client.scene, config, 'ns', source, lambda: backend)
        assert backend.calls == 3
        assert cluster_scene(db, scan, config)[0]['status'] == 'accepted'
        assert all('visibility' not in json.loads(r[0]) for r in db.connection.execute('SELECT quality FROM observations'))


def test_pose_gate_includes_boundary_and_requires_valid_landmarks():
    policy = defaults()['detection']
    with textured_image() as image:
        quality = assess(image, FACE)
    policy['max_pose_deviation'] = quality['pose_deviation']
    assert rejection(quality, policy) is None
    assert rejection(dict(quality, pose_deviation=policy['max_pose_deviation'] + .001), policy) == 'face_pose'
    assert rejection(dict(quality, landmarks_inside=False), policy) == 'face_landmarks_invalid'


def test_visibility_gate_accepts_exact_threshold(tmp_path):
    config, client, backend = settings(tmp_path), MediaClient(), Backend()
    class Boundary(Visibility):
        def measure(self, *args):
            value = super().measure(*args)
            value['visible_face_fraction'] = .8
            return value
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        scan, _ = scan_scene(db, 'scope', client.scene, config, 'ns', source,
                              lambda: backend, visibility=Boundary())
        assert backend.calls == 3 and cluster_scene(db, scan, config)[0]['status'] == 'accepted'


@pytest.mark.parametrize('field,value', [
    ('max_pose_deviation', -.1), ('max_pose_deviation', 1.1), ('max_pose_deviation', True),
    ('max_pose_deviation', float('nan')), ('min_visible_face_fraction', -.1),
    ('min_visible_face_fraction', 1.1), ('min_visible_face_fraction', True),
    ('min_visible_face_fraction', None), ('min_visible_face_fraction', float('inf')),
    ('require_complete_face', 'true'), ('require_complete_face', 1), ('require_complete_face', None),
])
def test_invalid_detection_quality_settings_fail_before_work(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        settings(tmp_path, **{field: value})
    assert not (tmp_path / 'faces_state.sqlite3').exists()


def test_visibility_contract_and_detection_policy_invalidate_scan_but_not_identity_space(tmp_path, monkeypatch):
    import stash_ai.recognition.scanning as scanning
    config = settings(tmp_path)
    initial = scan_identity(config, namespace(config))
    changed = settings(tmp_path, min_visible_face_fraction=.9)
    assert namespace(changed) == namespace(config)
    assert scan_identity(changed, namespace(changed)) != initial
    monkeypatch.setattr(scanning, 'CONTRACT_ID', 'new-visibility-contract')
    assert scan_identity(config, namespace(config)) != initial
    disabled = settings(tmp_path, min_visible_face_fraction=0)
    initial = scan_identity(disabled, namespace(disabled))
    monkeypatch.setattr(scanning, 'CONTRACT_ID', 'another-visibility-contract')
    assert scan_identity(disabled, namespace(disabled)) == initial


def test_scan_and_portrait_share_measurements_and_reruns_keep_published_identity(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch)
    config = enabled(config)
    models = []
    def factory(*args):
        model = Visibility()
        models.append(model)
        return model
    services.visibility = factory
    assert run(config, services, '1') == 0
    assert len(models) == 1 and models[0].calls == 3 and models[0].closed
    performer, writes = stash.links['1'][-1], list(stash.writes)
    assert run(config, services, '1') == 0
    assert len(models) == 1 and stash.writes == writes and len(backends) == 1
    stricter = enabled(config, min_visible_face_fraction=.9)
    assert run(stricter, services, '1') == 0
    assert len(models) == 2 and models[-1].calls == 3 and models[-1].closed
    assert stash.links['1'][-1] == performer and sum(w[0] == 'create' for w in stash.writes) == 1
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 2
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1
        assert all(json.loads(row[0])['visibility']['contract'] == CONTRACT_ID
                   for row in db.connection.execute('SELECT quality FROM observations'))


@pytest.mark.parametrize('failure', ['exception', 'invalid'])
def test_visibility_failure_preserves_checkpoint_and_retries_without_publishing(tmp_path, monkeypatch, failure):
    config, stash, services, backends = setup(tmp_path, monkeypatch)
    config = enabled(config)
    data = plain(config.data)
    data['execution']['checkpoint_cues'] = 1
    config = replace(config, data=data)
    models = []
    class Failed(Visibility):
        def measure(self, *args):
            value = super().measure(*args)
            if self.calls == 2:
                if failure == 'exception':
                    raise RuntimeError('synthetic visibility failure')
                value['visible_face_fraction'] = float('nan')
            return value
    def factory(*args):
        model = Visibility() if models else Failed()
        models.append(model)
        return model
    services.visibility = factory
    assert run(config, services, '1') == 1
    assert not stash.writes and models[0].closed and backends[0].closed
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (1, 'partial')
        assert db.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 1
    assert run(config, services, '1') == 0
    assert models[1].calls == 2 and models[1].closed and backends[1].calls == 2
    assert sum(w[0] == 'create' for w in stash.writes) == 1


def test_all_occluded_faces_are_reported_without_publishing_even_with_portraits_disabled(tmp_path, monkeypatch, capsys):
    config, stash, services, backends = setup(tmp_path, monkeypatch, enabled=False)
    config = enabled(config)
    class Covered(Visibility):
        def measure(self, *args):
            value = super().measure(*args)
            value['visible_face_fraction'] = .79
            return value
    services.visibility = Covered
    assert run(config, services, '1') == 0
    assert not stash.writes and backends[0].calls == 0
    assert 'face_occluded=3' in capsys.readouterr().out
    services.visibility = lambda *args: pytest.fail('cached rejection loaded visibility model')
    assert run(config, services, '1') == 0
    assert not stash.writes and len(backends) == 1
    assert 'face_occluded=3' in capsys.readouterr().out
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 0
        assert all(json.loads(r[0])['counts']['observations_rejected'] == 3
                   for r in db.connection.execute('SELECT summary FROM runs'))
        progress = [json.loads(r[0]) for r in db.connection.execute("SELECT data FROM progress WHERE phase='detection'")]
        assert all(p['rejected'] == {'face_occluded': 3} for p in progress)
        assert {p['reused'] for p in progress} == {False, True}


@pytest.mark.parametrize('fail', [False, True])
def test_detection_visibility_asset_is_required_independently_of_portraits(tmp_path, monkeypatch, fail):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    config = settings(tmp_path)
    data = plain(config.data)
    data['portraits'].update(enabled=False)
    data['portraits']['weights']['visibility'] = 0
    config = replace(config, data=data)
    downloaded = []
    monkeypatch.setattr(runtime, 'runtime', lambda *args: None)
    monkeypatch.setattr(models, 'verify', lambda *args: None)
    def download(repo, filename, *args, **kwargs):
        downloaded.append(filename)
        if filename == models.ASSETS['faceocc'].filename and fail:
            raise OSError('required visibility asset unavailable')
    monkeypatch.setattr(downloads, 'install_hf_asset', download)
    if fail:
        with pytest.raises(OSError, match='required visibility asset unavailable'):
            models.ensure_assets(config)
    else:
        assert set(models.ensure_assets(config)) == {'yunet', 'sface', 'faceocc'}
    assert downloaded.count(models.ASSETS['faceocc'].filename) == 1


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_unalignable_face_is_rejected_when_only_pose_filter_is_disabled(tmp_path):
    from tests.test_recognition_clustering import setup
    from tests.test_recognition_identity_merges import changed
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.aura import similarity_matrix
    from tests.helpers.recognition import Backend,MediaClient,FACE
    config = changed(setup(tmp_path),models={'recognizer':'auraface'})
    bad = checked_detection([16,16,128,128,49,80,111,80,80,80,56,80,104,80,.99])
    assert config.data['detection']['max_pose_deviation'] is None
    class AlignmentBackend(Backend):
        def detect(self,image):return [FACE,bad]
        def embed(self,image,face):
            similarity_matrix(face.landmarks)
            return super().embed(image,face)
    client,model = MediaClient('1'),AlignmentBackend()
    with FaceDatabase(config.database) as db,SpriteInput(client,client.scene,config) as source:
        scan_scene(db,'scope',client.scene,config,'ns',source,lambda:model)
        assert db.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (3,'complete')
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0] == 3
        assert db.connection.execute("SELECT COUNT(*) FROM observations WHERE rejected='face_landmarks_invalid' AND embedding IS NULL").fetchone()[0] == 3
        assert model.calls == 3



def test_legacy_complete_scan_rechecks_alignment_without_loading_models(tmp_path):
    from stash_ai.recognition.rebuild import rebuild_selected
    from tests.test_recognition_clustering import setup as simple_setup, scan
    config = simple_setup(tmp_path)
    with FaceDatabase(config.database) as db:
        identifier, _ = scan(db, config, '1')
        selection = [{'scene': '1', 'scan': identifier}]
        assert rebuild_selected(db, selection, config)[identifier][0]['status'] == 'accepted'
        rows = db.connection.execute('SELECT id,quality FROM observations WHERE scan=? ORDER BY cue', (identifier,)).fetchall()
        for row in rows:
            quality = json.loads(row['quality'])
            quality.pop('alignment_valid')
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        bad = [16,16,128,128,49,80,111,80,80,80,56,80,104,80,.99]
        db.connection.execute('UPDATE observations SET geometry=? WHERE id=?', (json.dumps(bad), rows[0]['id']))
        db.connection.commit()
        client = MediaClient('1')
        with SpriteInput(client, client.scene, config) as source:
            assert scan_scene(db, 'scope', client.scene, config, 'ns', source,
                              lambda: pytest.fail('legacy scan loaded models')) == (identifier, True)
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0] == 2
        rejected = db.connection.execute('SELECT rejected,embedding,crop FROM observations WHERE id=?', (rows[0]['id'],)).fetchone()
        assert rejected[:] == ('face_landmarks_invalid', None, None)
        replayed = rebuild_selected(db, selection, config)[identifier]
        assert sum(row['observations'] for row in replayed if row['status'] == 'accepted') == 0
        assert sum(row['observations'] for row in replayed) == 2
