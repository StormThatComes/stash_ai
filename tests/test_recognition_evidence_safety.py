"""Accepted crop budgets and uncertain co-presence preserve reliable evidence."""
from dataclasses import replace
from io import BytesIO
import json
import uuid

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.quality import crop_bytes
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.scanning import scan_scene
from stash_ai.recognition.visibility import LazyVisibility, VisibilityError, prepare, measurement
from tests.helpers.recognition import FACE, Backend, MediaClient, textured_image
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_identity_merges import changed, fragments


@pytest.mark.parametrize('limit', [1024, 1048576])
@pytest.mark.parametrize('portraits', [False, True])
def test_valid_crop_budgets_complete_scans_with_honest_encoding_metadata(tmp_path, limit, portraits):
    path = tmp_path / 'faces.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0, min_visible_face_fraction: 0}\n'
                    f'storage: {{max_crop_bytes: {limit}}}\nportraits: {{enabled: {str(portraits).lower()}}}\n')
    config, client, backend = load(path), MediaClient(), Backend()
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        identifier, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source, lambda: backend)
        assert not reused and backend.calls == 3
        assert db.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (3, 'complete')
        for row in db.connection.execute('SELECT * FROM observations'):
            quality = json.loads(row['quality'])
            assert row['rejected'] is None and row['embedding'] and len(row['crop']) <= limit
            with Image.open(BytesIO(row['crop'])) as crop:
                assert list(crop.size) == quality['crop_encoding']['pixels']
                assert crop.width <= quality['crop_bounds'][2] - quality['crop_bounds'][0]
            assert quality['crop_encoding']['jpeg_quality'] <= 90
            if limit == 1024:
                assert quality['crop_encoding']['jpeg_quality'] < 90
        again, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source,
                                   lambda: pytest.fail('complete scan must reuse embeddings'))
        assert reused and again == identifier


def test_normal_crop_budget_preserves_original_jpeg_bytes_and_borrowed_pixels():
    with textured_image() as image:
        details = {}
        data, bounds = crop_bytes(image, FACE, details=details)
        with image.crop(bounds) as source, source.convert('RGB') as crop:
            output = BytesIO()
            crop.save(output, format='JPEG', quality=90)
        assert data == output.getvalue() and not details['downscaled']
        assert image.getpixel((0, 0))


def test_unmeasurable_partial_face_is_rejected_without_losing_other_valid_cues(tmp_path):
    path = tmp_path / 'faces.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0, require_complete_face: false}\n'
                    'execution: {checkpoint_cues: 1}\n')
    config, client, backend = load(path), MediaClient(count=4), Backend()
    row = list(FACE.row)
    # Keep enough real source pixels for embedding alignment, but let the
    # FaceOcc visibility ellipse still cross the left image boundary.
    for index in (0, 4, 6, 8, 10, 12):
        row[index] -= 40
    detections = iter([FACE, replace(FACE, row=tuple(row)), FACE, FACE])
    backend.detect = lambda image: [next(detections)]
    class Visibility:
        def __init__(self, *args): pass
        def measure(self, image, face):
            _, region = prepare(image, face)
            return measurement(np.ones((1, 1, 256, 256)), region, 'cpu')
        def close(self): pass
    visibility = LazyVisibility(config, {}, Visibility)
    try:
        with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
            identifier, _ = scan_scene(db, 'scope', client.scene, config, 'ns', source,
                                        lambda: backend, visibility=visibility)
            assert db.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (4, 'complete')
            rows = db.connection.execute('SELECT * FROM observations ORDER BY cue').fetchall()
            assert backend.calls == 3 and len(rows) == 4
            assert rows[1]['rejected'] == 'face_visibility_unmeasurable'
            assert json.loads(rows[1]['quality'])['alignment_valid'] is True
            assert rows[1]['embedding'] is rows[1]['crop'] is None
            assert not json.loads(rows[1]['quality'])['complete']
            assert all(rows[i]['rejected'] is None for i in (0, 2, 3))
            assert cluster_scene(db, identifier, config)[0]['observations'] == 3
    finally:
        visibility.close()


def test_generic_visibility_failure_still_aborts_with_a_durable_checkpoint(tmp_path):
    config, client, backend = load(root=tmp_path), MediaClient(), Backend()
    class Broken:
        calls = 0
        def __init__(self, *args): pass
        def measure(self, image, face):
            self.calls += 1
            if self.calls == 2:
                raise ValueError('invalid model output')
            _, region = prepare(image, face)
            return measurement(np.ones((1, 1, 256, 256)), region, 'cpu')
        def close(self): pass
    visibility = LazyVisibility(config, {}, Broken)
    try:
        with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
            with pytest.raises(VisibilityError, match='invalid model output'):
                scan_scene(db, 'scope', client.scene, config, 'ns', source, lambda: backend, visibility=visibility)
            assert db.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (1, 'partial')
            assert db.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 1
    finally:
        visibility.close()


def pair_cues(db, identifier):
    for row in db.connection.execute('SELECT id,cue FROM observations WHERE scan=?', (identifier,)).fetchall():
        db.connection.execute('UPDATE observations SET cue=?,seconds=? WHERE id=?',
                              (row['cue'] // 2, row['cue'] // 2, row['id']))
    db.connection.commit()


@pytest.mark.parametrize('margin', [.1, .25])
@pytest.mark.parametrize('similarity,merges', [(.8, 0), (.95, 1)])
def test_withheld_co_present_faces_keep_the_stronger_merge_guard(tmp_path, margin, similarity, merges):
    config = changed(setup(tmp_path), recognition={'ambiguity_margin': margin})
    a, b = np.array([1., 0, 0, 0]), np.array([similarity, np.sqrt(1-similarity**2), 0, 0])
    with FaceDatabase(config.database) as db:
        fragments(db, config, [a, b])
        selected, _ = scan(db, config, '10', [a, b] * 3, count=6)
        pair_cues(db, selected)
        cluster_scene(db, selected, config)
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == merges
        if merges:
            proof = json.loads(db.connection.execute('SELECT evidence FROM identity_merges').fetchone()[0])
            assert proof['cooccurring'] and proof['scores']['minimum_cross_similarity'] >= .9


def test_duplicate_ambiguous_detections_do_not_invent_co_presence(tmp_path):
    config = changed(setup(tmp_path), recognition={'ambiguity_margin': .25})
    a, b = np.array([1., 0, 0, 0]), np.array([.8, .6, 0, 0])
    bridge = (a + b) / np.linalg.norm(a + b)
    with FaceDatabase(config.database) as db:
        fragments(db, config, [a, b])
        selected, _ = scan(db, config, '10', [bridge] * 6, count=6)
        pair_cues(db, selected)
        assert cluster_scene(db, selected, config)[0]['status'] == 'accepted'
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 1
        proof = json.loads(db.connection.execute('SELECT evidence FROM identity_merges').fetchone()[0])
        assert not proof['cooccurring'] and len(proof['support_cues']) == 3


@pytest.mark.parametrize('superseded', [False, True])
def test_cross_scene_uncertain_context_uses_only_the_latest_scan(tmp_path, superseded):
    config = changed(setup(tmp_path), recognition={'ambiguity_margin': .25})
    a, b = np.array([1., 0, 0, 0]), np.array([.8, .6, 0, 0])
    bridge = (a + b) / np.linalg.norm(a + b)
    with FaceDatabase(config.database) as db:
        fragments(db, config, [a, b])
        paired, _ = scan(db, config, '10', [a, b], count=2)
        pair_cues(db, paired)
        assert all(row['status'] == 'withheld' for row in cluster_scene(db, paired, config))
        if superseded:
            replacement, _ = scan(db, config, '10', [bridge], count=1)
            cluster_scene(db, replacement, config)
        selected, _ = scan(db, config, '11', [bridge] * 3)
        cluster_scene(db, selected, config)
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == int(superseded)


@pytest.mark.parametrize('margin', [.1, .25])
def test_cold_replay_preserves_uncertain_co_presence_under_reference_drift(tmp_path, margin):
    config = changed(setup(tmp_path), recognition={'ambiguity_margin': margin},
                     clustering={'cohesion_threshold': .35, 'max_samples_per_person': 2})
    a = np.array([1., 0, 0, 0])
    b = np.array([.39, np.sqrt(1-.39**2), 0, 0])
    drift, paired = np.array([np.sqrt(.9), np.sqrt(.1), 0, 0]), np.array([.6, .8, 0, 0])
    with FaceDatabase(config.database) as db:
        selected = []
        for scene, vectors in [('1', [a]*3), ('2', [b]*3), ('3', [drift]*2), ('4', [drift, paired]*3)]:
            identifier, _ = scan(db, config, scene, vectors, count=len(vectors))
            selected.append({'scene': scene, 'scan': identifier})
        # Stable, controlled ordering of the bounded representative sampler.
        for item, base in zip(selected, [20000, 10000, 1, 1000]):
            for offset, row in enumerate(db.connection.execute('SELECT id FROM observations WHERE scan=? ORDER BY cue',
                                                               (item['scan'],)).fetchall()):
                db.connection.execute('UPDATE observations SET id=? WHERE id=?',
                                      (str(uuid.UUID(int=base+offset)), row['id']))
        pair_cues(db, selected[-1]['scan'])
        result = rebuild_selected(db, selected, config)
        assert len({r['person'] for rows in result.values() for r in rows if r['status'] == 'accepted'}) == 2


def test_downscaled_retained_crop_maps_landmarks_to_encoded_pixels(tmp_path):
    from stash_ai.recognition.portrait_availability import retained_image
    config = changed(setup(tmp_path), storage={'max_crop_bytes': 1024})
    client = MediaClient()
    with Image.fromarray(np.random.default_rng(4).integers(0, 255, (160, 160, 3), dtype='uint8')) as image:
        output = BytesIO()
        image.save(output, format='PNG')
        client.data['http://stash/sprite/1.jpg'] = output.getvalue()
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        identifier, _ = scan_scene(db, 'scope', client.scene, config, 'ns', source, Backend)
        row = dict(db.connection.execute('SELECT * FROM observations WHERE scan=? LIMIT 1', (identifier,)).fetchone())
        quality = json.loads(row['quality'])
        assert quality['crop_encoding']['downscaled']
        image, face = retained_image(row)
        with image:
            bounds = quality['crop_bounds']
            sx, sy = image.width / (bounds[2] - bounds[0]), image.height / (bounds[3] - bounds[1])
            assert face.box == pytest.approx(((FACE.box[0]-bounds[0])*sx, (FACE.box[1]-bounds[1])*sy,
                                             FACE.box[2]*sx, FACE.box[3]*sy))
            assert face.landmarks[0] == pytest.approx(((FACE.landmarks[0][0]-bounds[0])*sx,
                                                     (FACE.landmarks[0][1]-bounds[1])*sy))
        quality['crop_encoding']['pixels'][0] += 1
        row['quality'] = json.dumps(quality)
        with pytest.raises(ValueError, match='dimensions'):
            retained_image(row)


def test_small_crop_budget_still_supplies_a_portrait_when_thumbnail_read_fails(tmp_path, monkeypatch):
    import yaml
    from stash_ai.recognition.run import execute
    from types import SimpleNamespace
    from tests.test_recognition_vtt_portraits import setup as portraits
    config, stash, services, _ = portraits(tmp_path, monkeypatch, enabled=False)
    data = yaml.safe_load(config.source_path.read_text())
    data['storage'] = {'max_crop_bytes': 1024}
    config.source_path.write_text(yaml.safe_dump(data))
    config = load(config.source_path)
    args = SimpleNamespace(scene_id=['1'], all_scenes=False, progress_style='off')
    assert execute(args, config, services=services) == 0
    data['portraits']['enabled'] = True
    config.source_path.write_text(yaml.safe_dump(data))
    config = load(config.source_path)
    factory, calls = services.source, []
    def source(*args):
        calls.append(1)
        if len(calls) > 1:
            raise OSError('thumbnail unavailable after scan verification')
        return factory(*args)
    services.source = source
    services.backend = lambda *args: pytest.fail('fallback must reuse accepted embeddings')
    assert execute(args, config, services=services) == 0
    assert len(stash.images) == 1 and len(next(iter(stash.images.values()))) <= 1024
    with FaceDatabase(config.database) as db:
        row = db.connection.execute("SELECT metadata FROM portraits WHERE status='published'").fetchone()
        assert json.loads(row[0])['source_type'] == 'recognition_crop'


import json
import os
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('known', [False, True])
def test_co_present_assignment_respects_stronger_guard(tmp_path, known):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.clustering import cluster_scene
    from tests.test_recognition_clustering import setup, scan
    from tests.test_recognition_evidence_safety import pair_cues
    config = setup(tmp_path)
    a, b = np.array([1.,0,0,0]), np.array([.8,.6,0,0])
    assert config.data['recognition']['cooccurrence_merge_threshold'] == .9
    with FaceDatabase(config.database) as db:
        if known:
            previous, _ = scan(db, config, '1', [a]*3)
            cluster_scene(db, previous, config)
        selected, _ = scan(db, config, '2', [a,b]*3, count=6)
        pair_cues(db, selected)
        # Make the two detections geometrically distinct within each cue.
        for row in db.connection.execute('SELECT id,embedding,geometry FROM observations WHERE scan=?',(selected,)).fetchall():
            geometry = json.loads(row['geometry'])
            # Two small nonoverlapping valid faces inside the same 160px cue.
            shifted = np.frombuffer(row['embedding'], dtype='<f4')[1] > .1
            for index in (0,2,4,6,8,10,12):
                geometry[index] = geometry[index] / 3 + (80 if shifted and index != 2 else 0)
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?',(json.dumps(geometry),row['id']))
        db.connection.commit()
        result = cluster_scene(db, selected, config)
        assert not any(row['status'] == 'accepted' and row['observations'] == 6 for row in result)
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members WHERE observation IN (SELECT id FROM observations WHERE scan=?)',(selected,)).fetchone()[0] != 1
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0
        from stash_ai.recognition.rebuild import rebuild_selected
        replayed = rebuild_selected(db, [{'scene':'2', 'scan': selected}], config)[selected]
        assert not any(row['status'] == 'accepted' and row['observations'] == 6 for row in replayed)



@pytest.mark.parametrize('similarity', [.95, 1.])
def test_ordinary_assignment_allows_strong_reflections_and_exact_duplicates(tmp_path, similarity):
    config = setup(tmp_path)
    a = np.array([1., 0, 0, 0])
    b = np.array([similarity, np.sqrt(1 - similarity ** 2), 0, 0])
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '2', [a, b] * 3, count=6)
        pair_cues(db, selected)
        result = cluster_scene(db, selected, config)
        assert len(result) == 1 and result[0]['status'] == 'accepted'
        assert result[0]['observations'] == 6
        replayed = rebuild_selected(db, [{'scene': '2', 'scan': selected}], config)[selected]
        assert len(replayed) == 1 and replayed[0]['observations'] == 6



def test_incohesive_rival_cannot_hide_same_cue_competition(tmp_path):
    from stash_ai.recognition.clustering import Group, co_presence_allows, vector
    config = changed(setup(tmp_path), recognition={'ambiguity_margin': .01})
    a, b = np.array([1., 0, 0, 0]), np.array([.6, .8, 0, 0])
    group = Group('target', [a, b], True)
    rival = Group('rival', [a, a, a, np.array([.4, np.sqrt(.84), 0, 0])], True)
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '2', [a, a] * 3, count=6)
        pair_cues(db, selected)
        rows = db.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue,id', (selected,)).fetchall()
        geometry = json.loads(rows[1]['geometry'])
        geometry[0] += 1
        db.connection.execute('UPDATE observations SET geometry=? WHERE id=?', (json.dumps(geometry), rows[1]['id']))
        assert not co_presence_allows(db.connection, rows[0], group, [group, rival], vector(rows[0]), config)
