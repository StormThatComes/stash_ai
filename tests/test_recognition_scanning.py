import json

import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput, bounded_media
from stash_ai.recognition.scanning import scan_scene
from tests.helpers.recognition import MediaClient, Backend


def config(tmp_path):
    p = tmp_path / 'face.yml'
    p.write_text('version: 1\ndetection: {min_sharpness: 0}\nexecution: {checkpoint_cues: 1}\n')
    return load(p)


def test_selected_scene_results_and_reuse_in_one_sqlite(tmp_path):
    settings, client = config(tmp_path), MediaClient('6306')
    backend = Backend()
    with FaceDatabase(settings.database) as state, SpriteInput(client, client.scene, settings) as source:
        scan, reused = scan_scene(state, 'scope', client.scene, settings, 'namespace', source, lambda: backend)
        assert not reused and backend.closed
        rows = state.connection.execute('SELECT * FROM observations').fetchall()
        assert len(rows) == 3 and all(row['embedding'] and row['crop'] for row in rows)
        assert all('visibility_method' in json.loads(row['quality']) for row in rows)
        assert state.connection.execute('SELECT status FROM scans').fetchone()[0] == 'complete'
        scan2, reused = scan_scene(state, 'scope', client.scene, settings, 'namespace', source,
                                   lambda: pytest.fail('cached evidence loaded models'))
        assert scan2 == scan and reused
    assert all('/6306.' in url for url in client.calls)


def test_failed_inference_resumes_only_completed_cues(tmp_path):
    settings, client = config(tmp_path), MediaClient()
    failing = Backend(fail_at=2)
    with FaceDatabase(settings.database) as state, SpriteInput(client, client.scene, settings) as source:
        with pytest.raises(RuntimeError, match='synthetic'):
            scan_scene(state, 'scope', client.scene, settings, 'ns', source, lambda: failing)
        row = state.connection.execute('SELECT * FROM scans').fetchone()
        assert row['cursor'] == 1 and row['status'] == 'partial'
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 1
        next_backend = Backend()
        scan_scene(state, 'scope', client.scene, settings, 'ns', source, lambda: next_backend)
        assert next_backend.calls == 2
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 3


def test_no_face_is_complete_scan_not_inferred_presence(tmp_path):
    settings, client = config(tmp_path), MediaClient()
    backend = Backend()
    backend.detect = lambda image: []
    with FaceDatabase(settings.database) as state, SpriteInput(client, client.scene, settings) as source:
        scan_scene(state, 'scope', client.scene, settings, 'ns', source, lambda: backend)
        assert state.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (3, 'complete')
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0


def test_retained_image_byte_budget_and_cross_origin_are_enforced(tmp_path):
    client = MediaClient()
    with pytest.raises(ValueError, match='origin'):
        bounded_media(client, 'http://elsewhere/image', 1024, 5)
    assert not client.calls
    with pytest.raises(ValueError, match='budget'):
        bounded_media(client, client.url + 'sprite/1.jpg', 1, 5)
    settings = config(tmp_path)
    with SpriteInput(client, client.scene, settings) as source:
        source.hashes[source.cues[0].image] = 'changed'
        with pytest.raises(ValueError, match='changed'):
            list(source.frames())
