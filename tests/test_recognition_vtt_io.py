import hashlib
import json
import uuid

import pytest

from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.portrait_availability import candidate_observations
from stash_ai.recognition.vtt_portraits import store
from tests.helpers.recognition import MediaClient
from tests.test_recognition_clustering import setup


def test_thumbnail_reuses_decoded_pixels_but_rejects_replaced_bytes(tmp_path, monkeypatch):
    import stash_ai.recognition.sprite_cache as media
    config, client = setup(tmp_path), MediaClient(count=3)
    original, opened = media.open_image, []
    def image(*args, **kwargs):
        opened.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(media, 'open_image', image)
    with SpriteInput(client, client.scene, config) as source:
        with source.thumbnail(0) as first, source.thumbnail(1) as second:
            assert first.tobytes() == second.tobytes()
        assert len(opened) == 1
        source.cache[source.cues[0].image] = b'changed bytes'
        with pytest.raises(ValueError, match='changed'):
            source.thumbnail(2)


def test_oversized_portrait_sheet_uses_bounded_crop_path(tmp_path):
    config, client = setup(tmp_path), MediaClient()
    with SpriteInput(client, client.scene, config) as source:
        source.portrait_sheet_bytes = 1
        with source.thumbnail(0) as first, source.thumbnail(1) as second:
            assert first.tobytes() == second.tobytes()
        assert source._portrait_sheet[2] is None


def test_unchanged_vtt_metadata_does_not_write_or_drop_cached_jpeg(tmp_path):
    with FaceDatabase(tmp_path / 'faces.sqlite') as database:
        db = database.connection
        metadata = {'source_type': 'recognition_crop', 'crop': [0, 0, 1, 1], 'score': .5}
        db.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                   ('p', 'scope', 'person', '1', encoded(metadata), b'jpeg', hashlib.sha256(b'jpeg').hexdigest(), 'published'))
        db.commit()
        before = db.total_changes
        store(database, 'p', 'scope', {'person': 'person'}, '1', dict(metadata))
        assert db.total_changes == before
        store(database, 'p', 'scope', {'person': 'person'}, '1', {'score': .7, 'source_type': 'vtt'})
        row = db.execute('SELECT * FROM portraits WHERE id=?', ('p',)).fetchone()
        assert row['image'] == b'jpeg' and row['status'] == 'published'
        assert json.loads(row['metadata']) == {**metadata, 'score': .7}


def test_candidate_batches_are_lazy_and_keep_exact_uuid_addressing(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as database:
        db = database.connection
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ('scan', 'scope', '1', 'ns', 'input', 'policy', '{}', '{}', 0, 0, 'complete'))
        oid, policy = str(uuid.uuid4()), 'policy'
        exact = str(uuid.uuid5(uuid.UUID(oid), policy))
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                   (exact, 'scan', policy, 'p', 'accepted', '{}'))
        statements = []
        db.set_trace_callback(statements.append)
        pairs = list(candidate_observations(db, [{'id': oid}, {'id': 'not-a-uuid'}], 'scan', policy))
        assert not statements
        assert pairs[0][1]()['person'] == 'p'
        assert sum(sql.startswith('SELECT id,person,status,payload') for sql in statements) == 1
        with pytest.raises(ValueError):
            pairs[1][1]()
