"""Retained evidence survives revalidation of derived identity decisions."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.names import sync_name
from stash_ai.recognition.portraits import choose
from stash_ai.recognition.publication import PendingRecognition, save_operation
from tests.helpers.recognition import Backend, MediaClient, Stash, textured_image
from tests.test_recognition_clustering import scan

QUERY = np.array([.68, .64, math.sqrt(1 - .68 ** 2 - .64 ** 2), 0], dtype='float32')


def config_at(root, source='vtt'):
    path = root / 'facial_recognition.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0}\n'
                    f'portraits: {{source: {source}, min_sharpness: 0, margin: .1, max_candidates: 2, time_offsets: [0], '
                    'min_visible_face_fraction: 0, weights: {visibility: 0}}\n')
    return load(path)


def test_candidate_cache_reassesses_late_rival_without_changing_evidence(tmp_path):
    config = config_at(tmp_path)
    with FaceDatabase(config.database) as db:
        first, _ = scan(db, config, '1', count=32)
        person = cluster_scene(db, first, config)[0]['person']
        selected, _ = scan(db, config, '2', [QUERY], count=1)
        assert cluster_scene(db, selected, config)[0]['person'] == person
        rival, _ = scan(db, config, '3', [np.array([0., 1, 0, 0])] * 3)
        cluster_scene(db, rival, config)
        before = {table: [tuple(row) for row in db.connection.execute('SELECT * FROM ' + table)]
                  for table in ('observations', 'members')}
        assert cluster_scene(db, selected, config)[0]['status'] == 'withheld'
        payload = json.loads(db.connection.execute('SELECT payload FROM candidates WHERE scan=?', (selected,)).fetchone()[0])
        assert payload['reason'] == 'ambiguous_identity' and payload['reference_context']
        assert before == {table: [tuple(row) for row in db.connection.execute('SELECT * FROM ' + table)] for table in before}
        statements = []
        db.connection.set_trace_callback(statements.append)
        assert cluster_scene(db, selected, config)[0]['status'] == 'withheld'
        assert not any(sql.startswith(('INSERT', 'UPDATE', 'DELETE')) for sql in statements)


def test_withheld_candidate_can_match_later_known_identity_with_stable_cache(tmp_path):
    config = config_at(tmp_path)
    with FaceDatabase(config.database) as db:
        short, _ = scan(db, config, '1', count=1)
        assert cluster_scene(db, short, config)[0]['status'] == 'withheld'
        known, _ = scan(db, config, '2')
        person = cluster_scene(db, known, config)[0]['person']
        assert cluster_scene(db, short, config)[0]['person'] == person
        statements = []
        db.connection.set_trace_callback(statements.append)
        assert cluster_scene(db, short, config)[0]['status'] == 'accepted'
        assert not any(sql.startswith(('INSERT', 'UPDATE', 'DELETE')) for sql in statements)


class Decoder:
    url = 'http://stash/scene/2/stream'

    def __init__(self, *args): pass
    def __enter__(self): return self
    def frame(self, seconds):
        return textured_image(), {'requested_seconds': seconds, 'actual_seconds': seconds}
    def close(self): pass


def no_inference(*args, **kwargs):
    pytest.fail('cached eligibility must not initialize inference or read new media')


@pytest.mark.parametrize('source', ['vtt', 'video'])
def test_cached_portrait_is_withheld_after_rival_without_inference_or_media(tmp_path, source):
    config = config_at(tmp_path, source)
    with FaceDatabase(config.database) as db:
        first, _ = scan(db, config, '1', count=32)
        person = cluster_scene(db, first, config)[0]['person']
        selected, _ = scan(db, config, '2', [QUERY], count=1)
        assert cluster_scene(db, selected, config)[0]['person'] == person
        media = MediaClient('2', count=1)
        client = SimpleNamespace(url=media.url, media=media.media, scene=lambda identifier: media.scene)
        target = {'person': person, 'performer': '101', 'scans': [{'scene': '2', 'scan': selected}]}
        winner = choose(db, client, config, 'scope', target, {}, lambda *args: Backend(vector=QUERY), Decoder)
        assert winner is not None
        # Stable references also reuse the stored image without loading a model.
        assert choose(db, client, config, 'scope', target, {}, no_inference, no_inference)['id'] == winner['id']
        rival, _ = scan(db, config, '3', [np.array([0., 1, 0, 0])] * 3)
        cluster_scene(db, rival, config)
        client.scene = no_inference
        client.media.get = no_inference
        assert choose(db, client, config, 'scope', target, {}, no_inference, no_inference) is None
        current = db.connection.execute('SELECT * FROM portraits WHERE id=?', (winner['id'],)).fetchone()
        assert current['image'] == winner['image'] and current['digest'] == winner['digest']
        assert json.loads(current['metadata'])['reason'] == 'portrait_identity_ambiguous'
        assert json.loads(current['metadata'])['reference_context']


def test_legacy_video_cache_without_embedding_keeps_bytes_but_is_not_published(tmp_path):
    config = config_at(tmp_path, 'video')
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '1')
        person = cluster_scene(db, selected, config)[0]['person']
        target = {'person': person, 'performer': '101', 'scans': [{'scene': '1', 'scan': selected}]}
        winner = choose(db, Stash(), config, 'scope', target, {}, lambda *args: Backend(), Decoder)
        metadata = json.loads(winner['metadata'])
        metadata.pop('identity_embedding')
        with db.connection:
            db.connection.execute('UPDATE portraits SET metadata=?', (json.dumps(metadata),))
        fallback = choose(db, Stash(), config, 'scope', target, {}, no_inference, no_inference)
        assert fallback is not None and fallback['id'] != winner['id']
        assert json.loads(fallback['metadata'])['source_type'] == 'vtt'
        assert db.connection.execute('SELECT image FROM portraits WHERE id=?', (winner['id'],)).fetchone()[0] == winner['image']


@pytest.mark.parametrize('live,desired,observed', [('Second', 'Second', False), ('First', 'First', True), ('Third', 'First', False)])
def test_unknown_name_reconciles_only_against_original_payload(tmp_path, live, desired, observed):
    config = config_at(tmp_path)
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '1')
        person = cluster_scene(db, selected, config)[0]['person']
        token = db.connection.execute('SELECT token FROM people WHERE id=?', (person,)).fetchone()[0]
        client = Stash()
        client.profiles['101'] = {'id': '101', 'name': live, 'details': token, 'image_path': ''}
        identifier = save_operation(db, scope='scope', scene='', person=person, kind='name', target='101',
            payload={'name': 'First'}, precondition={'name': 'Original', 'token': token}, status='unknown')
        target, plan = {'person': person, 'performer': '101'}, {'101': {'before': live, 'after': desired, 'token': token}}
        if observed:
            assert sync_name(db, client, None, 'scope', target, plan) == {'names_unchanged': 1}
        else:
            with pytest.raises(PendingRecognition, match='original intent'):
                sync_name(db, client, None, 'scope', target, plan)
        row = db.connection.execute('SELECT * FROM publications WHERE id=?', (identifier,)).fetchone()
        assert row['status'] == ('observed' if observed else 'unknown')
        assert json.loads(row['payload']) == {'name': 'First'} and not client.writes
