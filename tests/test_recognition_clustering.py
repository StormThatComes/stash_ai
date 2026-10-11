import json
import numpy as np
import pytest

from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.scanning import scan_scene
from tests.helpers.recognition import MediaClient, Backend


def setup(tmp_path):
    path = tmp_path / 'face.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0}\n')
    return load(path)


def scan(state, config, scene, vectors=None, count=3):
    client = MediaClient(scene, count)
    backend = Backend()
    if vectors is not None:
        iterator = iter(vectors)
        backend.embed = lambda image, face: next(iterator)
    with SpriteInput(client, client.scene, config) as source:
        identifier, _ = scan_scene(state, 'scope', client.scene, config, 'ns', source, lambda: backend)
    return identifier, client


def test_known_person_from_other_scene_is_reused_without_other_media(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        a = cluster_scene(state, first, config)
        assert a[0]['status'] == 'accepted'
        second, client = scan(state, config, '2')
        b = cluster_scene(state, second, config)
        assert a[0]['person'] == b[0]['person']
        assert state.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1
        assert cluster_scene(state, second, config) == b
        assert all('/2.' in url for url in client.calls)


def test_new_identity_needs_independent_times_and_corruption_is_not_reinferred(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        short, _ = scan(state, config, '1', count=1)
        result = cluster_scene(state, short, config)
        assert result[0]['status'] == 'withheld'
        assert state.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 0
        next_scan, _ = scan(state, config, '2')
        state.connection.execute('UPDATE observations SET embedding=? WHERE scan=?', (b'corrupt', next_scan))
        state.connection.commit()
        with pytest.raises(ValueError, match='embedding'):
            cluster_scene(state, next_scan, config)


def test_weak_bridge_is_not_transitive_union(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        # A-B and B-C are similar, but A-C fails cohesion. A single-link
        # connected-components implementation would wrongly merge all three.
        a, b, c = np.array([1.,0,0,0]), np.array([.8,.6,0,0]), np.array([.28,.96,0,0])
        identifier, _ = scan(state, config, '1', [a,a,a,b,c,c,c], count=7)
        rows = cluster_scene(state, identifier, config)
        accepted = [row for row in rows if row['status'] == 'accepted']
        assert len(accepted) == 2
        assert sorted(row['observations'] for row in accepted) == [3, 4]


@pytest.mark.parametrize('query', [[1.,1,0,0], [.68,.64,.358329457,0]])
def test_two_existing_similar_people_withhold_ambiguous_new_observation(tmp_path, query):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        cluster_scene(state, first, config)
        second, _ = scan(state, config, '2', [np.array([0.,1,0,0])] * 3)
        cluster_scene(state, second, config)
        third, _ = scan(state, config, '3', [np.array(query)] * 3)
        rows = cluster_scene(state, third, config)
        assert all(row['status'] == 'withheld' for row in rows)
        assert state.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 2


def test_policy_change_reuses_vectors_and_does_not_change_other_scene_members(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        cluster_scene(state, first, config)
        before = state.connection.execute('SELECT * FROM members').fetchall()
        path = tmp_path / 'other.yml'
        path.write_text('version: 1\nrecognition: {match_threshold: .8}\n')
        stricter = load(path)
        assert cluster_scene(state, first, stricter)[0]['status'] == 'accepted'
        assert [tuple(row) for row in state.connection.execute('SELECT * FROM members')] == [tuple(row) for row in before]


def test_local_pair_evaluation_reports_only_dataset_results():
    from stash_ai.recognition.evaluation import evaluate_pairs
    result = evaluate_pairs([([1.,0], [1.,0], True), ([1.,0], [0.,1], False)], .65)
    assert result['false_match'] == result['false_miss'] == 0
    assert result['samples'] == 2
    assert result['scope'] == 'supplied-local-pairs-only'
