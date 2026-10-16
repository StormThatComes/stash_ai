"""Scene context may attach weak faces, never establish or bridge identities."""
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_clustering import scan, setup
from tests.test_recognition_rebuild import changed, library, partition
from tests.test_recognition_run import run


A = np.array([1., 0., 0., 0.])
B = np.array([.63, math.sqrt(1 - .63**2), 0., 0.])


def replay(db, config, scenes):
    selected = []
    for scene, vectors in scenes.items():
        identifier, _ = scan(db, config, scene, vectors, count=len(vectors))
        selected.append({'scene': scene, 'scan': identifier})
    rebuild_selected(db, selected, config)
    return selected


def candidates(db, scan, config):
    return [(row['status'], row['person'], json.loads(row['payload'])) for row in db.connection.execute(
        "SELECT c.* FROM candidates c JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
        'WHERE c.scan=? AND c.policy=? ORDER BY o.cue,o.id', (scan, matching_policy(config)))]


def test_same_scene_borderline_face_attaches_but_another_scene_does_not(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': [A]*3 + [B], '2': [B]*3})
        rows = candidates(db, selected[0]['scan'], config)
        assert {status for status, _, _ in rows} == {'accepted'}
        assert len({person for _, person, _ in rows}) == 1
        proof = rows[-1][2]
        assert proof['reason'] == 'same_scene_identity'
        assert proof['scores'][0] == pytest.approx(.63)
        assert proof['same_scene_context']['scene'] == '1'
        assert len(proof['same_scene_context']['support']) == 3
        assert proof['same_scene_context']['threshold'] == pytest.approx(.62)
        assert all(status == 'withheld' for status, _, _ in candidates(db, selected[1]['scan'], config))
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0


@pytest.mark.parametrize('clustering', [
    {'same_scene_match_bonus': 0}, {'same_scene_min_support': 4}, {'cohesion_threshold': .64},
])
def test_context_respects_disable_support_and_original_cohesion(tmp_path, clustering):
    config = setup(tmp_path, clustering)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': [A]*3 + [B]})
        assert candidates(db, selected[0]['scan'], config)[-1][0] == 'withheld'


def test_duplicate_timestamps_do_not_establish_a_local_core(tmp_path):
    config = changed(setup(tmp_path), recognition={'min_identity_support': 1})
    with FaceDatabase(config.database) as db:
        identifier, _ = scan(db, config, '1', [A]*3 + [B], count=4)
        db.connection.execute('UPDATE observations SET seconds=0 WHERE scan=? AND cue<3', (identifier,))
        db.connection.commit()
        rebuild_selected(db, [{'scene': '1', 'scan': identifier}], config)
        assert candidates(db, identifier, config)[-1][0] == 'withheld'


def test_context_does_not_override_raw_ambiguity(tmp_path):
    rival = np.array([0., 1., 0., 0.])
    ambiguous = np.array([.63, .64, math.sqrt(1 - .63**2 - .64**2), 0.])
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': [A]*3 + [rival]*3 + [ambiguous]})
        rows = candidates(db, selected[0]['scan'], config)
        assert len({person for status, person, _ in rows if status == 'accepted'}) == 2
        assert rows[-1][0] == 'withheld'


def test_two_borderline_faces_in_one_cue_keep_co_presence_protection(tmp_path):
    peer = np.array([.63, 0., math.sqrt(1 - .63**2), 0.])
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        identifier, _ = scan(db, config, '1', [A]*3 + [B, peer], count=5)
        db.connection.execute('UPDATE observations SET cue=3,seconds=3 WHERE scan=? AND cue=4', (identifier,))
        db.connection.commit()
        rebuild_selected(db, [{'scene': '1', 'scan': identifier}], config)
        rows = candidates(db, identifier, config)
        assert [status for status, _, _ in rows].count('accepted') == 3
        assert [status for status, _, _ in rows].count('withheld') == 2


def test_contextual_attachment_cannot_extend_the_core_transitively(tmp_path):
    beyond = np.array([.61, math.sqrt(1 - .61**2), 0., 0.])
    assert float(np.dot(B, beyond)) > .99
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': [A]*3 + [B, beyond]})
        rows = candidates(db, selected[0]['scan'], config)
        assert rows[-2][0] == 'accepted' and rows[-1][0] == 'withheld'


def test_early_weak_face_can_use_later_strong_support_in_its_scene(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': [A]*3, '2': [B] + [A]*3})
        rows = candidates(db, selected[1]['scan'], config)
        assert rows[0][0] == 'accepted' and rows[0][2]['reason'] == 'same_scene_identity'


def test_retuning_context_replays_cached_embeddings_and_matches_cold_run(tmp_path, monkeypatch):
    vectors = {'1': [A]*3 + [B], '2': [B]*3}
    config, client, services, models = library(tmp_path, monkeypatch, vectors)
    disabled = changed(config, clustering={'same_scene_match_bonus': 0})
    assert run(disabled, services, '1', '2') == 0
    original = partition(disabled)
    assert run(config, services, '2', '1') == 0
    attached = partition(config)
    assert len(attached[0]) == len(original[0]) + 1
    writes = list(client.writes)
    assert run(config, services, '1', '2') == 0
    assert partition(config) == attached and client.writes == writes
    assert run(disabled, services, '1', '2') == 0
    assert partition(disabled) == original and len(models) == 1
    cold, _, cold_services, _ = library(tmp_path / 'cold', monkeypatch, vectors)
    assert run(cold, cold_services, '1', '2') == 0
    assert partition(cold) == attached


@pytest.mark.parametrize('text', [
    'same_scene_match_bonus: -0.01', 'same_scene_match_bonus: 0.11', 'same_scene_match_bonus: true',
    'same_scene_match_bonus: .nan', 'same_scene_min_support: 1', 'same_scene_min_support: true',
    'same_scene_min_support: 3.5',
])
def test_invalid_context_settings_fail_before_creating_state(tmp_path, text):
    from stash_ai.recognition.config import load
    path = tmp_path / 'invalid.yml'
    path.write_text('version: 1\nclustering: {' + text + '}\n')
    with pytest.raises(ValueError, match='clustering.same_scene'):
        load(path)
    assert not (tmp_path / 'faces_state.sqlite3').exists()
