"""Retuning must reproduce cold partitions while keeping owned profile labels."""
from dataclasses import replace
import json
import math
import sqlite3

import numpy as np
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase, APPLICATION_ID, SCHEMA_VERSION, SCHEMA_V3
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.publication import save_operation
from stash_ai.recognition.run import instance_scope
from stash_ai.recognition.stash import RecognitionStash
from tests.helpers.recognition import Backend, MediaClient
from tests.test_recognition_run import setup, run


def changed(config, **sections):
    data = plain(config.data)
    for name, fields in sections.items():
        data[name].update(fields)
    return replace(config, data=data)


def library(tmp_path, monkeypatch, vectors):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config, client, services, _ = setup(tmp_path, monkeypatch)
    loaded = []
    original_get = client.get
    def get(url, **kwargs):
        if '/vtt/' in url or '/sprite/' in url:
            scene = url.rsplit('/', 1)[-1].split('.')[0]
            return MediaClient(scene, count=len(vectors[scene])).get(url, **kwargs)
        return original_get(url, **kwargs)
    client.media.get = get
    class Source(SpriteInput):
        def __init__(self, client, scene, config):
            self.scene_id = scene['id']
            super().__init__(client, scene, config)
        def frames(self, start=0):
            for index, cue, image in super().frames(start):
                image.info['test_vector'] = vectors[self.scene_id][index]
                yield index, cue, image
    def backend(*args):
        model = Backend()
        def embed(image, face):
            model.calls += 1
            return np.asarray(image.info['test_vector'], dtype='float32')
        model.embed = embed
        loaded.append(model)
        return model
    services.source, services.backend = Source, backend
    return config, client, services, loaded


def partition(config):
    with FaceDatabase(config.database) as db:
        grouped = {}
        for row in db.connection.execute('SELECT m.person,s.scene,o.seconds,o.geometry FROM members m '
            'JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan'):
            grouped.setdefault(row['person'], []).append((row['scene'], row['seconds'], row['geometry']))
    return sorted(sorted(points) for points in grouped.values())


def two_people():
    return {'1': [[1., 0, 0, 0]] * 3, '2': [[.8, .6, 0, 0]] * 3}


def test_tight_loose_tight_matches_fresh_db_and_reactivates_same_profiles(tmp_path, monkeypatch):
    config, client, services, loaded = library(tmp_path, monkeypatch, two_people())
    strict = changed(config, recognition={'match_threshold': .95})
    assert run(strict, services, '2', '1') == 0
    initial = partition(strict)
    profiles = {scene: client.links[scene][-1] for scene in ('1', '2')}
    assert len(initial) == 2 and profiles['1'] != profiles['2']
    client.profiles[profiles['2']]['name'] = 'Preserved manual name'
    assert run(config, services, '1', '2') == 0
    assert len(partition(config)) == 1
    assert client.links['2'] == [profiles['1']]
    assert client.profiles[profiles['2']]['name'] == 'Preserved manual name'
    assert run(strict, services, '1', '2') == 0
    assert partition(strict) == initial
    assert {scene: client.links[scene][-1] for scene in profiles} == profiles
    assert len(loaded) == 1 and sum(w[0] == 'create' for w in client.writes) == 2
    writes = list(client.writes)
    assert run(strict, services, '2', '1') == 0 and client.writes == writes
    cold, cold_client, cold_services, _ = library(tmp_path / 'cold', monkeypatch, two_people())
    cold = changed(cold, recognition={'match_threshold': .95})
    assert run(cold, cold_services, '1', '2') == 0
    assert partition(strict) == partition(cold)
    assert all(w[0] != 'merge' for w in client.writes)


def test_loose_tight_loose_is_history_independent_and_oldest_survives(tmp_path, monkeypatch):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1', '2') == 0
    initial = partition(config)
    oldest = client.links['1'][-1]
    strict = changed(config, recognition={'match_threshold': .95})
    assert run(strict, services, '1', '2') == 0 and len(partition(strict)) == 2
    assert run(config, services, '2', '1') == 0
    assert partition(config) == initial and client.links['2'] == [oldest]
    assert client.links['1'] == ['77', oldest]


def merge_evidence():
    face = lambda x: [x, math.sqrt(1 - x*x), 0., 0.]
    return {'1': [face(1.)] * 3, '2': [face(.4)] + [face(.75)] * 4, '3': [face(.9)] * 3}


@pytest.mark.parametrize('changes', [
    {'recognition': {'min_merge_support': 1}, 'clustering': {'cohesion_threshold': .35}},
    {'recognition': {'ambiguity_margin': .01}},
    {'clustering': {'cohesion_threshold': .35}},
    {'clustering': {'max_samples_per_person': 2}},
])
def test_policy_changes_match_independent_cold_run(tmp_path, monkeypatch, changes):
    config, client, services, loaded = library(tmp_path, monkeypatch, merge_evidence())
    assert run(config, services, '3', '2', '1') == 0
    tuned = changed(config, **changes)
    assert run(tuned, services, '2', '1', '3') == 0
    actual = partition(tuned)
    cold, _, cold_services, _ = library(tmp_path / 'cold', monkeypatch, merge_evidence())
    cold = changed(cold, **changes)
    assert run(cold, cold_services, '1', '2', '3') == 0
    assert actual == partition(cold) and len(loaded) == 1


def test_merge_support_is_recomputed_in_both_directions(tmp_path, monkeypatch):
    config, client, services, _ = library(tmp_path, monkeypatch, merge_evidence())
    high = changed(config, recognition={'min_merge_support': 100}, clustering={'cohesion_threshold': .35})
    low = changed(high, recognition={'min_merge_support': 1})
    assert run(high, services, '1', '2', '3') == 0
    before = partition(high)
    assert len(before) == 2
    assert run(low, services, '1', '2', '3') == 0 and len(partition(low)) == 1
    assert run(high, services, '1', '2', '3') == 0 and partition(high) == before


@pytest.mark.parametrize('origins', [True, False])
def test_legacy_destructive_merge_can_be_split_without_reusing_inherited_token(tmp_path, monkeypatch, origins):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    strict = changed(config, recognition={'match_threshold': .95})
    assert run(strict, services, '1', '2') == 0
    with FaceDatabase(config.database) as db:
        oldest, source = db.connection.execute('SELECT * FROM people ORDER BY created,id').fetchall()
        db.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                              (source['id'], oldest['id'], 1., '{}', 'confirmed'))
        db.connection.execute('UPDATE members SET person=? WHERE person=?', (oldest['id'], source['id']))
        if not origins:
            db.connection.execute('DELETE FROM identity_origins')
        db.connection.commit()
    details = client.profiles[oldest['performer']]['details'] + '\n' + client.profiles[source['performer']]['details']
    client.merge_person(source['performer'], oldest['performer'], details)
    assert run(strict, services, '1', '2') == 0
    assert len(partition(strict)) == 2
    assert client.links['1'][-1] != client.links['2'][-1]
    with FaceDatabase(config.database) as db:
        active = db.connection.execute('SELECT DISTINCT p.id,p.performer FROM people p JOIN members m ON m.person=p.id').fetchall()
        assert len(active) == 2 and len({r['performer'] for r in active}) == 2
        assert source['id'] not in {r['id'] for r in active}


def test_raise_support_retires_assignments_and_lowering_restores_without_inference(tmp_path, monkeypatch):
    config, client, services, loaded = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1', '2') == 0
    initial, performer = partition(config), client.links['1'][-1]
    strict = changed(config, recognition={'min_identity_support': 4})
    assert run(strict, services, '1', '2') == 0
    assert partition(strict) == [] and client.links == {'1': ['77'], '2': []}
    assert performer in client.profiles
    assert run(config, services, '1', '2') == 0
    assert partition(config) == initial and client.links['2'] == [performer]
    assert len(loaded) == 1 and sum(w[0] == 'create' for w in client.writes) == 1


def test_smaller_rebuild_can_fit_new_budget_even_when_history_did_not(tmp_path, monkeypatch):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    strict = changed(config, recognition={'match_threshold': .95})
    assert run(strict, services, '1', '2') == 0 and len(partition(strict)) == 2
    smaller = changed(config, clustering={'max_people_per_lookup': 1})
    assert run(smaller, services, '1', '2') == 0 and len(partition(smaller)) == 1


def test_detection_change_removes_all_superseded_selected_members(tmp_path, monkeypatch):
    config, client, services, loaded = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1', '2') == 0
    performer = client.links['1'][-1]
    strict = changed(config, detection={'min_face_pixels': 200})
    assert run(strict, services, '1', '2') == 0
    assert partition(strict) == [] and client.links == {'1': ['77'], '2': []}
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0] == 6
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NOT NULL').fetchone()[0] == 6
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 0
    assert run(config, services, '1', '2') == 0 and client.links['2'] == [performer]
    assert len(loaded) == 2  # restore the original compatible scan, not a third inference


def test_subset_rebuild_does_not_change_other_scene_members_or_links(tmp_path, monkeypatch):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1', '2') == 0
    before = list(client.links['2'])
    strict = changed(config, detection={'min_face_pixels': 200})
    assert run(strict, services, '1') == 0
    assert client.links['1'] == ['77'] and client.links['2'] == before
    assert all(point[0] == '2' for group in partition(config) for point in group)


@pytest.mark.parametrize('visible', [True, False])
def test_uncertain_unlink_is_observed_or_blocks_retry(tmp_path, monkeypatch, visible):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1') == 0
    performer = client.links['1'][-1]
    real_remove, calls = client.remove_scene_person, []
    def lost(scene, person):
        calls.append((scene, person))
        if visible:
            real_remove(scene, person)
        raise TimeoutError('lost unlink acknowledgement')
    client.remove_scene_person = lost
    strict = changed(config, detection={'min_face_pixels': 200})
    assert run(strict, services, '1') == 1
    assert run(strict, services, '1') == (0 if visible else 1)
    assert len(calls) == 1
    if not visible:
        assert run(config, services, '1') == 1 and len(calls) == 1
        assert client.links['1'] == ['77', performer]


def test_manually_readded_retired_link_is_preserved(tmp_path, monkeypatch):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1') == 0
    performer = client.links['1'][-1]
    strict = changed(config, detection={'min_face_pixels': 200})
    assert run(strict, services, '1') == 0
    client.links['1'].append(performer)  # manual edit after acknowledged removal
    writes = list(client.writes)
    assert run(strict, services, '1') == 0 and client.writes == writes
    assert performer in client.links['1']


def test_preexisting_manual_link_to_owned_profile_is_not_adopted(tmp_path, monkeypatch):
    config, client, services, _ = library(tmp_path, monkeypatch, two_people())
    assert run(config, services, '1') == 0
    performer = client.links['1'][-1]
    client.links['2'] = [performer]
    assert run(config, services, '2') == 0
    strict = changed(config, detection={'min_face_pixels': 200})
    assert run(strict, services, '2') == 0 and client.links['2'] == [performer]


def test_legacy_uncertain_merge_blocks_only_involved_selection(tmp_path, monkeypatch):
    vectors = {**two_people(), '99': [[0., 0, 1., 0]] * 3}
    config, client, services, _ = library(tmp_path, monkeypatch, vectors)
    strict = changed(config, recognition={'match_threshold': .95})
    assert run(strict, services, '1', '2') == 0
    scope = instance_scope(client.url)
    with FaceDatabase(config.database) as db:
        people = db.connection.execute('SELECT * FROM people ORDER BY created,id').fetchall()
        source, target = people[1], people[0]
        db.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                              (source['id'], target['id'], 1., '{}', 'pending'))
        db.connection.execute('UPDATE members SET person=? WHERE person=?', (target['id'], source['id']))
        db.connection.commit()
        save_operation(db, scope=scope, scene='', person=source['id'], kind='merge', target=target['performer'],
            payload={'source': source['performer'], 'marker': 'not-observed'},
            precondition={'source_token': source['token'], 'destination_token': target['token']}, status='unknown')
    writes = list(client.writes)
    assert run(config, services, '1', '2') == 1 and client.writes == writes
    assert run(config, services, '99') == 0
    assert sum(w[0] == 'merge' for w in client.writes) == 0


def test_v3_migration_adds_origins_and_preserves_every_existing_table(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V3)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=3')
        db.execute('INSERT INTO metadata VALUES (?,?)', ('installation', '01234567-89ab-cdef-0123-456789abcdef'))
    with FaceDatabase(path) as db:
        assert db.connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
        assert db.connection.execute('SELECT * FROM identity_origins').fetchall() == []


def test_failed_v3_migration_is_atomic(tmp_path, monkeypatch):
    import stash_ai.recognition.database as database
    path = tmp_path / 'state.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V3)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=3')
    monkeypatch.setattr(database, 'REBUILD_SCHEMA', database.REBUILD_SCHEMA + '\nINVALID SQL;')
    with pytest.raises(sqlite3.Error):
        FaceDatabase(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert db.execute("SELECT name FROM sqlite_master WHERE name='identity_origins'").fetchone() is None


def test_better_historical_portrait_without_current_membership_cannot_win(tmp_path, monkeypatch):
    from stash_ai.recognition.models import checked_detection
    import stash_ai.recognition.scanning as scanning
    from tests.test_recognition_vtt_portraits import setup as portraits_setup
    config, client, services, _ = portraits_setup(tmp_path, monkeypatch, weights={'visibility': 0})
    assert run(config, services, '1') == 0
    performer = client.links['1'][-1]
    original = client.images[performer]
    with FaceDatabase(config.database) as db:
        row = db.connection.execute("SELECT id,metadata FROM portraits WHERE status='published'").fetchone()
        metadata = json.loads(row['metadata'])
        metadata['quality'].update(sharpness=100000., pixels=512, pose_deviation=0., exposure=127.5,
                                    clipped_fraction=0., complete=True, landmarks_inside=True)
        db.connection.execute('UPDATE portraits SET metadata=? WHERE id=?', (json.dumps(metadata), row['id']))
        db.connection.commit()
    face = checked_detection([150,70,150,150,187,122,262,122,225,145,196,175,253,175,.99])
    def backend(*args):
        model = Backend()
        model.detect = lambda image: [face]
        return model
    services.backend = backend
    assess = scanning.assess
    def poor_sharpness(*args):
        value = assess(*args)
        value['sharpness'] = 0.
        return value
    monkeypatch.setattr(scanning, 'assess', poor_sharpness)
    tuned = changed(config, detection={'min_face_pixels': 120})
    assert run(tuned, services, '1') == 0
    assert client.images[performer] != original


def test_remove_contract_uses_remove_and_validates_absence():
    client = RecognitionStash.__new__(RecognitionStash)
    def gql(query, variables):
        assert variables == {'input': {'ids': ['1'], 'performer_ids': {'ids': ['5'], 'mode': 'REMOVE'}}}
        return {'bulkSceneUpdate': [{'id': '1', 'performers': [{'id': '77'}]}]}
    client.gql = gql
    assert client.remove_scene_person('1', '5')['performers'] == [{'id': '77'}]
    client.gql = lambda *args: {'bulkSceneUpdate': [{'id': '1'}]}
    with pytest.raises(RuntimeError, match='acknowledgement'):
        client.remove_scene_person('1', '5')
