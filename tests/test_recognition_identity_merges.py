"""Historical fragments reconcile oldest-first from new, independently timed evidence."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import cluster_scene, _groups
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_merges import canonical
from stash_ai.recognition.merge_publication import canonical_targets
from stash_ai.recognition.publication import publish_person, PendingRecognition, save_operation
from stash_ai.recognition.stash import RecognitionStash
from stash_ai.runtime.report import RunReport
from tests.helpers.recognition import Stash
from tests.test_recognition_clustering import setup, scan


def changed(config, **values):
    result = replace(config, data=plain(config.data))
    for section, fields in values.items():
        result.data[section].update(fields)
    return result


def fragments(db, config, vectors=None):
    vectors = vectors or [np.array([1., 0, 0, 0]), np.array([.8, .6, 0, 0])]
    strict = changed(config, recognition={'match_threshold': .98, 'ambiguity_margin': .01})
    people, scans = [], []
    for index, embedding in enumerate(vectors, 1):
        identifier, _ = scan(db, strict, str(index), [embedding] * 3)
        people.append(cluster_scene(db, identifier, strict)[0]['person'])
        scans.append(identifier)
    assert len(set(people)) == len(vectors)
    return people, scans


def bridge(db, config, vector=None, count=3):
    vector = np.array([1., 0, 0, 0]) + np.array([.8, .6, 0, 0]) if vector is None else vector
    vector = vector / np.linalg.norm(vector)
    return scan(db, config, '10', [vector] * count, count=count)[0]


def test_four_fragments_merge_into_oldest_reassign_history_and_never_reappear(tmp_path):
    config = setup(tmp_path)
    vectors = []
    for index in range(4):
        vector = np.zeros(5)
        vector[0], vector[index + 1] = math.sqrt(.8), math.sqrt(.2)
        vectors.append(vector)
    with FaceDatabase(config.database) as db:
        people, old_scans = fragments(db, config, vectors)
        # Age wins over UUID ordering or the scene processing order.
        for person, created in zip(people, [20., 10., 30., 40.]):
            db.connection.execute('UPDATE people SET created=? WHERE id=?', (created, person))
        db.connection.commit()
        selected = bridge(db, config, np.array([1., 0, 0, 0, 0]))
        result = cluster_scene(db, selected, config)
        assert result == [{'person': people[1], 'status': 'accepted', 'observations': 3}]
        aliases = db.connection.execute('SELECT * FROM identity_merges').fetchall()
        assert len(aliases) == 3
        assert all(canonical(db.connection, p) == people[1] for p in people)
        assert all(json.loads(row['evidence'])['support_seconds'] == [0., 1., 2.] for row in aliases)
        assert {row[0] for row in db.connection.execute('SELECT DISTINCT person FROM members')} == {people[1]}
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 15
        assert {row[0] for row in db.connection.execute('SELECT DISTINCT person FROM candidates')} == {people[1]}
        assert [g.identifier for g in _groups(db.connection, 'scope', 'ns', config.data['clustering'])] == [people[1]]
        statements = []
        db.connection.set_trace_callback(statements.append)
        assert cluster_scene(db, selected, config) == result
        assert not any(s.startswith(('UPDATE', 'INSERT', 'DELETE')) for s in statements)
        for old_scan in old_scans:
            assert cluster_scene(db, old_scan, config)[0]['person'] == people[1]
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 4  # durable aliases


@pytest.mark.parametrize('count,repeat_time', [(1, False), (3, True)])
def test_merge_requires_independent_selected_cue_times(tmp_path, count, repeat_time):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        fragments(db, config)
        selected = bridge(db, config, count=count)
        if repeat_time:
            db.connection.execute('UPDATE observations SET seconds=0 WHERE scan=?', (selected,))
            db.connection.commit()
        assert all(row['status'] == 'withheld' for row in cluster_scene(db, selected, config))
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


@pytest.mark.parametrize('cosine', [.28, .62])
def test_bridge_does_not_merge_incompatible_or_below_match_groups(tmp_path, cosine):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        vectors = [np.array([1., 0, 0, 0]), np.array([cosine, math.sqrt(1 - cosine ** 2), 0, 0])]
        fragments(db, config, vectors)
        selected = bridge(db, config, sum(vectors))
        assert all(row['status'] == 'withheld' for row in cluster_scene(db, selected, config))
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_union_keeps_all_constituents_to_block_transitive_single_link_merge(tmp_path):
    config = setup(tmp_path)
    a, b, c = np.array([1.,0,0,0]), np.array([.8,.6,0,0]), np.array([.28,.96,0,0])
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config, [a,b,c])
        selected = bridge(db, config, b)
        cluster_scene(db, selected, config)
        assert len(_groups(db.connection, 'scope', 'ns', config.data['clustering'])) == 2
        assert canonical(db.connection, people[1]) == people[0]
        assert canonical(db.connection, people[2]) == people[2]


def test_one_bad_representative_blocks_merge_even_if_average_passes(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        people, old = fragments(db, config)
        bad = np.array([.58, math.sqrt(1 - .58 ** 2), 0, 0], dtype='<f4')
        db.connection.execute('UPDATE observations SET embedding=? WHERE id=(SELECT id FROM observations WHERE scan=? LIMIT 1)',
                             (bad.tobytes(), old[1]))
        db.connection.commit()
        selected = bridge(db, config)
        cluster_scene(db, selected, config)
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()
        assert canonical(db.connection, people[1]) == people[1]


def test_two_historical_faces_in_same_cue_are_negative_merge_evidence(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        _, old = fragments(db, config)
        db.connection.execute('UPDATE observations SET scan=? WHERE scan=?', (old[0], old[1]))
        db.connection.commit()
        selected = bridge(db, config)
        assert all(r['status'] == 'withheld' for r in cluster_scene(db, selected, config))
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_two_new_distinct_faces_in_same_cue_block_merge_before_alias_is_written(tmp_path):
    config = setup(tmp_path)
    a, b = np.array([1.,0,0,0]), np.array([.8,.6,0,0])
    with FaceDatabase(config.database) as db:
        fragments(db, config, [a,b])
        selected, _ = scan(db, config, '10', [a,b] * 3, count=6)
        for row in db.connection.execute('SELECT id,cue FROM observations WHERE scan=?', (selected,)).fetchall():
            db.connection.execute('UPDATE observations SET cue=?,seconds=? WHERE id=?',
                                  (row['cue'] // 2, row['cue'] // 2, row['id']))
        db.connection.commit()
        assert len([r for r in cluster_scene(db, selected, config) if r['status'] == 'accepted']) == 2
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_reference_budget_failure_rolls_back_merge_and_all_member_transfers(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config)
        before = [tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        selected = bridge(db, config)
        small = changed(config, clustering={'max_reference_bytes': 128})
        with pytest.raises(ValueError, match='references.*budget'):
            cluster_scene(db, selected, small)
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()
        assert [tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')] == before
        assert canonical(db.connection, people[1]) == people[1]


def published_fragments(db, config, client):
    people, scans = fragments(db, config)
    report = RunReport(write=lambda _: None)
    performers = [publish_person(db, client, report, 'scope', str(i), p) for i,p in enumerate(people, 1)]
    selected = bridge(db, config)
    assert cluster_scene(db, selected, config)[0]['person'] == people[0]
    return people, scans, performers, report


def test_stash_merge_preserves_old_id_name_image_notes_and_all_scene_links(tmp_path):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, scans, performers, report = published_fragments(db, config, client)
        oldest, youngest = performers
        client.profiles[oldest].update(name='Oldest name', image_path='old-photo')
        client.profiles[oldest]['details'] += '\nOld notes'
        client.profiles[youngest]['details'] += '\nNew notes'
        client.links['999'] = ['77', youngest]  # unselected historical association
        assert publish_person(db, client, report, 'scope', '10', people[0]) == oldest
        assert youngest not in client.profiles
        assert client.links['1'] == ['77', oldest] and client.links['2'] == [oldest]
        assert client.links['999'] == ['77', oldest] and client.links['10'] == [oldest]
        assert client.profiles[oldest]['name'] == 'Oldest name' and client.profiles[oldest]['image_path'] == 'old-photo'
        assert 'Old notes' in client.profiles[oldest]['details'] and 'New notes' in client.profiles[oldest]['details']
        before = list(client.writes)
        assert publish_person(db, client, report, 'scope', '10', people[0]) == oldest
        assert client.writes == before
        with pytest.raises(PendingRecognition, match='Merged identity'):
            publish_person(db, client, report, 'scope', '2', people[1])
        ready = {p: {'person': p, 'performer': f, 'scans': [{'scene': str(i), 'scan': s}]}
                 for i,(p,f,s) in enumerate(zip(people, performers, scans), 1)}
        assert canonical_targets(db, ready) == [{'person': people[0], 'performer': oldest,
            'scans': [{'scene': '1', 'scan': scans[0]}, {'scene': '2', 'scan': scans[1]}]}]


@pytest.mark.parametrize('visible', [True, False])
def test_lost_merge_response_observes_marker_or_blocks_blind_retry(tmp_path, visible):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _, performers, report = published_fragments(db, config, client)
        client.fail_merge, client.merge_visible = True, visible
        with pytest.raises(TimeoutError):
            publish_person(db, client, report, 'scope', '10', people[0])
        assert db.connection.execute("SELECT status FROM publications WHERE kind='merge'").fetchone()[0] == 'unknown'
        if visible:
            assert publish_person(db, client, report, 'scope', '10', people[0]) == performers[0]
            assert db.connection.execute('SELECT status FROM identity_merges').fetchone()[0] == 'observed'
        else:
            with pytest.raises(PendingRecognition, match='no blind merge retry'):
                publish_person(db, client, report, 'scope', '10', people[0])
            with pytest.raises(PendingRecognition, match='metadata withheld'):
                canonical_targets(db, {people[0]: {'person': people[0], 'scans': []}})
        assert sum(w[0] == 'merge' for w in client.writes) == 1


def test_manual_source_deletion_does_not_prove_unknown_merge(tmp_path):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _, performers, report = published_fragments(db, config, client)
        client.fail_merge, client.merge_visible = True, False
        with pytest.raises(TimeoutError):
            publish_person(db, client, report, 'scope', '10', people[0])
        del client.profiles[performers[1]]
        with pytest.raises(PendingRecognition, match='unresolved'):
            publish_person(db, client, report, 'scope', '10', people[0])


@pytest.mark.parametrize('which', [0, 1])
def test_ownership_edits_withhold_destructive_merge(tmp_path, which):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _, performers, report = published_fragments(db, config, client)
        client.profiles[performers[which]]['details'] = 'Manually detached'
        before = list(client.writes)
        with pytest.raises(PendingRecognition):
            publish_person(db, client, report, 'scope', '10', people[0])
        assert client.writes == before and all(p in client.profiles for p in performers)


def test_local_unpublished_fragment_is_not_created_just_to_merge(tmp_path):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config)
        selected = bridge(db, config)
        cluster_scene(db, selected, config)
        publish_person(db, client, RunReport(write=lambda _: None), 'scope', '10', people[0])
        assert db.connection.execute('SELECT status FROM identity_merges').fetchone()[0] == 'local'
        assert sum(w[0] == 'create' for w in client.writes) == 1
        assert not any(w[0] == 'merge' for w in client.writes)


def test_unpublished_fragment_with_uncertain_creation_is_not_discarded(tmp_path):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config)
        selected = bridge(db, config)
        cluster_scene(db, selected, config)
        save_operation(db, scope='scope', scene='', person=people[1], kind='create', target=None,
                       payload={}, precondition={}, status='unknown')
        with pytest.raises(PendingRecognition, match='creation is unresolved'):
            publish_person(db, client, RunReport(write=lambda _: None), 'scope', '10', people[0])
        assert sum(w[0] == 'create' for w in client.writes) == 1
        assert not any(w[0] == 'merge' for w in client.writes)


def test_merge_graphql_contract_and_nullable_performer_lookup():
    client = RecognitionStash.__new__(RecognitionStash)
    calls = []
    def gql(query, variables):
        calls.append((query, variables))
        return {'performerMerge': {'id': '5'}} if 'Merge' in query else {'findPerformer': None}
    client.gql = gql
    assert client.merge_person('6', '5', 'notes')['id'] == '5'
    assert calls[0][1] == {'input': {'source': ['6'], 'destination': '5', 'values': {'id': '5', 'details': 'notes'}}}
    assert client.lookup_performer('6') is None
    with pytest.raises(RuntimeError, match='ID'):
        client.performer('6')
    with pytest.raises(ValueError, match='itself'):
        client.merge_person('5', '5', '')


@pytest.mark.parametrize('lost_response', [False, True])
def test_survivor_can_later_merge_into_an_older_identity_with_transitive_recovery(tmp_path, lost_response):
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config, [np.array([.8,.6,0,0]), np.array([1.,0,0,0]), np.array([.8,-.6,0,0])])
        report = RunReport(write=lambda _: None)
        performers = [publish_person(db, client, report, 'scope', str(i), p) for i,p in enumerate(people, 1)]
        selected = bridge(db, config, np.array([1.8,-.6,0,0]))
        assert cluster_scene(db, selected, config)[0]['person'] == people[1]
        client.fail_merge = lost_response
        if lost_response:
            with pytest.raises(TimeoutError):
                publish_person(db, client, report, 'scope', '10', people[1])
        else:
            publish_person(db, client, report, 'scope', '10', people[1])
        relaxed = changed(config, recognition={'match_threshold': .5}, clustering={'cohesion_threshold': .2})
        later, _ = scan(db, relaxed, '11', [np.array([1.,0,0,0])] * 3)
        assert cluster_scene(db, later, relaxed)[0]['person'] == people[0]
        assert publish_person(db, client, report, 'scope', '11', people[0]) == performers[0]
        assert all(canonical(db.connection, p) == people[0] for p in people)
        assert all(p not in client.profiles for p in performers[1:])
        assert all(client.links[s][-1] == performers[0] for s in ('1','2','3','10','11') if client.links.get(s))
        assert sum(w[0] == 'merge' for w in client.writes) == 2
        previous = list(client.writes)
        publish_person(db, client, report, 'scope', '11', people[0])
        assert client.writes == previous


@pytest.mark.parametrize('column', ['scope', 'namespace'])
def test_merge_evidence_is_scoped_to_compatible_active_identities(tmp_path, column):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config)
        db.connection.execute(f'UPDATE people SET {column}=? WHERE id=?', ('other', people[1]))
        db.connection.commit()
        selected = bridge(db, config)
        assert cluster_scene(db, selected, config)[0]['person'] == people[0]
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_confirmed_remote_merge_survives_interrupted_alias_status_commit(tmp_path, monkeypatch):
    import stash_ai.recognition.merge_publication as module
    config, client = setup(tmp_path), Stash()
    with FaceDatabase(config.database) as db:
        people, _, performers, report = published_fragments(db, config, client)
        original = module._complete
        monkeypatch.setattr(module, '_complete', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('interrupted local commit')))
        with pytest.raises(RuntimeError, match='local commit'):
            publish_person(db, client, report, 'scope', '10', people[0])
        assert db.connection.execute("SELECT status FROM publications WHERE kind='merge'").fetchone()[0] == 'confirmed'
        monkeypatch.setattr(module, '_complete', original)
        assert publish_person(db, client, report, 'scope', '10', people[0]) == performers[0]
        assert sum(w[0] == 'merge' for w in client.writes) == 1


def test_supported_person_and_reflection_can_merge_despite_historical_cooccurrence(tmp_path):
    config = setup(tmp_path)
    a, b = np.array([1.,0,0,0]), np.array([.95, math.sqrt(1-.95**2),0,0])
    with FaceDatabase(config.database) as db:
        people, old = fragments(db, config, [a,b])
        db.connection.execute('UPDATE observations SET scan=? WHERE scan=?', (old[0], old[1]))
        # Controlled person/reflection ground truth; spatial separation is
        # deliberately irrelevant to the production identity evidence rule.
        geometry = [100,4,48,48,110,16,138,16,124,28,114,40,134,40,.99]
        db.connection.execute('UPDATE observations SET geometry=? WHERE id IN '
                              '(SELECT observation FROM members WHERE person=?)', (json.dumps(geometry),people[1]))
        db.connection.commit()
        selected = bridge(db, config, a+b)
        rows = cluster_scene(db, selected, config)
        assert rows == [{'person':people[0], 'status':'accepted', 'observations':3}]
        evidence = json.loads(db.connection.execute('SELECT evidence FROM identity_merges').fetchone()[0])
        assert evidence['cooccurring'] is True and evidence['cooccurrence_merge_threshold'] == .9
        assert evidence['support_seconds'] == [0.,1.,2.]
        assert canonical(db.connection, people[1]) == people[0]
        assert cluster_scene(db, selected, config) == rows


def test_uncertain_cooccurring_pair_records_reason_instead_of_claiming_distinct_people(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        people, old = fragments(db, config)
        db.connection.execute('UPDATE observations SET scan=? WHERE scan=?', (old[0],old[1]))
        db.connection.commit()
        selected = bridge(db, config)
        assert all(row['status']=='withheld' for row in cluster_scene(db, selected, config))
        records = [json.loads(row[0]) for row in db.connection.execute('SELECT payload FROM candidates WHERE scan=?', (selected,))]
        assert any(row.get('merge_withheld_reason')=='cooccurrence_identity_evidence_insufficient' for row in records)
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_duplicate_detections_in_one_cue_do_not_impose_a_one_person_rule(tmp_path):
    config = setup(tmp_path)
    a = np.array([1.,0,0,0])
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '1', [a]*6, count=6)
        for row in db.connection.execute('SELECT id,cue FROM observations WHERE scan=?', (selected,)).fetchall():
            db.connection.execute('UPDATE observations SET cue=?,seconds=? WHERE id=?',
                                  (row['cue']//2,row['cue']//2,row['id']))
        db.connection.commit()
        rows = cluster_scene(db, selected, config)
        assert len(rows)==1 and rows[0]['status']=='accepted' and rows[0]['observations']==6


def test_merging_accumulates_independent_selected_scenes(tmp_path):
    config = setup(tmp_path)
    bridge_vector = np.array([1.8, .6, 0., 0.])
    bridge_vector /= np.linalg.norm(bridge_vector)
    with FaceDatabase(config.database) as db:
        people, _ = fragments(db, config)
        for count, scene in enumerate(('10', '11', '12'), 1):
            selected, _ = scan(db, config, scene, [bridge_vector], count=1)
            cluster_scene(db, selected, config)
            assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == (1 if count == 3 else 0)
        evidence = json.loads(db.connection.execute('SELECT evidence FROM identity_merges').fetchone()[0])
        assert {item['scene'] for item in evidence['support_cues']} == {'10', '11', '12'}
        assert len(evidence['support_cues']) == 3
        assert canonical(db.connection, people[1]) == people[0]


def test_old_scan_does_not_supply_support_after_newer_detection_scan(tmp_path):
    config = setup(tmp_path)
    bridge_vector = np.array([1.8, .6, 0., 0.])
    bridge_vector /= np.linalg.norm(bridge_vector)
    with FaceDatabase(config.database) as db:
        fragments(db, config)
        for scene in ('10', '11'):
            selected, _ = scan(db, config, scene, [bridge_vector], count=1)
            cluster_scene(db, selected, config)
        # Later completed scan for scene 10 supersedes its stored nominations.
        new_detection = changed(config, detection={'min_sharpness': 1.0})
        rescanned, _ = scan(db, new_detection, '10', [bridge_vector], count=1)
        cluster_scene(db, rescanned, new_detection)
        third, _ = scan(db, config, '12', [bridge_vector], count=1)
        cluster_scene(db, third, config)
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_threshold_change_does_not_reuse_old_merge_nominations(tmp_path):
    config = setup(tmp_path)
    bridge_vector = np.array([1.8, .6, 0., 0.])
    bridge_vector /= np.linalg.norm(bridge_vector)
    with FaceDatabase(config.database) as db:
        fragments(db, config)
        for scene in ('10', '11'):
            selected, _ = scan(db, config, scene, [bridge_vector], count=1)
            cluster_scene(db, selected, config)
        tuned = changed(config, recognition={'match_threshold': .75})
        selected, _ = scan(db, tuned, '12', [bridge_vector], count=1)
        cluster_scene(db, selected, tuned)
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


def test_same_timestamp_in_one_scene_counts_only_once_across_repeated_faces(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        fragments(db, config)
        selected = bridge(db, config)
        db.connection.execute('UPDATE observations SET seconds=0 WHERE scan=?', (selected,))
        db.connection.commit()
        cluster_scene(db, selected, config)
        assert not db.connection.execute('SELECT * FROM identity_merges').fetchall()


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_legacy_merge_scope_follows_transitive_aliases(tmp_path):
    from tests.test_recognition_clustering import setup,scan
    from tests.test_recognition_identity_merges import fragments
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.publication import save_operation
    from stash_ai.recognition.scene_sync import reconcile_legacy_merges
    config = setup(tmp_path)
    with FaceDatabase(config.database) as db:
        people,scans = fragments(db,config,[np.array([1.,0,0,0]),np.array([.8,.6,0,0]),np.array([0.,0,1,0])])
        root,middle,source = people
        for person,performer in zip(people,['101','102','103']):
            db.connection.execute('UPDATE people SET performer=? WHERE id=?',(performer,person))
        for left,right,status in [(source,middle,'pending'),(middle,root,'pending')]:
            db.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(left,right,1.,'{}',status))
        db.connection.execute('UPDATE members SET person=?',(root,))
        db.connection.commit()
        save_operation(db,scope='scope',scene='',person=source,kind='merge',target='102',
                       payload={'source':'103','marker':'unobserved'},
                       precondition={'source_token':f'stash-ai-recognition:{db.installation}:{source}',
                                     'destination_token':f'stash-ai-recognition:{db.installation}:{middle}'},status='unknown')
        reads = []
        client = SimpleNamespace(find_token=lambda token:reads.append(token) or [],lookup_performer=lambda x:None)
        from stash_ai.recognition.publication import PendingRecognition
        with pytest.raises(PendingRecognition, match='unresolved'):
            reconcile_legacy_merges(db,client,'scope',['1','2','3'])
        assert reads == ['unobserved']
        assert db.connection.execute("SELECT status FROM publications WHERE kind='merge'").fetchone()[0] == 'unknown'

