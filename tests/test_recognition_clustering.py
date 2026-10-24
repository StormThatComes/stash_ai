import json
from dataclasses import replace
import numpy as np
import pytest

from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.scanning import scan_scene
from tests.helpers.recognition import MediaClient, Backend


def setup(tmp_path, clustering=None):
    path = tmp_path / 'face.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0, max_pose_deviation: null, min_visible_face_fraction: 0, require_complete_face: false}\n' +
                    'recognition: {min_identity_support: 3}\n' +
                    ('clustering: ' + json.dumps(clustering) + '\n' if clustering else ''))
    return load(path)


def scan(state, config, scene, vectors=None, count=3):
    client = MediaClient(scene, count)
    backend = Backend()
    if vectors is not None:
        backend.dimensions = len(vectors[0])
        iterator = iter(vectors)
        backend.embed = lambda image, face: next(iterator)
    with SpriteInput(client, client.scene, config) as source:
        identifier, _ = scan_scene(state, 'scope', client.scene, config, 'ns', source, lambda: backend)
    # These fixtures assign arbitrary vectors to repeated placeholder pixels.
    # They model independent observations, not retained pixel-equality evidence.
    state.connection.execute("UPDATE observations SET quality=json_remove(quality,'$.evidence_crop_sha256') WHERE scan=?", (identifier,))
    state.connection.commit()
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


@pytest.mark.parametrize('dimensions', [128, 512])
def test_reference_usage_counts_model_bytes_sample_cap_and_cached_lookup(tmp_path, dimensions):
    config = setup(tmp_path, {'max_samples_per_person': 2})
    embedding = np.zeros(dimensions, dtype='float32')
    embedding[0] = 1
    events = []
    with FaceDatabase(config.database) as state:
        selected, _ = scan(state, config, '1', [embedding] * 3)
        result = cluster_scene(state, selected, config, progress=lambda **v: events.append(v))
        assert result[0]['status'] == 'accepted'
        assert events[-1] == {
            'bytes': 2 * dimensions * 4, 'budget_bytes': 134217728,
            'peak_bytes': 2 * dimensions * 4, 'people': 1, 'samples': 2, 'stage': 'retained'}
        assert cluster_scene(state, selected, config, progress=lambda **v: events.append(v)) == result
        assert events[-1]['stage'] == 'cached'
        assert events[-1]['bytes'] == events[-1]['peak_bytes'] == 2 * dimensions * 4
        assert events[-1]['samples'] == 2


def test_reference_peak_includes_withheld_groups_but_current_usage_does_not(tmp_path):
    config = setup(tmp_path)
    events = []
    with FaceDatabase(config.database) as state:
        selected, _ = scan(state, config, '1', count=1)
        result = cluster_scene(state, selected, config, progress=lambda **v: events.append(v))
        assert result[0]['status'] == 'withheld'
        assert events[-1]['bytes'] == events[-1]['samples'] == events[-1]['people'] == 0
        assert events[-1]['peak_bytes'] == 16
        assert any(v['stage'] == 'matching' and v['bytes'] == 16 for v in events)


def test_reference_usage_reports_over_budget_new_samples_and_rolls_back(tmp_path):
    config = setup(tmp_path)
    events = []
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        cluster_scene(state, first, config)
        # A tiny internal limit makes boundary checks cheap; YAML requires >=1 MiB.
        config = replace(config, data=plain(config.data))
        config.data['clustering']['max_reference_bytes'] = 64
        selected, _ = scan(state, config, '2', [np.array([0., 1., 0., 0.])] * 3)
        with pytest.raises(ValueError, match=r'Selected.*required=80 bytes, limit=64 bytes.*people=2, samples=5'):
            cluster_scene(state, selected, config, progress=lambda **v: events.append(v))
        assert events[-1]['bytes'] == events[-1]['peak_bytes'] == 80
        assert events[-1]['budget_bytes'] == 64
        assert events[-1]['stage'] == 'matching'
        assert state.connection.execute('SELECT COUNT(*) FROM candidates WHERE scan=?', (selected,)).fetchone()[0] == 0
        assert state.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 3


@pytest.mark.parametrize('column', ['scope', 'namespace'])
def test_loading_reference_usage_failure_reports_only_compatible_people(tmp_path, column):
    config = setup(tmp_path)
    events = []
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        cluster_scene(state, first, config)
        selected, _ = scan(state, config, '2')
        config = replace(config, data=plain(config.data))
        config.data['clustering']['max_reference_bytes'] = 32
        with pytest.raises(ValueError, match=r'Retained.*required=48 bytes, limit=32 bytes'):
            cluster_scene(state, selected, config, progress=lambda **v: events.append(v))
        assert events[-1]['stage'] == 'loading'
        assert events[-1]['bytes'] == 48 and events[-1]['samples'] == 3
        # Neither another instance nor another recognizer contributes references.
        state.connection.execute(f"UPDATE people SET {column}='other'")
        state.connection.commit()
        config.data['clustering']['max_reference_bytes'] = 64
        cluster_scene(state, selected, config, progress=lambda **v: events.append(v))
        assert events[-1]['bytes'] == 48 and events[-1]['people'] == 1


def test_empty_selected_scan_reports_no_reference_lookup(tmp_path):
    config = setup(tmp_path)
    events = []
    with FaceDatabase(config.database) as state:
        selected, _ = scan(state, config, '1')
        state.connection.execute("UPDATE observations SET rejected='synthetic' WHERE scan=?", (selected,))
        state.connection.commit()
        assert cluster_scene(state, selected, config, progress=lambda **v: events.append(v)) == []
        assert len(events) == 1 and events[0]['stage'] == 'no_observations'
        assert events[0]['bytes'] == events[0]['peak_bytes'] == 0


def test_reference_progress_is_throttled_without_losing_final_or_failure_usage(monkeypatch):
    from types import SimpleNamespace
    from stash_ai.recognition.reference_usage import ReferenceUsage
    now, events = [0.], []
    monkeypatch.setattr('stash_ai.recognition.reference_usage.time', SimpleNamespace(monotonic=lambda: now[0]))
    usage = ReferenceUsage(16016, lambda **v: events.append(v))
    usage.begin('matching')
    for _ in range(1000):
        usage.add([np.array([1., 0., 0., 0.], dtype='float32')], kind='Selected')
    assert len(events) == 1
    now[0] = 1.
    usage.add([np.array([1., 0., 0., 0.], dtype='float32')], kind='Selected')
    assert len(events) == 2 and events[-1]['bytes'] == 16016
    with pytest.raises(ValueError, match='required=16032'):
        usage.add([np.array([1., 0., 0., 0.], dtype='float32')], kind='Selected')
    assert len(events) == 3 and events[-1]['bytes'] == events[-1]['peak_bytes'] == 16032


def test_stricter_replay_withholds_unsafe_reassignment_without_empty_identities(tmp_path):
    from tests.test_recognition_identity_merges import changed
    config = setup(tmp_path)
    a,b = np.array([1.,0,0,0]),np.array([.8,.6,0,0])
    with FaceDatabase(config.database) as db:
        selected,_ = scan(db,config,'1',[a]*3+[b]*3,count=6)
        person = cluster_scene(db,selected,config)[0]['person']
        other,_ = scan(db,config,'2',[a]*3)
        cluster_scene(db,other,config)
        db.connection.execute('UPDATE people SET performer=? WHERE id=?', ('42',person))
        db.connection.commit()
        before = [tuple(row) for row in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        evidence = [tuple(row) for row in db.connection.execute('SELECT * FROM observations ORDER BY id')]
        strict = changed(config,recognition={'match_threshold':.98,'ambiguity_margin':.01})
        rows = cluster_scene(db,selected,strict)
        assert all(row['status']=='withheld' for row in rows)
        assert any(json.loads(row[0])['reason']=='policy_reassignment_requires_review'
                   for row in db.connection.execute('SELECT payload FROM candidates WHERE scan=?',(selected,)))
        assert before == [tuple(row) for row in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        assert evidence == [tuple(row) for row in db.connection.execute('SELECT * FROM observations ORDER BY id')]
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==1
        assert db.connection.execute('SELECT performer FROM people').fetchone()[0]=='42'
        assert cluster_scene(db,selected,strict)==rows


def test_valid_one_mib_budget_does_not_overlap_reference_reload_allocations(tmp_path,monkeypatch):
    import uuid
    import weakref
    from stash_ai.recognition import clustering
    config = setup(tmp_path,{'max_samples_per_person':128,'max_reference_bytes':1048576})
    a,b = np.zeros(1024,dtype='float32'),np.zeros(1024,dtype='float32')
    a[0],b[1]=1,1
    with FaceDatabase(config.database) as db:
        for scene,embedding in [('1',a),('2',b)]:
            selected,_ = scan(db,config,scene,[embedding]*3)
            person = cluster_scene(db,selected,config)[0]['person']
            row = list(db.connection.execute('SELECT * FROM observations WHERE scan=? LIMIT 1',(selected,)).fetchone())
            for index in range(125):
                values = list(row)
                values[0],values[2],values[3]=str(uuid.uuid4()),index+3,float(index+3)
                db.connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',values)
                db.connection.execute('INSERT INTO members VALUES (?,?)',(values[0],person))
        db.connection.commit()
        selected,_ = scan(db,config,'3',[a]*3)
        original,allocations,events = clustering.vector,[],[]
        def tracked(row):
            result=original(row)
            allocations.append(weakref.ref(result))
            return result
        def progress(**values):
            live=sum(ref().nbytes for ref in allocations if ref() is not None)
            events.append((values,live))
        monkeypatch.setattr(clustering,'vector',tracked)
        assert cluster_scene(db,selected,config,progress=progress)[0]['status']=='accepted'
        for values,live in events:
            if values['stage'] in ('loading','retained','cached'):
                assert live <= values['budget_bytes']
                assert live == values['bytes']
        assert events[-1][0]['bytes']==events[-1][0]['peak_bytes']==1048576


def test_default_accepts_one_detected_face_without_relaxing_merges(tmp_path):
    assert load(root=tmp_path).data['recognition']['min_identity_support'] == 1
    config = setup(tmp_path)
    # Reuse the fixture's disabled quality gates, but exercise the new default
    # singleton acceptance instead of the legacy 3-cue regression fixture.
    config = replace(config, data=plain(config.data))
    config.data['recognition']['min_identity_support'] = 1
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '1', count=1)
        result = cluster_scene(db, selected, config)
        assert result[0]['status'] == 'accepted'
        assert result[0]['observations'] == 1
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1
