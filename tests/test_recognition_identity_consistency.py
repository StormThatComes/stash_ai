import pytest
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_clustering import setup as cluster_config, scan
from tests.test_recognition_rebuild import changed
import json
from dataclasses import replace
from stash_ai.recognition.clustering import cluster_scene


def test_cold_replay_removes_members_withheld_by_final_decision(tmp_path):
    import math
    config = changed(cluster_config(tmp_path), clustering={'max_samples_per_person':2,'cohesion_threshold':.35},
                     recognition={'min_identity_support':1,'min_merge_support':1,'ambiguity_margin':.05})
    angles = [[135,-45,0,45,30,0],[90,60,-30,-30,120,60],[15,105,45,-90,150,-90]]
    with FaceDatabase(config.database) as db:
        selected=[]
        for scene, values in enumerate(angles,1):
            vectors=[np.array([math.cos(math.radians(a)),math.sin(math.radians(a)),0,0]) for a in values]
            identifier,_=scan(db,config,str(scene),vectors,count=len(vectors))
            selected.append({'scene':str(scene),'scan':identifier})
        rebuild_selected(db,selected,config)
        rows=db.connection.execute("SELECT m.observation,c.status,c.person FROM members m JOIN candidates c "
            "ON json_extract(c.payload,'$.observation')=m.observation WHERE c.status='withheld'").fetchall()
        assert not rows


@pytest.mark.parametrize('joint', [False, True])
def test_blocked_better_rival_requires_unique_joint_assignment(tmp_path, monkeypatch, joint):
    if not joint:
        monkeypatch.setattr('stash_ai.recognition.joint_claims.recover', lambda *args: None)
    from tests.test_recognition_evidence_safety import pair_cues
    config = cluster_config(tmp_path)
    a = np.array([1., 0, 0, 0])
    b = np.array([.4, np.sqrt(.84), 0, 0])
    q = np.array([.8, .6, 0, 0])
    p = .8 * b + np.array([0, 0, .6, 0])
    assert abs(q @ a - q @ b) < config.data['recognition']['ambiguity_margin']
    with FaceDatabase(config.database) as db:
        first, _ = scan(db, config, '1', [a] * 3)
        second, _ = scan(db, config, '2', [b] * 3)
        selected, _ = scan(db, config, '3', [q, p] * 3, count=6)
        pair_cues(db, selected)
        query_ids = []
        for row in db.connection.execute('SELECT * FROM observations WHERE scan=?', (selected,)).fetchall():
            geometry = json.loads(row['geometry'])
            shifted = np.frombuffer(row['embedding'], dtype='<f4')[2] > .1
            for index in (0,2,4,6,8,10,12):
                geometry[index] = geometry[index] / 3 + (80 if shifted and index != 2 else 0)
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?', (json.dumps(geometry), row['id']))
            if not shifted: query_ids.append(row['id'])
        db.connection.commit()
        result = rebuild_selected(db, [{'scene':'1','scan':first},{'scene':'2','scan':second},{'scene':'3','scan':selected}], config)
        weaker = result[first][0]['person']
        for identifier in query_ids:
            member = db.connection.execute('SELECT person FROM members WHERE observation=?', (identifier,)).fetchone()
            if not joint:
                assert member is None
                continue
            assert member['person'] == weaker
            payload = json.loads(db.connection.execute(
                "SELECT payload FROM candidates WHERE json_extract(payload,'$.observation')=?",
                (identifier,)).fetchone()[0])
            assert payload['reference_role'] == 'attachment'
            proof = payload['shared_reference']['joint_assignment']
            assert len(proof) == 2 and len({claim['identity'] for claim in proof}) == 2
        if joint:
            stronger = result[second][0]['person']
            assert stronger != weaker
            peers = db.connection.execute('SELECT m.observation,m.person FROM members m JOIN observations o '
                'ON o.id=m.observation WHERE o.scan=?', (selected,)).fetchall()
            assert len(peers) == 6
            assert all(row['person'] == (weaker if row['observation'] in query_ids else stronger) for row in peers)


def test_reference_selection_preserves_partition_for_identical_vectors(tmp_path):
    import math
    config=changed(cluster_config(tmp_path),clustering={'max_samples_per_person':2,'cohesion_threshold':.35},
                   recognition={'min_identity_support':1,'min_merge_support':1,'ambiguity_margin':.05})
    angles=[[-90,60,135,-90,15,135],[-75,-30,-75,30,30,-60],[15,75,75,-15,-15,-90]]
    partitions=[]
    for sharpness in (0,1):
        current=replace(changed(config,detection={'min_sharpness':sharpness}),database=tmp_path/f'{sharpness}.sqlite3')
        with FaceDatabase(current.database) as state:
            selected=[]
            for scene,values in enumerate(angles,1):
                vectors=[np.array([math.cos(math.radians(a)),math.sin(math.radians(a)),0,0]) for a in values]
                identifier,_=scan(state,current,str(scene),vectors,count=6)
                selected.append({'scene':str(scene),'scan':identifier})
            assert state.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0]==18
            rebuild_selected(state,selected,current)
            groups={}
            for row in state.connection.execute('SELECT m.person,s.scene,o.cue FROM members m JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan'):
                groups.setdefault(row[0],[]).append((row[1],row[2]))
            partitions.append(sorted(sorted(group) for group in groups.values()))
    assert partitions[0] == partitions[1]


def test_reference_loader_rejects_cross_instance_membership(tmp_path):
    from stash_ai.recognition.clustering import _groups
    config=cluster_config(tmp_path)
    with FaceDatabase(config.database) as state:
        selected,_=scan(state,config,'1')
        cluster_scene(state,selected,config)
        state.connection.execute("UPDATE people SET scope='other-server',namespace='other-model'")
        state.connection.commit()
        with pytest.raises(ValueError, match='incompatible retained membership'):
            _groups(state.connection,'other-server','other-model',config.data['clustering'])


def test_identity_rebuild_rejects_duplicate_scene_versions(tmp_path):
    config=cluster_config(tmp_path)
    with FaceDatabase(config.database) as state:
        old,_=scan(state,config,'1')
        newer,_=scan(state,changed(config,detection={'min_sharpness':1}),'1')
        with pytest.raises(ValueError, match='one selected scan per scene'):
            rebuild_selected(state,[{'scene':'1','scan':old},{'scene':'1','scan':newer}],config)
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==0


def test_historical_binding_reference_peak_is_reported(tmp_path):
    config=changed(cluster_config(tmp_path),recognition={'min_identity_support':1},clustering={'max_samples_per_person':2})
    vectors=np.eye(128,dtype='float32')
    with FaceDatabase(config.database) as state:
        old,_=scan(state,config,'1',list(vectors[:20]),count=20)
        assert len(cluster_scene(state,old,config))==20
        historical_bytes=20*128*4
        current,_=scan(state,config,'1',[vectors[20]]*3)
        events=[]
        rebuild_selected(state,[{'scene':'1','scan':current}],config,progress=lambda **v:events.append(v))
        assert max(v['peak_bytes'] for v in events)>=historical_bytes
        assert len(state.connection.execute('SELECT * FROM members').fetchall())==3
