import pytest
import json
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_clustering import setup as cluster_config
from tests.test_recognition_run import setup as run_setup, run
import numpy as np
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_clustering import setup as cluster_config, scan
from tests.test_recognition_rebuild import changed


def test_boundary_observation_cache_retires_legacy_acceptance(tmp_path):
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.media import SpriteInput
    from tests.helpers.recognition import FACE, Backend, MediaClient
    config=cluster_config(tmp_path)
    geometry=list(FACE.row);geometry[-1]=config.data['detection']['score_threshold']
    old=Backend();old.detect=lambda image:[checked_detection(geometry)]
    client=MediaClient()
    with FaceDatabase(config.database) as db,SpriteInput(client,client.scene,config) as source:
        identifier,_=scan_scene(db,'scope',client.scene,config,'ns',source,lambda:old)
        # Simulate the prior single-candidate ORT boundary behavior. The fixed
        # decoder excludes it, but the unchanged scan policy never invokes it.
        for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
            quality=json.loads(row['quality']);quality.pop('alignment_valid',None)
            db.connection.execute('UPDATE observations SET quality=?,rejected=NULL,embedding=? WHERE id=?',
                (json.dumps(quality), np.array([1,0,0,0],dtype='<f4').tobytes(),row['id']))
        db.connection.commit()
        fixed=Backend();fixed.detect=lambda image:[]
        assert scan_scene(db,'scope',client.scene,config,'ns',source,lambda:fixed) == (identifier,True)
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0] == 0



def test_retained_padding_is_rechecked_under_new_geometry_contract_without_inference(tmp_path):
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene, geometry_contract
    from stash_ai.recognition.models import checked_detection
    from tests.helpers.recognition import FACE, Backend, MediaClient

    config = cluster_config(tmp_path)
    client = MediaClient()
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        identifier, reused = scan_scene(db, 'scope', client.scene, config, 'ns', source, Backend)
        assert not reused
        selection = [{'scene': '1', 'scan': identifier}]
        assert sum(r['observations'] for r in rebuild_selected(db, selection, config)[identifier]
                   if r['status'] == 'accepted') == 3
        rows = db.connection.execute('SELECT id,cue,geometry,quality FROM observations ORDER BY cue').fetchall()
        old_quality = json.loads(rows[0]['quality'])
        changed_geometry = list(FACE.row)
        for index in range(4, 14, 2):
            changed_geometry[index] += 90  # Still overlaps source, mostly padding.
        face = checked_detection(changed_geometry)
        old_quality['alignment_valid'] = True
        old_quality['geometry_contract'] = digest([
            'real-source-alignment-v2', changed_geometry, list(source.cues[rows[0]['cue']].rect)])
        db.connection.execute('UPDATE observations SET geometry=?,quality=? WHERE id=?', (
            json.dumps(changed_geometry), json.dumps(old_quality), rows[0]['id']))
        db.connection.commit()

        assert scan_scene(db, 'scope', client.scene, config, 'ns', source,
                          lambda: pytest.fail('retained scan must not load models')) == (identifier, True)
        bad = db.connection.execute('SELECT rejected,embedding,crop,quality FROM observations WHERE id=?',
                                    (rows[0]['id'],)).fetchone()
        assert bad['rejected'] == 'face_landmarks_invalid'
        assert bad['embedding'] is None and bad['crop'] is None
        new_quality = json.loads(bad['quality'])
        assert new_quality['alignment_valid'] is False
        assert new_quality['geometry_contract'] == geometry_contract(face, source.cues[rows[0]['cue']])
        assert db.connection.execute('SELECT COUNT(*) FROM members WHERE observation=?',
                                     (rows[0]['id'],)).fetchone()[0] == 0
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL').fetchone()[0] == 2


def test_incomplete_candidate_cache_rebuilds_without_removing_valid_link(tmp_path, monkeypatch):
    config, client, services, _ = run_setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = client.links['1'][-1]
    with FaceDatabase(config.database) as state:
        state.connection.execute('DELETE FROM candidates')
        state.connection.commit()
    assert run(config, services, '1') == 0
    assert performer in client.links['1']
    with FaceDatabase(config.database) as state:
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 3


def test_completed_scan_refuses_invalid_geometry_despite_alignment_flag(tmp_path,monkeypatch):
    config,client,services,_=run_setup(tmp_path,monkeypatch)
    assert run(config,services,'1') == 0
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE observations SET geometry='[]'")
        state.connection.commit()
    assert run(config,services,'1') == 1
    assert client.links['1']
    with FaceDatabase(config.database) as state:
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==3


def test_pilot_rejects_mixed_scan_versions_before_inference(tmp_path):
    from stash_ai.recognition.pilot import evaluate_observations
    from tests.helpers.recognition import Backend
    config=cluster_config(tmp_path)
    altered=changed(config,detection={'min_sharpness':1})
    a,b=np.array([1.,0,0,0]),np.array([.8,.6,0,0])
    with FaceDatabase(config.database) as db:
        old,_=scan(db,config,'1',[a]*3)
        new,_=scan(db,altered,'1',[b]*3)
        rows=db.connection.execute('SELECT id,scan,embedding FROM observations ORDER BY id').fetchall()
        labels={r['id']:r['scan'] for r in rows}
        iterator=iter(np.frombuffer(r['embedding'],dtype='<f4') for r in rows)
        class Adapter(Backend):
            def embed(self,*a):return next(iterator)
        run_id=db.start_run('scope',['1'],{})
        with pytest.raises(ValueError, match='one retained scan revision per scene'):
            evaluate_observations(db,run_id,{'sface':config},labels,assets=lambda c:{},backend=lambda *a:Adapter())
        # Each real complete scan has three eligible observations by itself.
        for identifier in (old,new):
            result=rebuild_selected(db,[{'scene':'1','scan':identifier}],config)[identifier]
            assert sum(r['observations'] for r in result if r['status']=='accepted') == 3
