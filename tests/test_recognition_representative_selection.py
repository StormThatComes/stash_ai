"""Reference diversity and eye-frame pose invariants."""
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import _groups, cluster_scene, reference_context
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.quality import pose_measurements
from stash_ai.recognition.reference_index import references
from tests.test_recognition_clustering import scan, setup
from tests.helpers.recognition import FACE


def test_late_scene_and_distinct_view_replace_redundant_early_cues(tmp_path):
    config = setup(tmp_path, {'max_samples_per_person': 2})
    front = np.array([1.,0.,0.,0.])
    other = np.array([.8,.6,0.,0.])
    with FaceDatabase(config.database) as db:
        first, _ = scan(db, config, '1', [front]*20, count=20)
        cluster_scene(db, first, config)
        second, _ = scan(db, config, '2', [other]*3)
        cluster_scene(db, second, config)
        full = _groups(db.connection, 'scope','ns',config.data['clustering'])
        assert len(full) == 1 and len(full[0].samples) == 2
        assert sorted(round(float(v[0]), 2) for v in full[0].samples) == [.8,1.]
        with references(db.connection, 'scope','ns',config):
            assert reference_context(_groups(db.connection,'scope','ns',config.data['clustering'])) == reference_context(full)


def test_better_quality_can_replace_early_identical_reference(tmp_path):
    config = setup(tmp_path, {'max_samples_per_person':2})
    with FaceDatabase(config.database) as db:
        sid, _ = scan(db,config,'1',count=3)
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.sharpness',0,'$.pixels',32) WHERE cue<2")
        db.connection.commit()
        cluster_scene(db,sid,config)
        group = _groups(db.connection,'scope','ns',config.data['clustering'])[0]
        best = db.connection.execute('SELECT id FROM observations WHERE cue=2').fetchone()[0]
        assert best in group.sample_ids


@pytest.mark.parametrize('degrees', [-30,-15,15,30])
def test_roll_does_not_turn_a_frontal_face_into_auxiliary(degrees):
    points = np.array(FACE.landmarks)
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    rotated = (points-80) @ rotation.T + 80
    baseline = pose_measurements(points)
    actual = pose_measurements(rotated)
    assert actual['pose_deviation'] == pytest.approx(baseline['pose_deviation'])
    assert actual['roll_deviation'] > .2


def test_incremental_scene_append_matches_full_selector_without_reading_crops(tmp_path):
    from stash_ai.recognition.reference_index import _active
    config = setup(tmp_path, {'max_samples_per_person': 2})
    front, other = np.array([1.,0.,0.,0.]), np.array([.8,.6,0.,0.])
    with FaceDatabase(config.database) as db:
        first, _ = scan(db,config,'1',[front]*40,count=40)
        second, _ = scan(db,config,'2',[other]*3)
        cluster_scene(db,first,config)
        reads=[]
        import sqlite3
        def authorize(action,table,column,*_):
            if action == sqlite3.SQLITE_READ and table == 'observations' and column == 'crop':
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        with references(db.connection,'scope','ns',config):
            _groups(db.connection,'scope','ns',config.data['clustering'])
            cluster_scene(db,second,config)
            db.connection.set_authorizer(authorize)
            indexed=_groups(db.connection,'scope','ns',config.data['clustering'])
            token=_active.set(None)
            try:
                full=_groups(db.connection,'scope','ns',config.data['clustering'])
            finally:
                _active.reset(token)
                db.connection.set_authorizer(None)
            assert reference_context(indexed)==reference_context(full)
