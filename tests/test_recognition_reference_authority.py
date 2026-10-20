"""Weak membership never trains strict references; view and cue regressions."""
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import _groups
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.reference_index import references
from tests.test_recognition_clustering import scan
from tests.test_recognition_cross_scene import config_for
from tests.test_recognition_pose_policy import configured, replay, FRONT
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('indexed', [False, True])
def test_context_member_cannot_poison_auxiliary_references(tmp_path, indexed):
    config = configured(tmp_path)
    borderline = np.array([.63, math.sqrt(1-.63**2), 0., 0.])
    profile = np.array([.75, -math.sqrt(1-.75**2), 0., 0.])
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': ([FRONT]*3 + [borderline, profile], {4})})
        rows = candidates(db, selected[0]['scan'], config)
        assert [r[2]['reason'] for r in rows[-2:]] == ['same_scene_identity', 'auxiliary_identity']
        assert all(r[0] == 'accepted' for r in rows)
        if indexed:
            with references(db.connection, 'scope', 'ns', config):
                assert len(_groups(db.connection, 'scope', 'ns', config.data['clustering'])[0].samples) == 3
        else:
            assert len(_groups(db.connection, 'scope', 'ns', config.data['clustering'])[0].samples) == 3


def test_role_update_invalidates_incremental_reference_snapshot(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': ([FRONT]*4, set())})
        with references(db.connection, 'scope', 'ns', config):
            assert len(_groups(db.connection, 'scope', 'ns', config.data['clustering'])[0].samples) == 4
            db.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','same_scene_identity') "
                                  "WHERE rowid=(SELECT MAX(rowid) FROM candidates)")
            assert len(_groups(db.connection, 'scope', 'ns', config.data['clustering'])[0].samples) == 3


def test_wrong_pose_view_does_not_shadow_valid_view_of_same_person(tmp_path):
    config = configured(tmp_path, view_prototypes_enabled=True)
    seed = np.array([.8, .5, math.sqrt(.11), 0.])
    wrong_pose = np.array([.8, .6, 0., 0.])
    query = np.array([.55, math.sqrt(1-.55**2), 0., 0.])
    with FaceDatabase(config.database) as db:
        sid, _ = scan(db, config, '1', [FRONT]*3 + [seed]*3 + [wrong_pose]*3 + [query]*2, count=11)
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',0.8) WHERE cue>=3")
        for row in db.connection.execute('SELECT id,geometry FROM observations WHERE cue BETWEEN 6 AND 8').fetchall():
            geometry = json.loads(row['geometry'])
            geometry[8] = 64
            db.connection.execute('UPDATE observations SET geometry=? WHERE id=?', (json.dumps(geometry), row['id']))
        db.connection.commit()
        rebuild_selected(db, [{'scene': '1', 'scan': sid}], config)
        assert [r[2]['reason'] for r in candidates(db, sid, config)[-2:]] == ['view_identity']*2


def test_unrelated_copresent_person_does_not_veto_cross_scene_confirmation(tmp_path):
    config = config_for(tmp_path)
    rival = np.array([0., 0., 1., 0.])
    with FaceDatabase(config.database) as db:
        first, _ = scan(db, config, '1', [FRONT] + [rival]*3, count=4)
        second, _ = scan(db, config, '2', [FRONT], count=1)
        db.connection.execute('UPDATE observations SET cue=0,seconds=0 WHERE scan=? AND cue=1', (first,))
        db.connection.commit()
        rebuild_selected(db, [{'scene': '1', 'scan': first}, {'scene': '2', 'scan': second}], config)
        assert candidates(db, second, config)[0][0] == 'accepted'
        people = db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]
        assert people == 2
