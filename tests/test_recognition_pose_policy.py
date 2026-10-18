"""Pose-aware replay: only anchors can seed people, profiles can attach safely."""
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import _groups, matching_policy
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.pose_policy import is_anchor
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.scanning import scan_identity
from tests.test_recognition_clustering import scan, setup

FRONT = np.array([1., 0., 0., 0.])
PROFILE = np.array([.8, .6, 0., 0.])
OTHER = np.array([0., 1., 0., 0.])


def configured(tmp_path, **clustering):
    return setup(tmp_path, {'max_anchor_pose_deviation': .2, **clustering})


def replay(db, config, scenes):
    selected = []
    for scene, (vectors, profile_cues) in scenes.items():
        identifier, _ = scan(db, config, scene, vectors, count=len(vectors))
        for cue in profile_cues:
            db.connection.execute(
                "UPDATE observations SET quality=json_set(quality,'$.pose_deviation',0.8) "
                "WHERE scan=? AND cue=?", (identifier, cue))
        db.connection.commit()
        selected.append({'scene': scene, 'scan': identifier})
    rebuild_selected(db, selected, config)
    return selected


def membership(db, scan):
    return list(db.connection.execute(
        'SELECT o.cue,c.status,c.person,c.payload FROM candidates c '
        "JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
        'WHERE c.scan=? ORDER BY o.cue', (scan,)))


def test_profile_attaches_to_anchor_without_becoming_reference(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': ([FRONT]*3 + [PROFILE], {3})})
        rows = membership(db, selected[0]['scan'])
        assert len(rows) == 4
        assert all(row['status'] == 'accepted' for row in rows)
        assert len({row['person'] for row in rows}) == 1
        assert json.loads(rows[-1]['payload'])['reason'] == 'auxiliary_identity'
        person = rows[0]['person']
        groups = _groups(db.connection, 'scope', 'ns', config.data['clustering'])
        assert len(groups) == 1 and groups[0].identifier == person
        assert len(groups[0].samples) == 3
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0


def test_only_profiles_cannot_create_identity(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': ([FRONT]*3, {0, 1, 2})})
        rows = membership(db, selected[0]['scan'])
        assert len(rows) == 3 and all(row['status'] == 'withheld' for row in rows)
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 0
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 0


def test_profile_without_anchor_match_stays_withheld(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {
            '1': ([FRONT]*3, set()),
            '2': ([OTHER]*3, {0, 1, 2}),
        })
        assert all(r['status'] == 'withheld' for r in membership(db, selected[1]['scan']))
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1


@pytest.mark.parametrize('minimum,expected', [(1, 'accepted'), (2, 'withheld')])
def test_auxiliary_support_needs_distinct_times(tmp_path, minimum, expected):
    config = configured(tmp_path, auxiliary_min_support=minimum)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': ([FRONT]*3 + [PROFILE], {3})})
        assert membership(db, selected[0]['scan'])[-1]['status'] == expected


def test_auxiliary_matching_can_be_disabled(tmp_path):
    config = configured(tmp_path, auxiliary_matching=False)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {'1': ([FRONT]*3 + [PROFILE], {3})})
        profile = membership(db, selected[0]['scan'])[-1]
        assert profile['status'] == 'withheld'
        assert json.loads(profile['payload'])['reason'] == 'auxiliary_disabled'


def test_pose_threshold_retunes_without_scan_identity_change(tmp_path):
    baseline = setup(tmp_path)
    strict = configured(tmp_path)
    assert matching_policy(strict) != matching_policy(baseline)
    assert scan_identity(strict, 'ns') == scan_identity(baseline, 'ns')


@pytest.mark.parametrize('value', [-0.01, 1.1, True, float('nan'), '0.2'])
def test_invalid_anchor_threshold_rejected(tmp_path, value):
    path = tmp_path / 'invalid.yml'
    path.write_text('version: 1\nclustering: ' + json.dumps({'max_anchor_pose_deviation': value}) + '\n')
    with pytest.raises(ValueError, match='max_anchor_pose_deviation'):
        load(path)


def test_missing_pose_cannot_seed_when_enabled():
    assert not is_anchor('{}', {'max_anchor_pose_deviation': .2})
    assert is_anchor('{}', {'max_anchor_pose_deviation': None})
