"""One identity may own independent, verified pose/appearance view prototypes."""
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import _groups, matching_policy
from stash_ai.recognition.config import load, plain
from dataclasses import replace
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.view_prototypes import pose_view, confirmed_views
from tests.test_recognition_pose_policy import configured, membership, replay, FRONT

SEED = np.array([.8, .6, 0., 0.])
LATER = np.array([.55, math.sqrt(1-.55**2), 0., 0.])
OTHER = np.array([0., 1., 0., 0.])


def multiview(tmp_path, **changes):
    return configured(tmp_path, view_prototypes_enabled=True, **changes)


def policy_membership(db, config, scan):
    """Past candidate policies are historical; only current-policy decisions count."""
    return list(db.connection.execute(
        'SELECT o.cue,c.status,c.person,c.payload FROM candidates c '
        "JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
        'WHERE c.scan=? AND c.policy=? ORDER BY o.cue,o.id',
        (scan, matching_policy(config))))


def assigned_members(db, scan):
    return list(db.connection.execute(
        'SELECT m.observation,m.person FROM members m '
        'JOIN observations o ON o.id=m.observation '
        'WHERE o.scan=? ORDER BY o.cue,o.id', (scan,)))


def run_example(db, config, *, seed_count=3, late_count=2):
    return replay(db, config, {
        '1': ([FRONT]*3 + [SEED]*seed_count, set(range(3,3+seed_count))),
        '2': ([LATER]*late_count, set(range(late_count))),
    })


def test_confirmed_view_recovers_profiles_which_cannot_match_frontal_anchor(tmp_path):
    config = multiview(tmp_path)
    assert float(np.dot(SEED, FRONT)) > config.data['recognition']['match_threshold']
    assert float(np.dot(LATER, FRONT)) < config.data['recognition']['match_threshold']
    assert float(np.dot(LATER, SEED)) > config.data['clustering']['view_match_threshold']
    with FaceDatabase(config.database) as db:
        selected = run_example(db, config)
        first = membership(db, selected[0]['scan'])
        second = membership(db, selected[1]['scan'])
        assert [json.loads(row['payload'])['reason'] for row in first[3:]] == ['auxiliary_identity']*3
        assert [row['status'] for row in second] == ['accepted', 'accepted']
        assert {row['person'] for row in second} == {first[0]['person']}
        assert all(json.loads(row['payload'])['reason'] == 'view_identity' for row in second)
        proofs = [json.loads(row['payload'])['view_prototype'] for row in second]
        assert all(p['anchor_verified_support'] >= 3 and p['pose'] == 'other' for p in proofs)
        anchors = _groups(db.connection, 'scope', 'ns', config.data['clustering'])
        assert len(anchors) == 1 and len(anchors[0].samples) == 3
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0


def test_disabled_multiview_preserves_legacy_profile_rejection(tmp_path):
    config = configured(tmp_path, view_prototypes_enabled=False)
    with FaceDatabase(config.database) as db:
        selected = run_example(db, config)
        second = membership(db, selected[1]['scan'])
        assert all(row['status'] == 'withheld' for row in second)


def test_one_or_two_confirmations_cannot_promote_new_view(tmp_path):
    config = multiview(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = run_example(db, config, seed_count=2)
        assert all(row['status'] == 'withheld' for row in membership(db, selected[1]['scan']))


def test_single_new_view_match_cannot_attach(tmp_path):
    config = multiview(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = run_example(db, config, late_count=1)
        second = membership(db, selected[1]['scan'])
        assert second[0]['status'] == 'withheld'
        assert json.loads(second[0]['payload'])['reason'] == 'view_insufficient_support'



def test_one_identity_can_own_two_distinct_confirmed_view_clusters(tmp_path):
    other_view = np.array([.8, -.6, 0., 0.])
    assert float(np.dot(SEED, other_view)) < .8
    config = multiview(tmp_path)
    with FaceDatabase(config.database) as db:
        replay(db, config, {
            '1': ([FRONT]*3 + [SEED]*3 + [other_view]*3, set(range(3, 9))),
        })
        views = confirmed_views(
            db.connection, matching_policy(config), config.data['clustering'],
            remaining_bytes=config.data['clustering']['max_reference_bytes'])
        assert len(views) == 2
        assert len({view.group.identifier for view in views}) == 1
        assert all(view.support_count == 3 for view in views)
        assert all(view.pose == 'other' for view in views)


def test_competing_view_prototypes_respect_ambiguity_margin(tmp_path):
    a = np.array([.75, 0., math.sqrt(1-.75**2), 0.])
    b = np.array([0., .75, math.sqrt(1-.75**2), 0.])
    uncertain = np.array([.35, .35, math.sqrt(1-2*.35**2), 0.])
    assert float(np.dot(a, uncertain)) > .82
    assert float(np.dot(b, uncertain)) > .82
    config = multiview(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {
            '1': ([FRONT]*3 + [a]*3, {3,4,5}),
            '2': ([OTHER]*3 + [b]*3, {3,4,5}),
            '3': ([uncertain]*2, {0,1}),
        })
        assert all(row['status'] == 'withheld' for row in membership(db, selected[2]['scan']))
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 2


def test_views_never_seed_identity_or_merge_proofs(tmp_path):
    config = multiview(tmp_path, auxiliary_min_support=3)
    with FaceDatabase(config.database) as db:
        selected = replay(db, config, {
            '1': ([SEED]*3, {0,1,2}),
            '2': ([LATER]*3, {0,1,2}),
        })
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 0
        assert all(row['status'] == 'withheld' for item in selected
                   for row in membership(db, item['scan']))
        assert db.connection.execute('SELECT COUNT(*) FROM merge_support').fetchone()[0] == 0


def test_new_matching_options_change_replay_not_embedding_namespace(tmp_path):
    off = configured(tmp_path)
    on = multiview(tmp_path)
    assert matching_policy(off) != matching_policy(on)
    assert scan_identity(off, 'ns') == scan_identity(on, 'ns')


def test_promotion_prototypes_bounded_and_nontransitive(tmp_path):
    config = multiview(tmp_path, max_view_prototypes_per_person=1)
    with FaceDatabase(config.database) as db:
        selected = run_example(db, config)
        views = confirmed_views(db.connection, matching_policy(config), config.data['clustering'],
                                remaining_bytes=config.data['clustering']['max_reference_bytes'])
        # Only the three anchor-confirmed SEED observations are training votes.
        # The two second-pass matches cannot increase promotion support.
        assert len(views) == 1
        assert views[0].support_count == 3
        assert views[0].pose == 'other'


@pytest.mark.parametrize('field,value', [
    ('view_prototypes_enabled', 'true'), ('view_prototypes_enabled', 1),
    ('view_min_support', 1), ('view_min_support', 0),
    ('view_match_threshold', -0.1), ('view_match_threshold', 1.1),
    ('view_cohesion_threshold', float('nan')),
    ('view_ambiguity_margin', True), ('view_attach_min_support', 0),
    ('max_view_prototypes_per_person', 0), ('max_view_prototypes_per_person', 9),
])
def test_view_configuration_rejects_invalid_values(tmp_path, field, value):
    path = tmp_path/'invalid.yml'
    path.write_text('version: 1\nclustering: '+json.dumps({field: value})+'\n')
    with pytest.raises(ValueError, match=field):
        load(path)


@pytest.mark.parametrize('options', [
    {'view_prototypes_enabled': True},
    {'view_prototypes_enabled': True, 'max_anchor_pose_deviation': .2,
     'auxiliary_matching': False},
])
def test_views_require_anchor_policy_and_auxiliary_matching(tmp_path, options):
    path = tmp_path/'invalid.yml'
    path.write_text('version: 1\nclustering: '+json.dumps(options)+'\n')
    with pytest.raises(ValueError, match='view_prototypes_enabled'):
        load(path)



def test_loose_tight_loose_replay_reuses_embeddings_and_reproduces_views(tmp_path):
    config = multiview(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = run_example(db, config)
        first_scan, second_scan = (item['scan'] for item in selected)
        original = policy_membership(db, config, second_scan)
        assert len(original) == 2
        assert [row['status'] for row in original] == ['accepted', 'accepted']
        initial_members = assigned_members(db, second_scan)
        assert len(initial_members) == 2
        data = plain(config.data)
        data['clustering']['view_prototypes_enabled'] = False
        disabled = replace(config, data=data)
        assert matching_policy(config) != matching_policy(disabled)
        rebuild_selected(db, selected, disabled)
        # Prior candidate histories intentionally remain available by policy.
        assert len(membership(db, second_scan)) == 4
        current = policy_membership(db, disabled, second_scan)
        assert len(current) == 2
        assert [row['status'] for row in current] == ['withheld', 'withheld']
        assert assigned_members(db, second_scan) == []
        rebuild_selected(db, selected, config)
        restored = policy_membership(db, config, second_scan)
        assert len(restored) == 2
        assert [row['status'] for row in restored] == ['accepted', 'accepted']
        assert [row['person'] for row in restored] == [row['person'] for row in original]
        assert assigned_members(db, second_scan) == initial_members
        # No additional scan or embedding inference is required for policy tuning.
        assert len(db.connection.execute('SELECT id FROM scans').fetchall()) == 2


def test_view_reference_budget_fails_instead_of_silently_dropping_views(tmp_path):
    config = multiview(tmp_path)
    with FaceDatabase(config.database) as db:
        run_example(db, config)
        with pytest.raises(ValueError, match='max_reference_bytes'):
            confirmed_views(db.connection, matching_policy(config), config.data['clustering'],
                            remaining_bytes=1)


def test_pose_hint_keeps_signed_direction_and_does_not_claim_yaw_degrees():
    neutral = [16,16,128,128,49,60,111,60,80,84,56,108,104,108,.99]
    left = neutral.copy()
    right = neutral.copy()
    left[8] = 64
    right[8] = 96
    assert pose_view(neutral) == 'other'
    assert pose_view(left) == 'left'
    assert pose_view(right) == 'right'
    assert pose_view('not-json') == 'other'
