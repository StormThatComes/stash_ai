"""Head composition outranks quality and auto tiers without dropping reserves."""
import json

import pytest

from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.portrait_framing import person_bounds, head_framing
from stash_ai.recognition.portraits import fill_portrait
from tests.helpers.recognition import Visibility
from tests.test_recognition_body_integration import seeded, crop
from tests.test_recognition_paired_portraits import paired, SMALL_FACE, NoPairedThumbnail
from tests.test_recognition_portrait_precedence import evidence, choose, forbidden, add_body


@pytest.mark.parametrize('x', [100, 200])
@pytest.mark.parametrize('scale', [1, 2])
def test_person_crop_centers_head_instead_of_only_containing_it(x, scale):
    geometry = dict(bounds=[300, 500, 300+300*scale, 500+200*scale],
                    keypoints=[[300+x*scale, 550]] + [[-1, -1]]*16)
    bounds = person_bounds((300, 200), geometry, .8)
    assert bounds == [x-80, 0, x+80, 200]
    assert head_framing((300, 200), geometry, bounds)['fallback'] is False


@pytest.mark.parametrize('x', [10, 290])
@pytest.mark.parametrize('ratio', [None, .8])
def test_unavoidable_source_edge_is_a_reserve(x, ratio):
    geometry = dict(bounds=[0, 0, 300, 200], keypoints=[[x, 50]])
    bounds = person_bounds((300, 200), geometry, ratio)
    assert 0 <= bounds[0] < bounds[2] <= 300
    composition = head_framing((300, 200), geometry, bounds)
    assert composition['fallback'] and composition['reason'] == 'head_off_center'


def test_missing_head_cannot_authorize_centered_pose():
    geometry = dict(bounds=[0, 0, 300, 200], keypoints=[[-1, -1]]*17)
    assert head_framing((300, 200), geometry, [70, 0, 230, 200])['reason'] == 'head_unavailable'


def side_pair(evidence, index=2):
    # An excellent face at the edge of a wide person box (e.g. another person
    # fills its centre). Membership is valid; portrait composition is poor.
    paired(evidence, dict(bounds=[0, 0, 160, 160], keypoints=[[80, 30]]), index=index)
    face = list(SMALL_FACE.row)
    face[0] -= 50
    for column in (4, 6, 8, 10, 12):
        face[column] -= 50
    db = evidence[0].connection
    db.execute('UPDATE observations SET geometry=? WHERE id=?', (encoded(face), evidence[4][index]))
    db.commit()


@pytest.mark.parametrize('ratio', [None, .8])
def test_centered_paired_candidate_beats_higher_quality_edge_candidate(evidence, ratio):
    paired(evidence)
    side_pair(evidence)
    evidence[1].data['portraits']['aspect_ratio'] = ratio
    winner = choose(evidence)
    metadata = json.loads(winner['metadata'])
    assert metadata['observation'] == evidence[4][0]
    assert not metadata['head_framing']['fallback']
    assert choose(evidence, forbidden)['id'] == winner['id']


def test_centered_retained_person_beats_live_off_center_candidate(evidence):
    paired(evidence)
    side_pair(evidence)
    metadata = json.loads(choose(evidence, NoPairedThumbnail)['metadata'])
    assert metadata['observation'] == evidence[4][0]
    assert metadata['source_type'] == 'person_crop'
    assert not metadata['head_framing']['fallback']


def test_unusable_centered_candidate_keeps_off_center_reserve(evidence):
    paired(evidence, retained=False)
    side_pair(evidence)
    db = evidence[0].connection
    db.execute('DELETE FROM members WHERE observation NOT IN (?,?)', (evidence[4][0], evidence[4][2]))
    db.commit()
    metadata = json.loads(choose(evidence, NoPairedThumbnail)['metadata'])
    assert metadata['observation'] == evidence[4][2]
    assert metadata['head_framing']['fallback']


@pytest.mark.parametrize('other_faces', [True, False])
def test_off_center_paired_is_only_used_when_no_better_category_exists(evidence, other_faces):
    side_pair(evidence, index=0)
    if not other_faces:
        evidence[0].connection.execute('DELETE FROM members WHERE observation!=?', (evidence[4][0],))
        evidence[0].connection.commit()
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['selection_category'] == ('face' if other_faces else 'face+pose')
    if not other_faces:
        assert metadata['head_framing']['reason'] == 'head_off_center'


def test_paired_crop_follows_verified_face_over_competing_pose_head(evidence):
    paired(evidence, dict(bounds=[0, 0, 160, 160], keypoints=[[130, 30]]))
    evidence[1].data['portraits']['aspect_ratio'] = .5
    metadata = json.loads(choose(evidence)['metadata'])
    # Verified face is centred at x=66; the pose estimator points to x=130.
    assert metadata['crop'] == [26, 0, 106, 160]
    assert metadata['head_framing']['horizontal_offset'] == 0


def test_changed_face_position_invalidates_completed_paired_ranking(evidence):
    paired(evidence)
    side_pair(evidence)
    assert json.loads(choose(evidence)['metadata'])['observation'] == evidence[4][0]
    paired(evidence, index=2)
    assert json.loads(choose(evidence)['metadata'])['observation'] == evidence[4][2]


@pytest.mark.parametrize('head', [10, None])
def test_custom_pose_priority_still_defers_bad_or_unknown_composition(evidence, head):
    identifier = add_body(evidence)
    evidence[1].data['portraits']['auto_precedence'] = ['pose', 'face', 'face+pose']
    db = evidence[0].connection
    geometry = dict(bounds=[0, 0, 120, 240], keypoints=[] if head is None else [[head, 20]])
    db.execute('UPDATE body_observations SET geometry=? WHERE id=?', (encoded(geometry), identifier))
    db.commit()
    assert json.loads(choose(evidence)['metadata'])['selection_category'] == 'face'


@pytest.mark.parametrize('ratio', [None, .8])
def test_body_ranking_prefers_centered_and_keeps_reserve_if_alone(tmp_path, ratio):
    from stash_ai.recognition.body_portraits import choose
    from stash_ai.recognition.portrait_refresh import retained_targets
    config, _, ids, people = seeded(tmp_path, vectors=[[1, 0]])
    config.data['portraits'].update(kind='person', aspect_ratio=ratio)
    with FaceDatabase(config.database) as state:
        db = state.connection
        crop(db, ids[0])
        side, center, unknown = ids[0]
        for oid, x, sharpness in [(side, 5, 1000), (center, 60, 0), (unknown, None, 1000)]:
            geometry = dict(bounds=[0, 0, 120, 240], keypoints=[] if x is None else [[x, 20]])
            db.execute("UPDATE body_observations SET geometry=?,quality=json_set(quality,'$.quality.sharpness',?) WHERE id=?",
                       (encoded(geometry), sharpness, oid))
        db.commit()
        target = retained_targets(state, config, 'scope')[people[0]]
        assert json.loads(choose(state, config, 'scope', target)['metadata'])['observation'] == center
        db.execute('DELETE FROM body_members WHERE observation!=?', (side,))
        db.commit()
        metadata = json.loads(choose(state, config, 'scope', target)['metadata'])
        assert metadata['observation'] == side and metadata['head_framing']['fallback']


def test_published_reserve_upgrades_to_face_and_does_not_downgrade(evidence):
    side_pair(evidence, index=0)
    state, config, client, target, ids = evidence
    config.data['portraits']['margin'] = 0
    db = state.connection
    members = db.execute('SELECT * FROM members WHERE observation!=?', (ids[0],)).fetchall()
    db.execute('DELETE FROM members WHERE observation!=?', (ids[0],))
    db.commit()
    def refresh():
        return fill_portrait(state, client, None, config, 'scope', target, {}, forbidden, forbidden,
                             visibility=Visibility(), expression=object())
    assert refresh()['portraits_filled'] == 1
    reserve_image = client.images['900']
    db.executemany('INSERT INTO members VALUES ('+','.join('?' for _ in members[0])+')', members)
    db.commit()
    assert refresh()['portraits_updated'] == 1
    assert client.images['900'] != reserve_image
    assert refresh()['portraits_preserved'] == 1
    client.images['900'] = b'manual replacement'
    assert refresh()['portraits_preserved'] == 1
    assert client.images['900'] == b'manual replacement'
