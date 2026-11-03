from io import BytesIO
from itertools import permutations
import json
import uuid

import numpy as np
from PIL import Image
import pytest
import yaml

from stash_ai.recognition.body_portraits import choose_kind
from stash_ai.recognition.clustering import cluster_scene, matching_policy
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.person_signal import contract, save
from stash_ai.recognition.portraits import policy_identity, fill_portrait
from tests.helpers.recognition import Stash, Visibility
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_rebuild import changed


@pytest.mark.parametrize('order', list(permutations(['face+pose', 'face', 'pose'])))
def test_yaml_accepts_every_complete_precedence_order(tmp_path, order):
    path = tmp_path / 'order.yml'
    path.write_text(yaml.safe_dump({'version': 1, 'portraits': {'auto_precedence': list(order)}}))
    config = load(path)
    assert config.data['portraits']['auto_precedence'] == order
    original = load(root=tmp_path)
    assert original.data['portraits']['auto_precedence'] == ('face+pose', 'face', 'pose')
    assert matching_policy(config) == matching_policy(original)
    assert (policy_identity(config) == policy_identity(original)) == (order == ('face+pose', 'face', 'pose'))


@pytest.mark.parametrize('order', [None, 'face+pose,face,pose', [], ['face'],
    ['face', 'pose', 'face'], ['face+pose', 'face', 'person'], ['face+pose', 'face', {}],
    ['face+pose', 'face', True], ['face+pose', 'face', 'pose', 'pose']])
def test_yaml_rejects_invalid_precedence_before_opening_database(tmp_path, order):
    path = tmp_path / 'bad.yml'
    path.write_text(yaml.safe_dump({'version': 1, 'portraits': {'auto_precedence': order}}))
    with pytest.raises(ValueError, match='portraits.auto_precedence'):
        load(path)
    assert not (tmp_path / 'faces_state.sqlite3').exists()


@pytest.fixture
def evidence(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        scan_id, _ = scan(state, config, '1')
        person = cluster_scene(state, scan_id, config)[0]['person']
        config = changed(config, sapiensid={'enabled': True}, portraits={'min_visible_face_fraction': 0})
        config.data['portraits']['weights']['visibility'] = 0
        db = state.connection
        rows = db.execute('SELECT id,quality FROM observations ORDER BY cue').fetchall()
        ids = [r['id'] for r in rows]
        for index, row in enumerate(rows):
            quality = json.loads(row['quality'])
            # The paired sample is deliberately worse than the unpaired faces.
            quality.update(sharpness=0. if index == 0 else 1000., pixels=32. if index == 0 else 512.)
            db.execute('UPDATE observations SET quality=? WHERE id=?', (encoded(quality), row['id']))
        save(db, ids[0], contract(config), {'person': {'bounds': [0, 0, 160, 160]},
                                          'identity_rejection': None}, np.ones(4096))
        db.execute('UPDATE people SET performer=? WHERE id=?', ('900', person))
        db.commit()
        client = Stash()
        token = db.execute('SELECT token FROM people WHERE id=?', (person,)).fetchone()[0]
        client.profiles['900'] = dict(id='900', name='Test', details=token, image_path='')
        target = dict(person=person, performer='900', scans=[dict(scene='1', scan=scan_id)])
        yield state, config, client, target, ids


def forbidden(*args, **kwargs):
    pytest.fail('Retained tiers must not load face/pose models or decode video')


def choose(evidence, source=SpriteInput):
    state, config, client, target, _ = evidence
    return choose_kind(state, client, config, 'scope', target, {}, forbidden, forbidden, source,
                       visibility=Visibility(), expression=object())


def add_body(evidence):
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.body_observations import contract as body_contract
    state, config, _, target, _ = evidence
    config.data['sapiensid']['collection']['enabled'] = True
    config.data['sapiensid']['identities']['enabled'] = True
    identifier = str(uuid.uuid4())
    with Image.new('RGB', (120, 240), (20, 90, 160)) as image:
        output = BytesIO()
        image.save(output, 'PNG')
    db = state.connection
    db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (identifier, target['scans'][0]['scan'], 0, 0., body_contract(config), None,
         encoded(dict(bounds=[0, 0, 120, 240], keypoints=[[-1, -1]] * 17)),
         encoded(dict(quality=dict(sharpness=1000, keypoints=17, width=120, height=240))),
         None, np.ones(4096, dtype='<f4').tobytes(), output.getvalue()))
    db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
               (identifier, target['person'], 'anchor', 'accepted', policy(config), '{}'))
    db.commit()
    return identifier


def test_default_tier_beats_quality_score_and_stops_before_lower_tiers(evidence):
    state, config, _, _, ids = evidence
    add_body(evidence)
    winner = choose(evidence)
    metadata = json.loads(winner['metadata'])
    assert metadata['observation'] == ids[0]
    assert metadata['selection_category'] == 'face+pose'
    assert metadata['quality_shortfalls']
    assert state.connection.execute('SELECT COUNT(*) FROM portraits').fetchone()[0] == 1


def test_quality_still_selects_the_best_candidate_within_paired_tier(evidence):
    state, config, _, _, ids = evidence
    save(state.connection, ids[2], contract(config), {'person': {'bounds': [0, 0, 160, 160]}}, np.ones(4096))
    state.connection.commit()
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['observation'] == ids[2] and metadata['selection_category'] == 'face+pose'


def test_quality_shortfalls_cannot_promote_pose_above_face(evidence):
    state, config, _, _, _ = evidence
    add_body(evidence)
    state.connection.execute("UPDATE person_evidence SET status='unavailable',embedding=NULL")
    state.connection.commit()
    config.data['portraits']['min_face_pixels'] = 4096
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['quality_shortfalls'] and metadata['selection_category'] == 'face'


@pytest.mark.parametrize('order,expected', [
    (['face', 'face+pose', 'pose'], 'face'),
    (['pose', 'face', 'face+pose'], 'pose'),
])
def test_custom_order_selects_first_usable_tier(evidence, order, expected):
    add_body(evidence)
    evidence[1].data['portraits']['auto_precedence'] = order
    winner = choose(evidence)
    assert json.loads(winner['metadata'])['selection_category'] == expected
    categories = {json.loads(row[0])['selection_category']
                  for row in evidence[0].connection.execute('SELECT metadata FROM portraits')}
    assert categories == {expected}


@pytest.mark.parametrize('kind,expected', [('face', 'face'), ('person', 'pose')])
def test_explicit_kind_ignores_auto_precedence(evidence, kind, expected):
    add_body(evidence)
    evidence[1].data['portraits'].update(kind=kind, auto_precedence=['pose', 'face+pose', 'face'])
    assert json.loads(choose(evidence)['metadata'])['selection_category'] == expected


@pytest.mark.parametrize('mutation', ['disabled', 'unavailable', 'stale', 'rejected', 'missing_person'])
def test_unusable_paired_evidence_falls_back_to_face(evidence, mutation):
    state, config, _, _, ids = evidence
    if mutation == 'disabled':
        config.data['sapiensid']['enabled'] = False
    elif mutation == 'unavailable':
        state.connection.execute("UPDATE person_evidence SET status='unavailable',embedding=NULL")
    elif mutation == 'stale':
        state.connection.execute("UPDATE person_evidence SET contract='obsolete'")
    elif mutation == 'rejected':
        state.connection.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.identity_rejection','person_overlap')")
    else:
        state.connection.execute("UPDATE person_evidence SET metadata=json_remove(metadata,'$.person')")
    state.connection.commit()
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['selection_category'] == 'face' and metadata['observation'] != ids[0]


def test_retained_paired_crop_beats_live_lower_tier(evidence):
    state, config, _, _, ids = evidence
    with Image.new('RGB', (160, 160), (20, 90, 160)) as image:
        output = BytesIO()
        image.save(output, 'PNG')
    state.connection.execute('UPDATE person_evidence SET crop=? WHERE observation=?', (output.getvalue(), ids[0]))
    state.connection.commit()
    class NoPairedThumbnail(SpriteInput):
        def thumbnail(self, index):
            if index == 0:
                raise OSError('paired source unavailable')
            return super().thumbnail(index)
    metadata = json.loads(choose(evidence, NoPairedThumbnail)['metadata'])
    assert metadata['selection_category'] == 'face+pose'
    assert metadata['source_type'] == 'person_crop'


def test_unencodable_paired_candidate_does_not_hide_face_fallback(evidence):
    state, _, _, _, ids = evidence
    state.connection.execute('UPDATE observations SET crop=NULL WHERE id=?', (ids[0],))
    state.connection.commit()
    class NoPairedThumbnail(SpriteInput):
        def thumbnail(self, index):
            if index == 0:
                raise OSError('paired source unavailable')
            return super().thumbnail(index)
    metadata = json.loads(choose(evidence, NoPairedThumbnail)['metadata'])
    assert metadata['selection_category'] == 'face' and metadata['observation'] != ids[0]


def test_anchor_gate_can_fall_through_to_body_and_body_gate_is_preserved(evidence):
    state, config, _, _, _ = evidence
    config.data['portraits']['anchor_only'] = True
    config.data['clustering']['max_anchor_pose_deviation'] = .2
    add_body(evidence)
    state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.9)")
    state.connection.commit()
    assert json.loads(choose(evidence)['metadata'])['selection_category'] == 'pose'
    state.connection.execute("UPDATE body_members SET role='auxiliary'")
    state.connection.commit()
    assert choose(evidence) is None


def test_pairing_changes_invalidate_completed_rankings(evidence, monkeypatch):
    from stash_ai.recognition import vtt_portraits
    state, config, _, _, ids = evidence
    assert json.loads(choose(evidence)['metadata'])['observation'] == ids[0]
    original = vtt_portraits.observation_identity
    monkeypatch.setattr(vtt_portraits, 'observation_identity', forbidden)
    assert json.loads(choose(evidence)['metadata'])['observation'] == ids[0]
    monkeypatch.setattr(vtt_portraits, 'observation_identity', original)
    db = state.connection
    db.execute("UPDATE person_evidence SET status='unavailable',embedding=NULL WHERE observation=?", (ids[0],))
    save(db, ids[2], contract(config), {'person': {'bounds': [0, 0, 160, 160]}}, np.ones(4096))
    db.commit()
    assert json.loads(choose(evidence)['metadata'])['observation'] == ids[2]
    db.execute("UPDATE person_evidence SET status='unavailable',embedding=NULL")
    db.commit()
    assert json.loads(choose(evidence)['metadata'])['selection_category'] == 'face'


def test_prior_pairing_is_rechecked_before_preserving_old_portrait(evidence):
    from stash_ai.recognition.portraits import _current_prior
    state, config, client, target, _ = evidence
    old = choose(evidence)
    state.connection.execute("UPDATE person_evidence SET status='unavailable',embedding=NULL")
    state.connection.commit()
    refreshed = _current_prior(state, old, client, config, 'scope', target, Visibility(), object())
    assert refreshed is None  # A pose crop cannot become a face crop by relabelling.


def test_paired_tier_does_not_use_video_hint_as_pose_proof(evidence):
    evidence[1].data['portraits']['source'] = 'video'
    metadata = json.loads(choose(evidence)['metadata'])
    assert metadata['selection_category'] == 'face+pose' and metadata['source_type'] == 'vtt'


def test_refresh_changes_order_without_rebuilding_and_preserves_manual_image(evidence):
    state, config, client, target, _ = evidence
    add_body(evidence)
    def refresh():
        return fill_portrait(state, client, None, config, 'scope', target, {}, forbidden, forbidden,
                             visibility=Visibility(), expression=object())
    assert refresh()['portraits_filled'] == 1
    original = client.images['900']
    memberships = [tuple(r) for r in state.connection.execute('SELECT * FROM members')]
    config.data['portraits']['auto_precedence'] = ['pose', 'face+pose', 'face']
    assert refresh()['portraits_updated'] == 1
    assert client.images['900'] != original
    assert [tuple(r) for r in state.connection.execute('SELECT * FROM members')] == memberships
    client.images['900'] = b'manual replacement'
    config.data['portraits']['auto_precedence'] = ['face+pose', 'face', 'pose']
    assert refresh()['portraits_preserved'] == 1
    assert client.images['900'] == b'manual replacement'
