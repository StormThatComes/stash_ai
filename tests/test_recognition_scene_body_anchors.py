import math

import numpy as np
import pytest

from stash_ai.recognition.assignment_explanation import body as explain_body
from stash_ai.recognition.body_global_banks import _banks
from stash_ai.recognition.body_identity import rebuild, scene_plan
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.scene_artifacts import body_key
from tests.test_recognition_body_identity import configuration, scene


def body_scene(db, config, angles, owners, *, cues=None):
    vectors = [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles]
    item, ids = scene(db, config, vectors=vectors, times=cues)
    db.executemany('INSERT INTO people VALUES (?,?,?,?,?,?)',
                   [(p, 'scope', 'ns', 1, None, 'face-' + p) for p in sorted(set(owners) - {None})])
    for i, owner in enumerate(owners):
        if owner is None:
            continue
        cue = cues[i] if cues else i
        fid = 'face-' + ids[i]
        vector = np.array([1, 0] if owner == 'A' else [0, 1], dtype='<f4')
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (fid, item['scan'], cue, float(cue), '', '[]', '[]',
                    encoded(dict(pixels=120, sharpness=100, pose_deviation=0)), None, 2, vector.tobytes(), None))
        db.execute('INSERT INTO members VALUES (?,?)', (fid, owner))
        db.execute('UPDATE body_observations SET face=? WHERE id=?', (fid, ids[i]))
        route = 'same_scene_identity' if i == 2 else 'known_identity'
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                   (fid, item['scan'], 'policy', owner, 'accepted', encoded(dict(reason=route, observation=fid))))
    return item, ids


@pytest.mark.parametrize('tracklets', [False, True])
def test_same_scene_body_confirms_then_assists_a_new_cue_only_locally(tmp_path, tracklets):
    config = configuration(tmp_path)
    config.data['sapiensid']['identities'].update(match_threshold=.85, cohesion_threshold=.80, attach_min_support=2)
    config.data['sapiensid']['identities'].update(tracklet_matching=tracklets, body_first_matching=tracklets)
    with FaceDatabase(config.database) as state:
        db = state.connection
        item, ids = body_scene(db, config, [0, 0, 25, 35], ['A', 'A', 'A', None])
        db.commit()
        decisions = scene_plan(db, item['scan'], config)[2]
        anchor, attached = decisions[ids[2]], decisions[ids[3]]
        assert (anchor['role'], anchor['reference_scope'], anchor['scene_anchor_check']) == ('anchor', 'scene', 'confirmed')
        assert set(anchor['witness_ids']) == set(ids[:2])
        assert (attached['person'], attached['role']) == ('A', 'auxiliary')
        assert ids[2] in attached['witness_ids']
        trace = explain_body(anchor, linked_face='face-' + ids[2], role='anchor', status='accepted',
                             policy='policy', current=True, merges={}, target='A')['decisions'][0]
        assert trace['metrics']['scene_anchor_check'] == 'confirmed'
        assert trace['metrics']['reference_scope'] == 'scene'
        rebuild(state, [item], config, {})
        bank = _banks(db, config, 'scope', 'ns')['A']
        assert {r['id'] for r in bank['cores'][0]} == set(ids[:2])
        assert bank['support'] == 2

        before = body_key(db, item['scan'], config)
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','anchor_view_identity') WHERE id=?",
                   ('face-' + ids[2],))
        assert body_key(db, item['scan'], config) != before
        decisions = scene_plan(db, item['scan'], config)[2]
        assert decisions[ids[2]]['role'] == 'auxiliary'
        assert decisions[ids[3]]['person'] != 'A'


@pytest.mark.parametrize('case,blocker', [
    ('no_root', 'no_strict_body_references'),
    ('close_rival', 'competing_body_identity'),
    ('duplicate', 'insufficient_independent_body_support'),
    ('same_cue', 'same_cue_owner_conflict'),
    ('poor_match', 'body_match_or_cohesion'),
    ('query_only', 'body_query_only'),
    ('chain', 'body_match_or_cohesion'),
])
def test_context_face_alone_cannot_seed_or_chain_local_anchors(tmp_path, case, blocker):
    config = configuration(tmp_path)
    config.data['sapiensid']['identities'].update(match_threshold=.85, cohesion_threshold=.80, attach_min_support=2)
    with FaceDatabase(config.database) as state:
        db = state.connection
        owners = ['A', 'A', 'A', 'B', 'B']
        angles = [0, 0, 25, 160, 160]
        if case == 'no_root':
            owners[:2] = [None, None]
        elif case == 'close_rival':
            # The faces of A and B are orthogonal. That cannot discount the SapiensID rival.
            angles[3:] = [49, 49]
        elif case == 'poor_match':
            angles[2] = 85
        elif case == 'chain':
            owners.append('A')
            angles.append(35)
        item, ids = body_scene(db, config, angles, owners, cues=[0, 1, 1, 3, 4] if case == 'same_cue' else None)
        target = ids[2]
        if case == 'duplicate':
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.pixel_digest','1-0') WHERE id=?", (target,))
        elif case == 'query_only':
            db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.query_only',1) WHERE id=?", (target,))
        elif case == 'chain':
            target = ids[5]
            db.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','same_scene_identity') WHERE id=?",
                       ('face-' + target,))
        decisions = scene_plan(db, item['scan'], config)[2]
        assert (decisions[target]['person'], decisions[target]['role']) == ('A', 'auxiliary')
        assert decisions[target]['scene_anchor_check'] == blocker
        if case == 'chain':
            assert decisions[ids[2]]['scene_anchor_check'] == 'confirmed'
