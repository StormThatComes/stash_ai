import json
import math
import sqlite3

import numpy as np
import pytest

from stash_ai.recognition import inspection_track_strategies as tracks
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.inspection_merge_paths import analyse
from stash_ai.recognition.inspection_pair_replay import apply_controls, describe, experiment
from stash_ai.recognition.scene_inspection import SceneReader
from tests.test_recognition_body_identity import configuration
from tests.test_recognition_inspection_merge_paths import face_scene
from tests.test_recognition_merge_report import seeded


def test_eight_routes_and_private_anchor_ablation(tmp_path):
    reader, scan, ids = face_scene(tmp_path)
    before = reader.config.database.read_bytes()
    report = analyse(reader, scan, ids)
    routes = {r['strategy']: r for r in report['routes'] if 'strategy' in r}
    assert set(routes) == {
        'turn',
        'crossing',
        'birth_recovery',
        'anchor_influence',
        'view_bridge',
        'local_global',
        'gap',
        'rival_control',
    }
    (ablate,) = routes['anchor_influence']['patches']
    assert not ablate['yaml']
    result = experiment(reader.config, scan['scope'], ids, {'diagnostic': ablate}, report['revision'])
    for key in ablate['omit_face_anchors']:
        assert result['trials']['baseline']['decisions'][key]['role'] == 'anchor'
        assert result['trials']['diagnostic']['decisions'][key]['role'] != 'anchor'
    assert 'Scene 1 cue' in describe(result, 'diagnostic')
    assert reader.config.database.read_bytes() == before
    birth = routes['birth_recovery']['patches']
    assert birth and all(p['birth_source'] in ids for p in birth)
    for patch in birth:
        assert any('anchor' in c['parameter'] for c in patch['changes'])
        assert any('threshold' in c['parameter'] for c in patch['changes'])


def test_automatic_controls_share_one_rebuild_but_keep_separate_results(tmp_path):
    config, ids = seeded(tmp_path, [[1, 0, 0, 0], [0.7, math.sqrt(0.51), 0, 0], [0, 0, 1, 0]])
    reader = SceneReader(config, '1')
    scan = reader.catalogue()['scan']
    report = analyse(reader, scan, ids[:2])
    strict = next(r for r in report['routes'] if r.get('family') == 'strict_core')
    option = strict['patches'][0]
    assert option['controls'] == [ids[2]]
    with pytest.raises(ValueError, match='every current scene'):
        experiment(config, 'scope', ids[:2], {'option': option}, report['revision'], scenes=['1', '2'])
    result = experiment(
        config, 'scope', ids[:2], {'auto': option, 'plain': dict(option, controls=[])}, report['revision']
    )
    auto, plain = (result['trials'][k] for k in ('auto', 'plain'))
    assert auto['owners'] is plain['owners']
    assert auto['controls'] == {ids[2]: True}
    assert not plain.get('controls')
    assert not apply_controls(dict(auto, control_results={ids[2]: False}), option)['success']


def row(key, cue, body=(1.0, 0.0), owner=None, face=None, orientation='front', shot=0, x=0):
    return dict(
        id=key,
        cue=cue,
        face_person=owner,
        face_anchor=owner is not None,
        face=None if face is None else 'face-' + key,
        _face_vector=np.array(face, 'f') if face is not None else None,
        _vector=np.array(body, 'f'),
        geometry=json.dumps(dict(bounds=[x, 0, x + 35, 100])),
        quality=json.dumps(
            dict(quality=dict(frame_signature='00' * 48, view_shot=shot, orientation=orientation))
        ),
    )


@pytest.mark.parametrize('cut', [False, True])
def test_turn_and_unassigned_bridge_obey_visual_shots(tmp_path, monkeypatch, cut):
    config = configuration(tmp_path)
    rows = [
        row('a', 0, owner='A', face=(1, 0)),
        row('missing', 1, body=(0.7, math.sqrt(0.51)), orientation='back'),
        row('b', 2, owner='B', face=(1, 0), shot=int(cut)),
    ]
    with sqlite3.connect(':memory:') as db:
        db.executescript(
            'CREATE TABLE body_members(observation,person,policy,status);'
            'CREATE TABLE body_observations(id,scan);'
        )
        for key, owner in [('a', 'A'), ('b', 'B')]:
            db.execute(
                'INSERT INTO body_members VALUES (?,?,?,?)', (key, owner, tracks.policy(config), 'accepted')
            )
            db.execute('INSERT INTO body_observations VALUES (?,?)', (key, 'scan'))
        monkeypatch.setattr(
            tracks,
            '_load_scene',
            lambda *_: (dict(occupied_cues=[], gap_boxes={}, motion=''), rows, {}, {}, []),
        )
        turn, _, gap = tracks.track_strategies(db, config, {'id': 'scan'}, ('A', 'B'))
    if cut:
        assert not turn['patches'] and not gap['patches']
    else:
        for route in (turn, gap):
            (patch,) = route['patches']
            assert patch['watch'] == ['missing']
            assert any(c['parameter'] == tracks.TRACK and c['proposed'] <= 0.7 for c in patch['changes'])
            assert any('First similarity blocker at cue 1' in n for n in route['notes'])


def test_contact_challenges_both_endpoint_permutations(tmp_path):
    config = configuration(tmp_path)
    rows = [
        row('before-A', 0, owner='A', face=(1, 0)),
        row('before-C', 0, (0, 1), 'C', (0, 1), x=55),
        row('hidden', 1),
        row('after-C', 2, (0, 1), 'C', (0, 1), x=55),
        row('after-B', 2, owner='B', face=(1, 0)),
    ]
    result = tracks._crossing(config, ('A', 'B'), rows, tracks.Scores(config))
    assert result['patches'][0]['controls'] == ['C']
    assert result['patches'][0]['watch'] == ['hidden']
    assert any('Keep the partner: weakest endpoint face 1.0000' in n for n in result['notes'])
    assert any('Swap with partner: weakest endpoint face 0.0000' in n for n in result['notes'])
    rows[-1]['_face_vector'] = rows[3]['_face_vector']
    assert (
        tracks._crossing(config, ('A', 'B'), rows, tracks.Scores(config))['status'] == 'Ambiguous endpoints'
    )


def test_intermediate_view_checks_both_legs_and_direct_core(tmp_path):
    from stash_ai.recognition.clustering import Group
    from stash_ai.recognition.global_views import IdentityView
    from stash_ai.recognition.inspection_catalogue_strategies import view_bridges

    config, ids = seeded(tmp_path)
    bank = []
    for owner, angle, pose in zip([*ids, 'C'], [0, 40, 20], ['left', 'right', 'neutral']):
        vector = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))], 'f')
        bank.append(IdentityView(Group(owner, (vector,), True), pose, pose, (pose,), 3, (vector,)))
    with FaceDatabase(config.database) as state:
        route, rivals = view_bridges(state.connection, config, ids, bank, None, bank)
    assert rivals == ['C']
    assert [w['label'] for w in route['witnesses']] == ['A / intermediate', 'Intermediate / B', 'A / B']
    assert any('not accepted' in n for n in route['notes'])
    assert route['patches'][0]['controls'] == ['C']


def test_local_global_variants_keep_global_similarity_floors(tmp_path):
    from stash_ai.recognition.inspection_merge_strategies import local_global
    from stash_ai.recognition.inspection_pair_variants import patch_values

    config = configuration(tmp_path)
    report = dict(
        identities=[
            dict(reference_samples=['a'], person_members=1),
            dict(reference_samples=['b'], person_members=1),
        ]
    )
    route = local_global(config, report, [])
    body_first, face_first = route['patches']
    assert patch_values(body_first)[tracks.LOCAL + 'body_first_matching']
    assert patch_values(face_first)['sapiensid.global_matching.enabled']
    assert all('threshold' not in key for p in route['patches'] for key in patch_values(p))


def test_private_body_ablation_uses_current_core_and_preserves_members(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    from tests.test_recognition_body_identity import scene

    config = configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        selected, _ = scene(db, config, vectors=[[1, 0]] * 3 + [[0.7, math.sqrt(0.51)]] * 4)
        db.execute(
            'UPDATE scans SET namespace=?,policy_sha=?',
            (namespace(config), scan_identity(config, namespace(config))),
        )
        db.commit()
        rebuild(state, [selected], config, {selected['scan']: []})
        ids = [
            r[0]
            for r in db.execute(
                "SELECT DISTINCT person FROM body_members WHERE status='accepted' ORDER BY person"
            )
        ]
    reader = SceneReader(config, '1')
    scan = reader.catalogue()['scan']
    before = config.database.read_bytes()
    report = analyse(reader, scan, ids)
    influence = next(r for r in report['routes'] if r.get('strategy') == 'anchor_influence')
    patch = influence['patches'][0]
    (first,) = patch['omit_body_anchors']
    result = experiment(config, scan['scope'], ids, {'option': patch}, report['revision'])
    assert result['trials']['baseline']['decisions'][first]['role'] == 'anchor'
    assert result['trials']['option']['decisions'].get(first, {}).get('role') != 'anchor'
    assert config.database.read_bytes() == before
