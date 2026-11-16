from dataclasses import replace
import math

import pytest
import yaml

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.inspection_merge_paths import analyse, verify
from stash_ai.recognition.scene_inspection import SceneReader
from tests.test_recognition_merge_report import seeded


def face_scene(tmp_path):
    config, ids = seeded(tmp_path)
    config = replace(config, document_version=2)
    with FaceDatabase(config.database) as state:
        db = state.connection
        scan = db.execute("SELECT id FROM scans WHERE scene='1'").fetchone()[0]
        db.execute("UPDATE observations SET scan=?, cue=cue+3,seconds=seconds+3, "
                   "quality=json_set(quality,'$.sharpness',55) WHERE scan<>?", (scan,scan))
        db.execute("UPDATE candidates SET scan=?", (scan,))
        db.execute("UPDATE people SET performer=NULL WHERE id=?", (ids[1],))
        db.commit()
    reader = SceneReader(config, '1')
    return reader, reader.catalogue()['scan'], ids


def test_pair_shows_measured_and_frozen_routes_and_public_yaml(tmp_path):
    reader, scan, ids = face_scene(tmp_path)
    before = reader.config.database.read_bytes()
    result = analyse(reader,scan,ids)
    routes = result['routes']
    strict = next(r for r in routes if r['title']=='Merge strict face anchors')
    assert strict['status'] == 'Frozen planner verified'
    assert yaml.safe_load(strict['patches'][0]['yaml'])['matching']['face']['match_threshold'] <= .7
    assert any('minimum cross similarity' in line for line in strict['notes'])
    assert any(r['title'].startswith('Merge by independent face nominations') for r in routes)
    forward, reverse = [r for r in routes if r['title'].startswith('Prevent a separate face core')]
    assert ids[1] in forward['title'] and 'Performer 900' in forward['title']
    change, = forward['patches'][0]['changes']
    assert (change['yaml_parameter'], change['proposed']) == ('matching.face.anchors.quality.min_sharpness',30.000001)
    assert not reverse['patches']
    assert reader.config.database.read_bytes() == before


def test_pair_checks_the_current_catalogue_and_captured_scan(tmp_path):
    reader, scan, ids = face_scene(tmp_path)
    with pytest.raises(ValueError, match='different identities'):
        analyse(reader, scan, (ids[0],ids[0]))
    with pytest.raises(ValueError, match='changed'):
        analyse(reader, scan, (ids[0],'missing'))
    with pytest.raises(ValueError, match='changed'):
        analyse(reader, dict(scan, id='old'), ids)


def test_face_support_uses_a_recorded_seed_and_independent_votes(tmp_path):
    reader, scan, ids = face_scene(tmp_path)
    from stash_ai.recognition.inspection_anchor_paths import alternatives
    with FaceDatabase(reader.config.database) as state:
        db = state.connection
        db.execute("""UPDATE candidates SET payload=json_set(payload,'$.face_comparison.stage',
            CASE WHEN observation_row.cue IN (0,3) THEN 'seed' ELSE 'match' END)
            FROM observations observation_row WHERE observation_row.id=json_extract(payload,'$.observation')""")
        db.execute('UPDATE observations SET seconds=0 WHERE cue=1')
        routes = alternatives(db,reader.config,scan,ids,dict(zip(ids,('A','B'))))
        patches = [p for r in routes for p in r['patches'] if p['title'].startswith('Require more')]
        assert len(patches) == 1
        assert patches[0]['changes'][0]['proposed'] == 3
        db.execute("UPDATE candidates SET policy='historical' WHERE person=?", (ids[0],))
        assert not alternatives(db,reader.config,scan,ids,dict(zip(ids,('A','B'))))


def test_body_only_pair_explains_local_and_global_and_preserves_copresence_veto(tmp_path):
    from tests.test_recognition_body_identity import configuration, scene
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.body_identity import rebuild
    config = configuration(tmp_path)
    config.data['sapiensid']['global_matching']['enabled'] = True
    with FaceDatabase(config.database) as state:
        db = state.connection
        selected, samples = scene(db,config,vectors=[[1,0]]*3+[[.7, math.sqrt(.51)]]*4)
        db.execute('UPDATE scans SET namespace=?,policy_sha=?',(namespace(config),scan_identity(config,namespace(config))))
        db.commit()
        rebuild(state,[selected],config,{selected['scan']:[]})
        ids = [r[0] for r in db.execute("SELECT DISTINCT person FROM body_members WHERE status='accepted' ORDER BY person")]
        assert len(ids) == 2
    reader = SceneReader(config,'1')
    scan = reader.catalogue()['scan']
    result = analyse(reader,scan,ids)
    local = next(r for r in result['routes'] if r['title']=='Form one local body core')
    assert local['status']=='Rebuild hypothesis'
    assert any(c['parameter']=='sapiensid.identities.match_threshold' for c in local['patches'][0]['changes'])
    assert any(p['changes'][0]['parameter']=='sapiensid.identities.min_support'
               for r in result['routes'] if r['title'].startswith('Prevent a separate body') for p in r['patches'])
    with FaceDatabase(config.database) as state:
        state.connection.execute('UPDATE body_observations SET cue=0 WHERE id=?',(samples[-1],))
        state.connection.commit()
    again = analyse(reader,scan,ids)
    local = next(r for r in again['routes'] if r['title']=='Form one local body core')
    assert local['status']=='Blocked' and not local['patches']
    global_route = next(r for r in again['routes'] if r['title']=='Merge independent body identities')
    assert not global_route['patches'] and any('person_copresence' in n for n in global_route['notes'])


def test_replay_checks_ownership_and_rejects_a_stale_option(tmp_path):
    reader, scan, ids = face_scene(tmp_path)
    report = analyse(reader,scan,ids)
    option = report['routes'][0]['patches'][0]
    before = reader.config.database.read_bytes()
    message = verify(reader.config,scan['scope'],ids,option,report['revision'])
    assert 'Replay on retained scans' in message and 'unassigned' in message
    assert reader.config.database.read_bytes() == before
    with FaceDatabase(reader.config.database) as state:
        state.connection.execute('UPDATE people SET performer=? WHERE id=?',('7',ids[1]))
        state.connection.commit()
    with pytest.raises(ValueError, match='changed'):
        verify(reader.config,scan['scope'],ids,option,report['revision'])


def test_replay_names_an_outside_identity_and_ignores_previously_unassigned(tmp_path, monkeypatch):
    from stash_ai.recognition import replay_simulation
    reader, scan, ids = face_scene(tmp_path)
    retained = dict(a=ids[0], b=ids[1], c='outside', d=None)
    def replay(config, variants, *, scope, source, consume):
        baseline = consume('baseline', retained.keys(), retained, dict.fromkeys(retained, 'joined'))
        return dict(selection=[scan['scene']], trials=dict(baseline=baseline))
    monkeypatch.setattr(replay_simulation, 'replay', replay)
    report = dict(scope=scan['scope'], identity_ids=ids, recommendations=[dict(changes=[])])
    with FaceDatabase(reader.config.database) as state:
        result = replay_simulation.merge_impact(reader.config, report, state.connection)
    assert result['trials']['baseline']['requested_pair']['additional_identity_ids'] == ['outside']


def test_catalogue_budget_keeps_measured_scene_routes_available(tmp_path, monkeypatch):
    from stash_ai.recognition import merge_report
    from stash_ai.recognition.resource_errors import ResourceLimit
    reader, scan, ids = face_scene(tmp_path)
    def exhausted(*args, **kwargs):
        raise ResourceLimit('Configured reference budget exhausted.')
    monkeypatch.setattr(merge_report,'analyse_pair',exhausted)
    result = analyse(reader,scan,ids)
    assert 'Configured reference budget exhausted' in result['summary']
    assert any(r['patches'] for r in result['routes'])


def test_face_and_body_core_prevention_combines_gates_for_the_same_direction(tmp_path):
    import json
    from stash_ai.recognition.body_identity import policy as body_policy
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.clustering import matching_policy
    from stash_ai.recognition.inspection_anchor_paths import alternatives
    from tests.test_recognition_body_identity import configuration
    reader, scan, ids = face_scene(tmp_path)
    config = replace(reader.config, data={**reader.config.data, 'sapiensid': configuration(tmp_path).data['sapiensid']})
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('UPDATE candidates SET policy=?', (matching_policy(config),))
        for cue, person in enumerate(ids):
            address = 'body-' + person
            quality = json.dumps({'quality': {'keypoints':12, 'sharpness':100+50*cue, 'width':100, 'height':240}})
            db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (address, scan['id'], cue, float(cue), contract(config), None, '{}', quality, None, bytes(16384), None))
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
                       (address, person, 'anchor','accepted',body_policy(config),'{"reason":"body_scene_core"}'))
        routes = alternatives(db,config,scan,ids,dict(zip(ids,('A','B'))))
    combined, = [r for r in routes if r['title'].startswith('Prevent both')]
    assert 'A → B' in combined['title']
    patch = yaml.safe_load(combined['patches'][0]['yaml'])
    assert patch['matching']['face']['anchors']['quality']['min_sharpness'] > 30
    assert any(c['parameter']=='sapiensid.identities.anchor_min_sharpness' for c in combined['patches'][0]['changes'])
