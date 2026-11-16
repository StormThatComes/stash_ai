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



def test_cross_scene_pair_resolves_by_performer_id_and_links_weakest_faces(tmp_path):
    from stash_ai.recognition.inspection_context import locate
    from stash_ai.recognition.inspection_pair_evidence import find_identity
    config, ids = seeded(tmp_path)
    reader = SceneReader(config, '1')
    scan = reader.catalogue()['scan']
    assert find_identity(reader, scan['scope'], '901')['identity'] == ids[1]
    with pytest.raises(ValueError):
        find_identity(reader, 'another-scope', ids[1])
    report = analyse(reader, scan, ids)
    strict = next(route for route in report['routes'] if route['title'] == 'Merge strict face anchors')
    assert strict['first_blocker'] == 'Similarity'
    assert strict['steps'][0]['status'] == 'Pass'
    weakest = next(w for w in strict['witnesses'] if w['label'] == 'Weakest cross comparison')
    scenes = {locate(config, kind, addr, scan['scope'])['scene'] for kind, addr in weakest['refs']}
    assert scenes == {'1', '2'}


def test_replay_keeps_protected_c_separate_and_observations_navigable(tmp_path):
    from stash_ai.recognition.inspection_pair_replay import experiment
    config, ids = seeded(tmp_path, [[1,0,0,0],[.7,math.sqrt(.51),0,0],[0,0,1,0]])
    reader = SceneReader(config,'1')
    scan = reader.catalogue()['scan']
    report = analyse(reader, scan, ids[:2])
    patch = report['routes'][0]['patches'][0]
    result = experiment(config,scan['scope'],ids,{'option':patch},report['revision'])
    trial = result['trials']['option']
    assert trial['protected'] and trial['counts'][ids[2]] == 3
    assert len(result['rows']) == 9
    assert all(row['kind']=='face' for row in result['rows'].values())
    assert all('role' in item for item in trial['decisions'].values())
    assert set(trial['owners']) == set(result['trials']['baseline']['owners'])


def test_protection_rejects_withheld_split_or_absorbed_c():
    from stash_ai.recognition.inspection_pair_replay import outcome
    retained = dict(a='A', b='B', c='C', d='C', e='D')
    rows = {k:dict(source=v) for k,v in retained.items() if v != 'D'}
    for c, d in ((None,'third'), ('third','fourth'), ('joined','joined')):
        result = outcome(('A','B','C'),rows,retained,dict(a='joined',b='joined',c=c,d=d,e='joined'))
        assert result['joined'] and not result['success'] and not result['protected']
        assert 'D' in result['outsiders']


def test_quick_preflight_includes_protected_scenes_and_full_counts_outside(tmp_path):
    from stash_ai.recognition.inspection_pair_variants import preflight
    from stash_ai.recognition.inspection_runtime import stamp
    config, ids = seeded(tmp_path,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    revision = stamp(config.database)
    quick = preflight(config,'scope',ids[:3],'1',False,revision)
    full = preflight(config,'scope',ids[:3],'1',True,revision)
    assert (quick['scenes'], quick['faces']) == (['1','2','3'],9)
    assert (full['scenes'], full['faces']) == (['1','2','3','4'],12)
    from stash_ai.recognition.inspection_pair_replay import experiment
    with pytest.raises(ValueError,match='every current scene'):
        experiment(config,'scope',ids[:3],{},revision,scenes=['1','2'])


def test_variants_share_baseline_and_minimized_yaml_reproduces_goal(tmp_path):
    from stash_ai.recognition.inspection_pair_replay import experiment
    from stash_ai.recognition.merge_report_policy import proposal
    from stash_ai.recognition.inspection_pair_variants import patch_values
    reader, scan, ids = face_scene(tmp_path)
    config = reader.config
    report = analyse(reader,scan,ids)
    patch = patch_values(report['routes'][0]['patches'][0])
    # An auxiliary-only gate cannot improve this fixture's strict anchor join.
    patch['clustering.auxiliary_min_support'] = 7
    option = proposal(config,patch)
    starts = []
    def observe(name, stage, item):
        if stage == 'start':
            starts.append(name)
    observe.telemetry_enabled = False
    result = experiment(config,'scope',ids,dict(option=option, duplicate=option, unchanged=proposal(config,{})),
                        report['revision'],progress=observe,refine_budget=12)
    assert starts.count('baseline') == 1 and 'duplicate' not in starts and 'unchanged' not in starts
    assert result['trials']['option'] is result['trials']['duplicate']
    assert result['probe_count'] <= 12 and result['trials']['minimized']['success']
    assert 'clustering.auxiliary_min_support' not in patch_values(result['minimized'])
    confirmed = experiment(config,'scope',ids,{'recheck':result['minimized']},report['revision'])
    assert confirmed['trials']['recheck']['success']
    assert confirmed['trials']['recheck']['owners'] == result['trials']['minimized']['owners']


def test_refinement_preserves_outsiders_and_coverage():
    from stash_ai.recognition.inspection_pair_variants import admissible
    initial = dict(success=True, outsiders=['D'],unassigned=2,coverage_lost=1)
    assert admissible(dict(initial,outsiders=[]), initial)
    for changed in (dict(success=False),dict(outsiders=['E']),dict(unassigned=3),dict(coverage_lost=2)):
        assert not admissible(dict(initial,**changed),initial)


def test_attribution_links_only_recorded_pair_evidence():
    from stash_ai.recognition.inspection_pair_replay import attribution
    ids = ('a','b')
    rows = {'x':dict(source='a',scene='2',cue=4),'y':dict(source='b',scene='5',cue=3)}
    seed = dict(route='candidate_group', stage='seed', role='anchor', links=[], reference_sources=[])
    matched = dict(seed, stage='match',reference_sources=['a'])
    result = dict(joined=True)
    initial = {'x':seed, 'y':seed}
    msg, key = attribution(ids,rows,initial,dict(x=seed,y=matched),result)
    assert 'Separate core avoided' in msg and key == 'y'
    msg, key = attribution(ids,rows,initial,dict(x=seed,y=dict(matched,links=[('b','a','body_global')])),result)
    assert 'Post-formation consolidation' in msg and key == 'y'
    _, key = attribution(ids,rows,initial,dict(x=seed,y=dict(matched,reference_sources=['c'])),result)
    assert key is None


def test_matrix_keeps_current_axes_and_formation_counts(tmp_path):
    from stash_ai.recognition.inspection_view_evidence import face_rows, formation
    reader,scan,ids=face_scene(tmp_path)
    matrix=analyse(reader,scan,ids)['matrix']
    assert len(matrix['axes'])==2
    for a in matrix['axes'][0]:
        for b in matrix['axes'][1]:
            assert any(p['medoids']==[a['id'],b['id']] for p in matrix['pairs'])
    rows=face_rows(reader,scan['scope'],ids[0])
    text=formation(reader,scan['scope'],ids[0],[r['id'] for r in rows])
    assert 'independent votes' in text and 'Independent witnesses:' in text
    with pytest.raises(ValueError,match='changed'):
        formation(reader,scan['scope'],ids[1],[rows[0]['id']])
