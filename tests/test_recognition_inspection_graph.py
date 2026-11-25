import json
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_inspection import IdentityReader
from stash_ai.recognition.inspection_graph import build, key
from tests.test_recognition_identity_report import seed


def test_graph_records_routes_not_anchor_label_authority(tmp_path):
    config, person, _ = seed(tmp_path, 4)
    with FaceDatabase(config.database) as state:
        db = state.connection
        candidates = db.execute('SELECT * FROM candidates ORDER BY id').fetchall()
        source = json.loads(candidates[1]['payload'])['observation']
        row = candidates[0]; payload = json.loads(row['payload']); target=payload['observation']
        payload.update(reason='anchor_view_identity',anchor_view=dict(medoid=source,support_ids=[source,'missing'],score=.8,threshold=.7))
        db.execute('UPDATE candidates SET payload=? WHERE id=?',(json.dumps(payload),row['id'])); db.commit()
    before = config.database.read_bytes()
    graph = build(IdentityReader(config,identity_id=person))
    nodes = {n['id']:n for n in graph['nodes']}
    assert nodes[key('face',target)]['state']=='anchor'
    assert nodes[key('face',target)]['authority']=='attachment_only'
    assert nodes['face:missing']['availability']=='unavailable_in_scope'
    assert any(n['type']=='view' for n in nodes.values())
    assert any(e['relation']=='view_support' for e in graph['edges'])
    assert all(e['source'] in nodes and e['target'] in nodes for e in graph['edges'])
    assert 'embedding' not in json.dumps(graph)
    assert config.database.read_bytes()==before


def test_graph_paging_filters_and_unknown_provenance(tmp_path):
    config, person, _ = seed(tmp_path, 6)
    reader = IdentityReader(config, identity_id=person)
    first, second = build(reader,limit=2), build(reader,limit=2,offset=2)
    ids = lambda g:{n['id'] for n in g['nodes'] if n.get('current_member')}
    assert first['total']==6 and first['more'] and ids(first).isdisjoint(ids(second))
    assert all(n.get('scene')=='2' for n in build(reader,scene='2')['nodes'] if n.get('current_member'))
    with FaceDatabase(config.database) as state:
        state.connection.execute('DELETE FROM candidates'); state.connection.commit()
    assert all(n['authority']=='unknown' for n in build(reader)['nodes'])
