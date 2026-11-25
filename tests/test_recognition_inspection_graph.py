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


def test_layout_is_finite_repeatable_and_respects_pins():
    import math
    from stash_ai.recognition.inspection_graph_layout import initial, step, project
    p=initial(['a','b','c'])
    assert p==initial(['a','b','c'])
    edges=[dict(source='a',target='b',relation='reference')]
    for _ in range(40):p,_=step(p,edges,{'a'})
    assert p['a']==initial(['a'])['a']
    assert all(math.isfinite(v) for xy in p.values() for v in xy)
    graph=dict(nodes=[dict(id='a',type='observation',current_member=True,scene='1'),
                      dict(id='b',type='decision')],edges=[dict(source='b',target='a',relation='assigned')])
    compact,groups=project(graph,set())
    expanded,_=project(graph,groups)
    assert len(compact['nodes'])==1 and len(expanded['nodes'])==2
    assert expanded['edges'][0]['source']=='b'


def test_diagnostics_exclude_rivals_and_new_similarities():
    from stash_ai.recognition.inspection_graph_diagnostics import related, bridges
    graph=dict(nodes=[dict(id=x) for x in 'abcde'],edges=[
        dict(source='a',target='b',relation='reference'),dict(source='b',target='c',relation='assigned'),
        dict(source='c',target='d',relation='reference'),dict(source='e',target='c',relation='rival'),
        dict(source='a',target='d',relation='similarity')])
    assert related(graph,'c')==set('abc')
    assert related(graph,'b',downstream=True)==set('bcd')
    assert bridges(graph)==[('b','c')]


def test_similarity_is_explicit_scoped_and_checks_revision(tmp_path):
    import numpy as np
    import pytest
    from stash_ai.recognition.inspection_graph_diagnostics import similarities
    config,person,_=seed(tmp_path,6)
    with FaceDatabase(config.database) as state:
        state.connection.execute('UPDATE observations SET embedding=?',(np.array([1,0,0,0],dtype='<f4').tobytes(),))
        state.connection.commit()
    reader=IdentityReader(config,identity_id=person);graph=build(reader)
    result=similarities(reader,graph)
    nodes={n['id']:n for n in graph['nodes']}
    assert result['edges'] and not result['missing']
    for e in result['edges']:
        a,b=nodes[e['source']],nodes[e['target']]
        assert a['kind']==b['kind']
        assert (a['scene'],a['cue'])!=(b['scene'],b['cue'])
        assert e['provenance']=='new_comparison' and e['score']==1
    with FaceDatabase(config.database) as state:
        state.connection.execute('DELETE FROM members');state.connection.commit()
    with pytest.raises(ValueError,match='changed'):similarities(reader,graph)


def test_multiple_recorded_views_are_not_fused_into_one(tmp_path):
    config,person,_=seed(tmp_path,2)
    with FaceDatabase(config.database) as state:
        row=state.connection.execute('SELECT * FROM candidates LIMIT 1').fetchone()
        payload=json.loads(row['payload']);payload.update(reason='global_view_identity',global_view=dict(
            view_evidence=[dict(medoid='a',support_ids=['x']),dict(medoid='b',support_ids=['y'])]))
        state.connection.execute('UPDATE candidates SET payload=? WHERE id=?',(json.dumps(payload),row['id']));state.connection.commit()
    graph=build(IdentityReader(config,identity_id=person))
    assert not any(n['type']=='view' for n in graph['nodes'])
    assert {'face:a','face:b'} <= {n['id'] for n in graph['nodes']}
