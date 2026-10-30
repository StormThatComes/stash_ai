from contextlib import closing
from unittest.mock import patch

from stash_ai.recognition.fragmentation_report import _open_readonly


def test_body_metadata_reads_are_chunked_ordered_and_keep_missing_witnesses(tmp_path):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.body_global_banks import metadata_rows
    from stash_ai.recognition.body_observations import compatible_contracts
    config,_,ids,_ = seeded(tmp_path)
    requested = (ids[0]*100)+['missing']
    with closing(_open_readonly(config.database)) as db:
        queries=[];db.set_trace_callback(queries.append)
        rows=list(metadata_rows(db,requested,body=True,contracts=compatible_contracts(config)))
    assert [r['id'] if r else 'missing' for r in rows] == requested
    assert sum(q.startswith('SELECT b.id,b.scan') for q in queries) == 2
    assert all('embedding' not in row and 'crop' not in row for row in rows if row)


def test_owner_metadata_batch_and_pair_report_plan_are_reused(tmp_path):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition import body_global,body_global_banks
    from stash_ai.recognition.merge_report import build
    from stash_ai.recognition.models import namespace
    config,_,_,people = seeded(tmp_path)
    with closing(_open_readonly(config.database)) as db:
        queries=[];db.set_trace_callback(queries.append)
        bank=body_global_banks.banks(db,config,'scope',namespace(config))
        assert len(bank)==3
        assert sum(q.startswith('SELECT m.person AS owner') for q in queries)==1
    with patch.object(body_global,'_report',wraps=body_global._report) as calls:
        report=build(config,people[0],people[1])
        assert report['independent_person']['enabled']
        assert calls.call_count==1


def test_equivalent_focus_labels_reuse_report_without_mutating_shared_value(tmp_path):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition import body_global
    from stash_ai.recognition.run_cache import scope
    from stash_ai.recognition.models import namespace
    config,_,_,people = seeded(tmp_path)
    with closing(_open_readonly(config.database)) as db,scope(config):
        with patch.object(body_global,'_report',wraps=body_global._report) as calls:
            first=body_global.report(db,config,'scope',namespace(config),focus=set(people[:2]))
            first['pairs'].clear()
            second=body_global.report(db,config,'scope',namespace(config),focus={p:42 for p in people[:2]})
            assert calls.call_count==1 and second['pairs']


def test_scan_head_index_avoids_sort_and_upgrades_existing_database(tmp_path):
    from tests.test_recognition_merge_report import seeded
    from stash_ai.recognition.database import FaceDatabase
    config,_=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        state.connection.execute('DROP INDEX scans_latest')
    with FaceDatabase(config.database) as state:
        query='SELECT id FROM scans WHERE scope=? AND namespace=? AND scene=? ORDER BY rowid DESC LIMIT 1'
        plan=list(state.connection.execute('EXPLAIN QUERY PLAN '+query,('scope','ns','1')))
        assert any('scans_latest' in r[3] for r in plan)
        assert not any('TEMP B-TREE' in r[3] for r in plan)


def test_compact_body_identity_report_is_lossless_and_sublinear_for_small_change():
    from stash_ai.recognition.trial_report_compaction import compact,expand,ENCODING
    from stash_ai.recognition.database import encoded
    baseline=[dict(person=f'p{i:04}',status='accepted',observations=3,anchors=3,auxiliary=0) for i in range(1000)]
    changed=[dict(row) for row in baseline[1:]]
    changed[0]['observations']=6
    report=dict(identities=changed,assignments=dict(accepted=3000),reasons={'same':1000})
    delta=compact(report,baseline)
    assert delta['identities_encoding']==ENCODING
    assert len(encoded(delta)) < len(encoded(report))//100
    assert expand(delta,baseline)==report
    assert compact(delta,baseline) is delta
    assert baseline[1]['observations']==3


def test_replay_compacts_intermediate_body_report_without_changing_decisions(tmp_path):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.replay_simulation import replay
    from stash_ai.recognition.trial_report_compaction import expand
    config,_,_,_=seeded(tmp_path)
    full=replay(config,{'same-policy':config},scope='scope')
    compact=replay(config,{'same-policy':config},scope='scope',compact_trial_reports=True)
    reference=compact['trials']['baseline']['independent_person']['identities']
    trial=compact['trials']['same-policy']
    assert 'identities' not in trial['independent_person']
    assert expand(trial['independent_person'],reference)==full['trials']['same-policy']['independent_person']
    assert trial['versus_baseline']==full['trials']['same-policy']['versus_baseline']
