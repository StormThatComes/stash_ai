import json
from types import SimpleNamespace

import pytest

from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.portrait_refresh import retained_targets
from stash_ai.recognition.portrait_writes import batched
from stash_ai.recognition.vtt_portraits import store
from tests.test_recognition_vtt_portraits import setup
from tests.test_recognition_run import run


def test_complete_vtt_ranking_skips_scores_and_keeps_fresh_image_checks(tmp_path,monkeypatch):
    from stash_ai.recognition import vtt_portraits
    config,stash,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        assert state.connection.execute("SELECT 1 FROM metadata WHERE key LIKE 'vtt-ranking-cache:%'").fetchone()
    monkeypatch.setattr(vtt_portraits,'observation_identity',lambda *a,**k:pytest.fail('complete ranking recomputed'))
    assert run(config,services,'1')==0
    performer=stash.links['1'][-1]
    stash.images[performer]=b'manual replacement'
    assert run(config,services,'1')==0
    assert stash.images[performer]==b'manual replacement'


@pytest.mark.parametrize('mutation',[
    "UPDATE observations SET quality=json_set(quality,'$.sharpness',0)",
    "UPDATE candidates SET payload=json_set(payload,'$.proof_revision','changed')",
    "UPDATE portraits SET metadata=json_set(metadata,'$.quality_errors.visibility','retry')",
    "UPDATE portraits SET metadata=json_set(metadata,'$.quality_source','recognition_crop','$.source_warning','retry')",
    "UPDATE metadata SET value='broken' WHERE key LIKE 'vtt-ranking-cache:%'",
])
def test_ranking_dependencies_and_retryable_states_recompute(tmp_path,monkeypatch,mutation):
    from stash_ai.recognition import vtt_portraits
    config,stash,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        state.connection.execute(mutation);state.connection.commit()
    calls=[];original=vtt_portraits.observation_identity
    def check(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(vtt_portraits,'observation_identity',check)
    assert run(config,services,'1')==0
    assert calls


def test_metadata_writes_batch_only_changes_and_rollback_failed_tail(tmp_path):
    with FaceDatabase(tmp_path/'faces.sqlite3') as state:
        db=state.connection;statements=[];db.set_trace_callback(statements.append)
        @batched
        def prepare(database,fail=False):
            for index in range(260):
                store(database,str(index),'scope',dict(person='person'),'1',dict(score=index))
                if fail and index==259:
                    raise RuntimeError('primary')
        with pytest.raises(RuntimeError,match='primary'):
            prepare(state,True)
        assert db.execute('SELECT COUNT(*) FROM portraits').fetchone()[0]==256
        assert statements.count('COMMIT')==2
        prepare(state)
        statements.clear()
        prepare(state)
        assert not any(sql.startswith(('BEGIN','UPDATE','INSERT','COMMIT')) for sql in statements)
        db.set_trace_callback(None)


def test_partial_provenance_index_ignores_bad_metadata_and_serves_lookup(tmp_path):
    path=tmp_path/'faces.sqlite3'
    with FaceDatabase(path) as state:
        state.connection.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
            ('bad','scope','person','1','bad-json',b'bytes','digest','selected'))
        state.connection.commit()
        state.connection.execute('DROP INDEX portraits_observation');state.connection.commit()
    with FaceDatabase(path) as state:
        plan=' '.join(row[3] for row in state.connection.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM portraits WHERE scope=? AND person=? AND json_valid(metadata) "
            "AND json_extract(metadata,'$.observation')=? AND json_extract(metadata,'$.quality_source')='recognition_crop' LIMIT 1",
            ('scope','person','observation')))
        assert 'portraits_observation' in plan
        store(state,'bad','scope',dict(person='person'),'1',dict(score=1,source_type='vtt'))
        row=state.connection.execute('SELECT * FROM portraits WHERE id=?',('bad',)).fetchone()
        assert row['image']==b'bytes' and json.loads(row['metadata'])['source_type']=='unverified_cache'


def test_selected_discovery_drives_from_scenes_and_deduplicates_chunks(tmp_path,monkeypatch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        scope=state.connection.execute('SELECT scope FROM people').fetchone()[0]
        expected=retained_targets(state,config,scope)
        statements=[];state.connection.set_trace_callback(statements.append)
        assert retained_targets(state,config,scope,['1']*260+[str(i+2) for i in range(259)])==expected
        assert len([sql for sql in statements if 'CROSS JOIN scans' in sql])==2
        assert retained_targets(state,config,scope,[])=={}
        assert retained_targets(state,config,scope,['1'],performer_ids=['99999'])=={}
        state.connection.set_trace_callback(None)


def test_ranking_cache_uses_bytes_instead_of_global_entry_cap(tmp_path):
    from stash_ai.recognition.vtt_ranking_cache import Ranking
    with FaceDatabase(tmp_path/'db') as state:
        ranking=Ranking.__new__(Ranking)
        ranking.db=state.connection;ranking.enabled=True;ranking.hit=False;ranking.limit=100000
        ranking.inputs=lambda **kwargs:'f'*64
        for index in range(100):
            ranking.key='vtt-ranking-cache:'+str(index)
            ranking.save([])
        assert ranking.db.execute("SELECT COUNT(*) FROM metadata WHERE key LIKE 'vtt-ranking-cache:%'").fetchone()[0]==100
        ranking.limit=2000;ranking.key='vtt-ranking-cache:newest';ranking.save([])
        size=ranking.db.execute("SELECT SUM(length(CAST(value AS BLOB))+length(key)+256) FROM metadata WHERE key LIKE 'vtt-ranking-cache:%'").fetchone()[0]
        assert size<=2000
        assert ranking.db.execute('SELECT 1 FROM metadata WHERE key=?',(ranking.key,)).fetchone()
