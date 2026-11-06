"""Small resource/semantic regressions; no timed assertions or model downloads."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from stash_ai.recognition.config import defaults
from stash_ai.recognition.disk_budget import Budget
from stash_ai.recognition.query_claims import Claims
from stash_ai.recognition.resource_errors import ResourceLimit


def config(tmp_path):
    return SimpleNamespace(data=defaults(),database=tmp_path/'faces.sqlite3')


def test_volume_budget_includes_database_cache_and_checkpoints(tmp_path):
    c=config(tmp_path);c.data['execution']['disk_total_bytes']=100
    c.database.write_bytes(b'a'*40)
    Path(str(c.database)+'.media.sqlite3').write_bytes(b'b'*40)
    checkpoint=Path(str(c.database)+'.rebuild');checkpoint.mkdir()
    (checkpoint/'image.sqlite3').write_bytes(b'c'*30)
    with pytest.raises(ResourceLimit,match='per-volume'):Budget(c).check()


def test_queries_spool_no_body_vectors_and_keep_support_semantics(tmp_path):
    c=config(tmp_path)
    proof={'signal':'face','core_observations':[]}
    with Claims(c) as claims:
        for i in range(4):
            row=dict(id=str(i),person='p',scan='s',cue=i,geometry='[]',dimensions=512,
                     embedding=b'\0'*2048,body=b'\0'*16384)
            vote=dict(scan='s',seconds=i,quality='{}')
            claims.add(row,('c'+str(i),str(i),'p',proof,None,vote),('p','face'))
        assert claims.resolve({},lambda _:3)==set()
        assert claims.support(('p','face'))==4
        assert all(claims.accepted(str(i)) for i in range(4))
        assert [r[1] for r in claims.db.execute('PRAGMA table_info(conflicts)')]==[
            'id','person','scan','cue','geometry','dimensions','embedding']
        assert not hasattr(claims,'claim_rows')


def test_query_disk_plan_limit_is_enforced_before_publication(tmp_path):
    c=config(tmp_path);c.data['sapiensid']['execution']['plan_bytes']=1
    with Claims(c) as claims:
        with pytest.raises(ResourceLimit,match='plan_bytes'):
            claims.add(dict(id='a',scan='s',cue=0,geometry='[]',embedding=b'0'*2048),
                       ('c','a','p',{},None,{}),('p','face'))
        assert list(claims.rows())==[]
