import pytest
from tests.test_recognition_rebuild import changed, library, partition, two_people, run
from stash_ai.recognition import rebuild


def test_append_reuses_independent_strict_scene_and_matches_cold(tmp_path,monkeypatch):
    cfg,_,services,_=library(tmp_path/'warm',monkeypatch,two_people())
    cfg=changed(cfg,clustering={'anchor_view_matching':True})
    assert run(cfg,services,'1')==0
    called=[]
    original=rebuild.cluster_scene
    def record(database,scan,*args,**kwargs):
        called.append(database.connection.execute('SELECT scene FROM scans WHERE id=?',(scan,)).fetchone()[0])
        return original(database,scan,*args,**kwargs)
    monkeypatch.setattr(rebuild,'cluster_scene',record)
    assert run(cfg,services,'1','2')==0
    assert called==['2']
    cold,_,services,_=library(tmp_path/'cold',monkeypatch,two_people())
    cold=changed(cold,clustering={'anchor_view_matching':True},execution={'rebuild_cache_bytes':0})
    assert run(cold,services,'1','2')==0
    assert partition(cfg)==partition(cold)


def test_parallel_body_plans_reuse_independent_scenes(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.compute import compute_scope
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition.body_scene_jobs import planned
    from stash_ai.recognition.body_identity import scene_plan
    from tests.test_recognition_body_global import configured,scene
    config=configured(tmp_path)
    config.data['execution']['rebuild_workers']=2
    with FaceDatabase(config.database) as state:
        selected=[scene(state.connection,config,str(i))[0] for i in range(1,4)]
        expected=[scene_plan(state.connection,i['scan'],config)[2] for i in selected]
        with scope(state,config),compute_scope(config):
            assert [p[2] for p in planned(state.connection,selected[:2],config)]==expected[:2]
            assert [p[2] for p in planned(state.connection,selected,config)]==expected


@pytest.mark.parametrize(('before','after'),[(10,32),(32,33)])
def test_append_matches_cold_partition_and_statuses(tmp_path,monkeypatch,before,after):
    import json
    from stash_ai.recognition.database import FaceDatabase
    scenes=[str(i+1) for i in range(after)]
    vectors={s:[[float(k==int(s)%4) for k in range(4)]]*2 for s in scenes}
    results=[]
    for name,steps in (('expanded',(before,after)),('cold',(after,))):
        cfg,_,services,_=library(tmp_path/name,monkeypatch,vectors)
        cfg=changed(cfg,clustering={'anchor_view_matching':True},execution={'profile_rebuild':True})
        for amount in steps:
            assert run(cfg,services,*scenes[:amount])==0
        with FaceDatabase(cfg.database) as db:
            statuses=[tuple(row) for row in db.connection.execute(
                "SELECT s.scene,o.cue,c.status FROM candidates c JOIN observations o "
                "ON json_extract(c.payload,'$.observation')=o.id JOIN scans s ON s.id=o.scan ORDER BY s.scene,o.cue")]
            if name=='expanded':
                summary=json.loads(db.connection.execute("SELECT summary FROM runs ORDER BY started DESC LIMIT 1").fetchone()[0])
                assert summary['rebuild_profile']['counts']['strict-scene_cache_hits']==before
        results.append((partition(cfg),statuses))
    assert results[0]==results[1]


def test_duplicate_artifact_neither_evicts_nor_prevents_checksum_repair():
    import sqlite3
    from stash_ai.recognition.derived_cache import Store,PREFIX
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
    store=Store(db,4096)
    for n in range(10):store.put('test',str(n),'x'*160)
    before=store.used
    store.put('test','9','x'*160)
    assert store.used==before and store.get('test','0')=='x'*160
    db.execute('UPDATE metadata SET value=replace(value,?,?) WHERE key=?',
               ('x','y',PREFIX+'test:9'))
    assert store.get('test','9') is None
    store.put('test','9','x'*160)
    assert store.used==before and store.get('test','9')=='x'*160


def test_cache_and_worker_input_include_external_pixel_representative(monkeypatch):
    import json
    import sqlite3
    from stash_ai.recognition import evidence_pixels
    db=sqlite3.connect(':memory:')
    db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE observations(id TEXT,quality TEXT,rejected TEXT)')
    db.execute('INSERT INTO observations VALUES (?,?,NULL)',('face',json.dumps({'evidence_pixel_sha256':'a'*64})))
    monkeypatch.setattr(evidence_pixels,'aliases',lambda db,*a,**kw: [('a'*64,'b'*64)] if db=='external' else ())
    with evidence_pixels.alias_scope('external'):
        evidence_pixels.canonicalize(db)
    raw=db.execute('SELECT quality FROM observations').fetchone()[0]
    assert json.loads(raw)['evidence_pixel_sha256']=='b'*64
    db.close()
