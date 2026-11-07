from stash_ai.recognition import rebuild
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.replay_input_cache import InputCache
from stash_ai.recognition.replay_memory import memory_replay
from tests.test_recognition_replay_checkpoints import configured,selection
from tests.test_recognition_clustering import setup


def test_append_reuses_precanonical_scene_rows_and_rebinds_current_selection(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config,count=3)
        def collect(items,use_cache):
            cache=InputCache(state,items,config,'scope','ns') if use_cache else None
            with memory_replay(config) as (db,_):
                rebuild._copy_selection(state,db,items,'scope','ns',bindings=False,inputs=cache)
                from stash_ai.recognition.evidence_pixels import canonicalize
                canonicalize(db)
                return list(map(tuple,db.execute('SELECT id,quality,rejected FROM observations ORDER BY id')))
        collect(selected[:2],True)
        expected=collect(selected,False)
        calls=[];original=rebuild._copy_observations
        def counted(db,replay,scan,**kw):
            calls.append(scan);return original(db,replay,scan,**kw)
        monkeypatch.setattr(rebuild,'_copy_observations',counted)
        assert collect(selected,True)==expected
        assert calls==[selected[-1]['scan']]


def test_existing_shard_pages_only_reserve_future_growth(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition.disk_budget import scope,reserve_file
    path=tmp_path/'faces';shard=tmp_path/'faces.cache'
    path.write_bytes(b'a'*100);shard.write_bytes(b'b'*400)
    config=SimpleNamespace(database=path,data={'execution':{'disk_total_bytes':800,'disk_free_bytes':0}})
    with scope(config), reserve_file(shard,900) as grant:
        assert grant==700
