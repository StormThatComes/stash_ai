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
