import json
import numpy as np


def test_sampler_prefix_state_matches_every_cold_prefix():
    from stash_ai.recognition.reference_selection import select
    rng=np.random.default_rng(71);rows=[];state={}
    for scene in range(20):
        added=[]
        for cue in range(17):
            vector=rng.normal(size=16).astype('<f4');vector/=np.linalg.norm(vector)
            added.append(dict(id=f'{scene}-{cue}',scan=str(scene),scene=str(scene),seconds=cue,
                dimensions=16,embedding=vector.tobytes(),geometry='{}',quality=json.dumps(dict(pixels=70+cue,sharpness=scene+10))))
        rows.extend(added)
        actual=select(added,8,state=state)
        expected=select(rows,8)
        assert [i for i,_ in actual]==[i for i,_ in expected]
        assert all(np.array_equal(a,b) for (_,a),(_,b) in zip(actual,expected))


def test_cached_global_views_keep_sources_and_rebuild_only_changed_owner(tmp_path,monkeypatch):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition import global_views
    from stash_ai.recognition.view_catalogue_cache import encode
    from tests.test_recognition_global_views import seed,views
    from tests.test_recognition_anchor_views import config_for,A,X
    c=config_for(tmp_path)
    with FaceDatabase(c.database) as state:
        a,owner=seed(state,c,'1',A);b,other=seed(state,c,'2',X)
        with scope(state,c):expected=encode(views(state,c))
        calls=[];original=global_views._catalogue
        def counted(*args,**kwargs):calls.append(kwargs.get('owner_filter'));return original(*args,**kwargs)
        monkeypatch.setattr(global_views,'_catalogue',counted)
        with scope(state,c):assert encode(views(state,c))==expected
        assert not calls
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.sharpness',777) WHERE scan=?",(a,))
        with scope(state,c):views(state,c)
        assert calls==[owner]
