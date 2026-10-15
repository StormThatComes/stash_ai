import json
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.clustering import cluster_scene
from tests.test_recognition_run import run
from tests.test_recognition_clustering import scan
from tests.helpers.recognition import Backend
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_clustering import setup as cluster_config, scan
from tests.test_recognition_rebuild import changed


def test_ineligible_prior_video_does_not_block_valid_replacement(tmp_path,monkeypatch):
    from PIL import Image
    from tests.test_recognition_metadata import setup,Decoder
    from tests.test_recognition_cache_revalidation import QUERY
    from stash_ai.recognition.portraits import fill_portrait
    from stash_ai.runtime.report import RunReport
    from tests.helpers.recognition import Visibility
    config,client,services=setup(tmp_path,monkeypatch)
    class Distinct(Decoder):
        def frame(self,seconds):
            image,timing=super().frame(seconds)
            if seconds:
                image.close();image=Image.new('RGB',(160,160),(120,120,120))
            return image,timing
    services.decoder=Distinct
    model=Backend()
    def embed(image,face):
        model.calls+=1
        return QUERY if model.calls==4 else np.array([1.,0,0,0])
    model.embed=embed;services.backend=lambda *a:model
    assert run(config,services,'1')==0
    original=dict(client.images)
    with FaceDatabase(config.database) as state:
        person=state.connection.execute('SELECT * FROM people').fetchone()
        old=state.connection.execute("SELECT * FROM portraits WHERE status='published'").fetchone()
        original_scan=state.connection.execute('SELECT id FROM scans').fetchone()[0]
        rival,_=scan(state,config,'2',[np.array([0.,1,0,0])]*3)
        state.connection.execute('UPDATE scans SET scope=?,namespace=? WHERE id=?',(person['scope'],person['namespace'],rival));state.connection.commit()
        cluster_scene(state,rival,config)
        target={'person':person['id'],'performer':person['performer'],'scans':[{'scene':'1','scan':original_scan}]}
        counts=fill_portrait(state,client,RunReport(False),config,person['scope'],target,{},services.backend,Distinct,visibility=Visibility())
        current=json.loads(state.connection.execute('SELECT metadata FROM portraits WHERE id=?',(old['id'],)).fetchone()[0])
        assert current['reason']=='portrait_identity_ambiguous'
        assert client.images!=original and counts['portraits_updated']==1


def test_refreshed_prior_rank_allows_better_original_thumbnail(tmp_path, monkeypatch):
    from tests.test_recognition_vtt_portraits import setup
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.visibility import CONTRACT_ID
    config, client, services, models = setup(tmp_path, monkeypatch)
    class Unavailable(SpriteInput):
        def thumbnail(self, index): raise OSError('thumbnail unavailable')
    class Quality:
        restored = False
        def __init__(self, *a): self.calls = 0
        def measure(self, image, face):
            self.calls += 1
            value = (0. if self.calls == 1 else .8) if self.restored else (1. if self.calls == 1 else .5)
            return {'visible_face_fraction':value,'visibility':{'contract':CONTRACT_ID}}
        def close(self): pass
    services.source, services.visibility = Unavailable, Quality
    assert run(config, services, '1') == 0
    original = dict(client.images)
    with FaceDatabase(config.database) as db:
        first = db.connection.execute("SELECT id,metadata FROM portraits WHERE status='published'").fetchone()
        original_id = first['id']
    services.source, Quality.restored = SpriteInput, True
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        best = db.connection.execute("SELECT id FROM portraits ORDER BY json_extract(metadata,'$.score') DESC LIMIT 1").fetchone()[0]
        assert best != original_id
        assert client.images != original
        assert db.connection.execute('SELECT status FROM portraits WHERE id=?', (best,)).fetchone()[0] == 'published'
    assert len(models) == 1


def test_video_hints_respect_zero_sharpness_weight(tmp_path):
    from stash_ai.recognition.portraits import hints
    config = changed(cluster_config(tmp_path), portraits={'max_candidates':1,
        'weights':{'frontal':0.,'sharpness':0.,'resolution':1.,'exposure':0.,'visibility':0.,
                   'neutral_expression':0.,'closed_mouth':0.,'open_mouth':0.,'open_eyes':0.}})
    with FaceDatabase(config.database) as state:
        selected,_ = scan(state,config,'1',count=9)
        person = cluster_scene(state,selected,config)[0]['person']
        rows=state.connection.execute('SELECT id,quality FROM observations ORDER BY cue').fetchall()
        for i,row in enumerate(rows):
            quality=json.loads(row['quality']);quality.update(sharpness=100. if i<8 else 0.,pixels=32. if i<8 else 512.)
            state.connection.execute('UPDATE observations SET quality=? WHERE id=?',(encoded(quality),row['id']))
        state.connection.commit()
        target={'person':person,'scans':[{'scene':'1','scan':selected}]}
        assert hints(state,target,config)[0][1]['id'] == rows[-1]['id']


def test_global_video_shortlist_prefers_best_scene_quality(tmp_path):
    from stash_ai.recognition.portraits import hints
    config=changed(cluster_config(tmp_path),portraits={'max_candidates':1})
    with FaceDatabase(config.database) as state:
        first,_=scan(state,config,'1');person=cluster_scene(state,first,config)[0]['person']
        second,_=scan(state,config,'2');cluster_scene(state,second,config)
        for row in state.connection.execute('SELECT id,scan,quality FROM observations').fetchall():
            quality=json.loads(row['quality']);quality.update(sharpness=0. if row['scan']==first else 1000.,pixels=32. if row['scan']==first else 512.)
            state.connection.execute('UPDATE observations SET quality=? WHERE id=?',(encoded(quality),row['id']))
        state.connection.commit()
        target={'person':person,'scans':[{'scene':'1','scan':first},{'scene':'2','scan':second}]}
        assert hints(state,target,config)[0][0]=='2'
