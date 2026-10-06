"""Regression coverage for the removal of direct video access; no video files/tools."""

from tests.helpers.evidence import empty_policies as empty_tag_policies
from copy import deepcopy
from dataclasses import asdict
import builtins
import io
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

from stash_ai.inference import sprites
from stash_ai.stash_api.client import StashClient
from stash_ai.storage import codec as state_codec
from stash_ai.cleanup import evidence as cleanup_evidence
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.storage.store as merge_store
from stash_ai.evidence.temporal.policy import TemporalPolicy, load_policy
from stash_ai.models import SOURCES
from tests.helpers.evidence import segment


pytestmark = pytest.mark.usefixtures("reset_temporal")


@pytest.fixture(autouse=True)
def forbid_video_access(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("A video tool or external process was invoked")
    monkeypatch.setattr(subprocess, 'Popen', blocked)
    monkeypatch.setattr(subprocess, 'run', blocked)
    monkeypatch.setattr(shutil, 'which', blocked)
    def guarded(function):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                assert Path(os.fsdecode(path)).suffix.lower() not in {'.mkv','.mp4','.avi','.mov','.webm'}, 'Video file touched'
            return function(path, *args, **kwargs)
        return call
    for module, name in ((builtins,'open'),(io,'open'),(os,'stat'),(os,'access')):
        monkeypatch.setattr(module, name, guarded(getattr(module, name)))


def old_video_payload():
    return dict(signature='old-video-job',video={'path':'/not-accessible/movie.mkv','size':7,'mtime_ns':123},
                samples=[{'time':.5,'tags':[['video_only',1.0]]}])




@pytest.mark.parametrize('flag',['--refine-markers','--no-refine-markers'])
def test_removed_cli_flags_fail_before_any_model_or_stash_access(flag,monkeypatch):
    monkeypatch.setattr(app.services,'load_model_settings',lambda *args:pytest.fail('invalid flags reached model configuration'))
    with pytest.raises(SystemExit) as error:app.main([flag])
    assert error.value.code==2


@pytest.mark.parametrize('enabled',[False,True])
def test_old_video_yaml_is_rejected_explicitly(tmp_path,enabled):
    path=tmp_path/'policy.yml'
    path.write_text('version: 1\nrefine_enabled: '+str(enabled).lower()+'\n')
    with pytest.raises(ValueError,match='Video-file analysis has been removed'):
        load_policy(path)


def test_scene_query_only_requests_sprite_and_vtt_media():
    client=StashClient.__new__(StashClient)
    client.create_markers=app.CREATE_MARKERS
    fields=client.scene_fields()
    assert 'paths { sprite vtt }' in fields
    assert 'files' not in fields and 'stream' not in fields and 'duration' not in fields


def test_existing_sprite_reader_uses_only_stash_http_bytes(monkeypatch):
    monkeypatch.setattr(app,'STASH_URL','http://stash.test:9999')
    client=StashClient.__new__(StashClient)
    client.url=app.STASH_URL
    client.request_timeout=app.REQUEST_TIMEOUT
    with Image.new('RGB',(2,1),'white') as sprite:
        buffer=io.BytesIO();sprite.save(buffer,format='PNG');png=buffer.getvalue()
    urls={'http://stash.test:9999/thumb.vtt':b'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nsprite.png#xywh=0,0,1,1\n\n00:00:01.000 --> 00:00:02.000\nsprite.png#xywh=1,0,1,1\n',
          'http://stash.test:9999/sprite.png':png}
    requested=[]
    def get(url,**kwargs):
        assert url in urls
        requested.append(url)
        return SimpleNamespace(status_code=200,content=urls[url],raise_for_status=lambda:None)
    client.media=SimpleNamespace(get=get)
    paths={'sprite':'/sprite.png','vtt':'/thumb.vtt'}
    cues,total=sprites.fetch_cues(app, client,paths)
    samples=[(cue.start,image.getpixel((0,0))) for cue,image in sprites.iter_sprite_frames(client,paths,cues)]
    assert total==2 and samples==[(0,(255,255,255)),(1,(255,255,255))]
    assert requested==list(urls)


def test_full_pipeline_ignores_local_movie_paths_in_scene_metadata(pipeline):
    p=pipeline
    p.stash.scene['files']=[{'path':'/not-accessible/movie.mkv','duration':10}]
    assert app.main(p.args)==0
    assert p.stash.scene['scene_markers']










def test_frozen_old_policies_do_not_reenable_retired_options(tmp_path):
    policies={model:frozenset() for model in SOURCES}
    records={model:segment(model,[[('core',.7)]]*3) for model in ('wd','dtq','oppai')}
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        for model,segments in records.items():state.put('1',model,{'markers':segments})
        _,spans=merge.evidence(app,state,'1',policies,policies)
        for span in spans:
            from stash_ai.db.snapshot_migration import complete_policy
            old = {**span['build_options']['policy'], 'refine_enabled': True, 'refine_path_map': [{'stash': '/media', 'local': '/not-accessible'}]}
            span['build_options']['policy'] = complete_policy(old)
        assert merge_intervals.union_intervals(spans)[0]['sources']==['wd','dtq','oppai']
        policies['wd']=frozenset({'core'})
        cleaned=cleanup_evidence.cleanup_support(app,state,'1',[s for s in spans if s['source']!='wd'],{'wd'},policies, tag_policies=empty_tag_policies())
        assert merge_intervals.union_intervals(cleaned)[0]['sources']==['dtq','oppai']
        assert all(not any(k.startswith('refine_') for k in span['build_options']['policy']) for span in cleaned)


def test_appended_results_never_retain_retired_video_payload(tmp_path):
    data=segment('wd',[[('core',.9)]]*3)
    with merge_store.StateStore(tmp_path/'state.sqlite3','fixture') as state:
        data['refinement']=old_video_payload()
        with pytest.raises(ValueError, match='Retired refinement'):
            state.put('1','wd',{'markers':data})
        data.pop('refinement')
        state.put('1','wd',{'markers':data})
        raw=state.connection.execute("SELECT data FROM results WHERE kind='markers'").fetchone()[0]
        records=state_codec.decode(raw)
        assert isinstance(records, dict) and 'refinement' not in records
        assert not any(key.startswith('refine_') for key in asdict(TemporalPolicy()))
