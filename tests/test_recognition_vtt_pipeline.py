"""Ownership, whole-cue checkpoints and dynamic-only microbatch regression tests."""
from dataclasses import replace
from types import SimpleNamespace
from threading import get_ident
from io import BytesIO
import numpy as np
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.scanning import scan_scene
from tests.test_recognition_scanning import config
from tests.helpers.recognition import MediaClient, Backend, FACE, Response


def test_cpu_jobs_and_single_inference_owner_keep_complete_cue_checkpoints(tmp_path, monkeypatch):
    from stash_ai.recognition import scanning
    settings = config(tmp_path)
    data = plain(settings.data); data['execution'].update(recognizer_batch_size=2,scan_cpu_workers=3)
    settings = replace(settings,data=data)
    owner, cpu, gpu, batches = get_ident(), set(), [], []
    original = scanning.assess
    def assess(*args):
        cpu.add(get_ident()); return original(*args)
    monkeypatch.setattr(scanning,'assess',assess)
    class Prepared(Backend):
        def detect(self,image):
            gpu.append(get_ident()); return [FACE]*3
        def prepare_embedding(self,image,face):
            cpu.add(get_ident()); return np.ones((1,3,112,112),dtype='float32')
        def embed_prepared_many(self,tensors):
            gpu.append(get_ident()); batches.append(len(tensors))
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError('second cue incomplete')
            return [np.array([1.,0,0,0],dtype='float32') for _ in tensors]
    client=MediaClient(count=3); backend=Prepared()
    with FaceDatabase(settings.database) as state, SpriteInput(client,client.scene,settings) as source:
        with pytest.raises(RuntimeError,match='second cue incomplete'):
            scan_scene(state,'scope',client.scene,settings,'ns',source,lambda:backend)
        assert set(gpu)=={owner} and cpu and owner not in cpu
        assert batches==[2,2,2,2]
        assert state.connection.execute('SELECT cursor,status FROM scans').fetchone()[:]==(2,'partial')
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==6
        assert backend.closed


@pytest.mark.parametrize('input_batch,output_batch,expected', [(1,1,[1,1,1]),('N','N',[3]),('N',1,[1,1,1])])
def test_microbatch_requires_dynamic_input_and_output_contracts(input_batch,output_batch,expected):
    from stash_ai.recognition.embedding_batches import infer
    calls=[]
    class Session:
        def get_inputs(self):return [SimpleNamespace(shape=[input_batch,3,112,112])]
        def get_outputs(self):return [SimpleNamespace(shape=[output_batch,4])]
        def run(self,outputs,values):
            count=len(values['in']); calls.append(count)
            return [np.tile([1.,0,0,0],(count,1)).astype('float32')]
    owner=SimpleNamespace(session=Session(),input_name='in',output_name='out',dimensions=4)
    values=infer(owner,[np.ones((1,3,112,112),dtype='float32') for _ in range(3)])
    assert len(values)==3 and calls==expected


def test_prepared_inference_failure_invalidates_lazy_backend(tmp_path):
    from stash_ai.recognition.backend import LazyBackend
    class Broken(Backend):
        def embed_prepared_many(self,tensors):raise RuntimeError('broken batch')
    model=Broken(); owner=LazyBackend(config(tmp_path),None,lambda *args:model)
    borrowed=owner.borrow()
    with pytest.raises(RuntimeError,match='broken batch'):
        borrowed.embed_prepared_many([])
    assert model.closed and owner.model is None


def test_shared_sprite_budget_retains_more_than_two_sheets_without_repeat_downloads(tmp_path):
    from stash_ai.recognition.sprite_cache import scope
    from PIL import Image
    settings=config(tmp_path); client=MediaClient(count=3)
    urls=[client.url+f'sprite/{i}.png' for i in range(3)]
    vtt='WEBVTT\n\n'+'\n\n'.join(f'00:00:0{i}.000 --> 00:00:0{i+1}.000\n{url}#xywh=0,0,16,16' for i,url in enumerate(urls))
    client.data[client.url+'vtt/1.vtt']=vtt.encode()
    for index,url in enumerate(urls):
        with Image.new('RGB',(32,32),(index*100,20,30)) as image:
            encoded=BytesIO();image.save(encoded,format='PNG');client.data[url]=encoded.getvalue()
    with scope(settings) as cache:
        with SpriteInput(client,client.scene,settings) as source:
            for _,_,image in source.frames():
                assert image.size==(16,16)
            for index in range(3):
                with source.thumbnail(index) as image:assert image.size==(16,16)
            assert all(client.calls.count(url)==1 for url in urls)
            assert cache.size<=cache.limit and cache.peak<=cache.limit
        # Source close cannot destroy shared entries needed by portraits.
        assert cache.entries
    assert not cache.entries and cache.size==0


def test_tiny_sprite_cache_streams_identical_pixels(tmp_path):
    from stash_ai.recognition.sprite_cache import scope
    settings=config(tmp_path); data=plain(settings.data);data['execution']['sprite_cache_bytes']=0
    settings=replace(settings,data=data);client=MediaClient()
    with scope(settings) as cache, SpriteInput(client,client.scene,settings) as source:
        with source.thumbnail(0) as first, source.thumbnail(1) as second:
            assert first.tobytes()==second.tobytes()
        assert not cache.entries


def test_prefetch_fills_depth_and_retires_in_requested_order(tmp_path, monkeypatch):
    from stash_ai.recognition.prefetch import ScenePrefetch
    from tests.test_recognition_prefetch import setup,readers
    settings,client,services,_=setup(tmp_path,monkeypatch)
    created=readers(client)
    inputs=ScenePrefetch(client,settings,services.source)
    try:
        inputs.fill(['1','2','3','4'])
        assert list(inputs.queued)==['1','2','3']
        for identifier in ('1','2','3'):
            with inputs.take(identifier) as value:assert str(value.scene['id'])==identifier
        assert not inputs.queued
    finally:
        inputs.close()
    assert all(reader.closed for reader in created)


def test_scan_worker_counts_do_not_change_retained_bytes(tmp_path):
    snapshots=[]
    for workers in (1,4):
        directory=tmp_path/str(workers);directory.mkdir()
        settings=config(directory);data=plain(settings.data);data['execution']['scan_cpu_workers']=workers
        settings=replace(settings,data=data);client=MediaClient()
        backend=Backend();backend.detect=lambda image:[FACE]*5
        with FaceDatabase(settings.database) as state, SpriteInput(client,client.scene,settings) as source:
            scan_scene(state,'scope',client.scene,settings,'ns',source,lambda:backend)
            snapshots.append([tuple(r) for r in state.connection.execute('SELECT * FROM observations ORDER BY id')])
    assert snapshots[0]==snapshots[1]


def test_portrait_identity_ranking_runs_on_cpu_without_changing_selection(tmp_path,monkeypatch):
    from stash_ai.recognition import portraits
    from tests.test_recognition_portrait_pipeline import multi,result
    from tests.test_recognition_run import run
    seen=[]; original=portraits.identity_reason
    def identity(*args):
        import threading
        seen.append(threading.current_thread().name)
        return original(*args)
    monkeypatch.setattr(portraits,'identity_reason',identity)
    settings,client,services,_,_=multi(tmp_path,monkeypatch,True)
    assert run(settings,services,'1')==0
    assert any(name.startswith('portrait-cpu') for name in seen)
    assert len(result(settings))==2


def test_frame_decode_uses_shared_budget_instead_of_portrait_sheet_cap(tmp_path,monkeypatch):
    from stash_ai.recognition import sprite_cache
    opened=[]; original=sprite_cache.open_image
    def opening(*args,**kwargs):
        opened.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(sprite_cache,'open_image',opening)
    settings=config(tmp_path);client=MediaClient()
    with SpriteInput(client,client.scene,settings) as source:
        source.portrait_sheet_bytes=1
        for _,_,image in source.frames():assert image.size==(160,160)
        assert len(opened)==1


def test_distinct_sprite_decodes_can_overlap_without_duplicate_decode(monkeypatch):
    from stash_ai.recognition import sprite_cache
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier,Lock
    from PIL import Image
    gate=Barrier(2); calls=[]; lock=Lock()
    original=sprite_cache.open_image
    def opened(*args,**kwargs):
        with lock:
            calls.append(1)
        gate.wait(timeout=5)
        return original(*args,**kwargs)
    monkeypatch.setattr(sprite_cache,'open_image',opened)
    out=BytesIO()
    with Image.new('RGB',(4,4),'red') as source:
        source.save(out,format='PNG')
    cache=sprite_cache.SpriteCache(10000)
    def get(key):
        return cache.crop(key,out.getvalue(),None,lambda image,cue:image.crop((0,0,2,2)))
    with ThreadPoolExecutor(2) as workers:
        first=workers.submit(get,'a');second=workers.submit(get,'b')
        first.result().close();second.result().close()
    assert len(calls)==2 and not cache.decoders
    get('a').close()
    assert len(calls)==2
    cache.close()
