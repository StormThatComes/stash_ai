from types import SimpleNamespace
from tests.test_recognition_clustering import setup as cluster_config
from stash_ai.recognition.media import SpriteInput
import pytest
from stash_ai.recognition.database import encoded
from tests.helpers.recognition import MediaClient
import json
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_run import run
from tests.test_recognition_rebuild import changed
from tests.helpers.recognition import Backend
import hashlib


def test_query_distinct_sheets_have_distinct_sanitized_fingerprints(tmp_path):
    from io import BytesIO
    from PIL import Image
    from tests.helpers.recognition import Response
    config=cluster_config(tmp_path)
    sheets=[]
    for color in ('red','blue'):
        with Image.new('RGB',(160,160),color) as image:
            output=BytesIO();image.save(output,format='PNG');sheets.append(output.getvalue())
    fingerprints=[];pixels=[]
    for last in ('a','b'):
        urls=['a','b',last]
        vtt=('WEBVTT\n\n'+'\n\n'.join(f'00:00:0{i}.000 --> 00:00:0{i+1}.000\nsheet.jpg?part={part}#xywh=0,0,160,160' for i,part in enumerate(urls))).encode()
        def get(url,**kw):return Response(vtt if '/vtt' in url else sheets[0 if url.endswith('=a') else 1])
        client=SimpleNamespace(url='http://stash/',media=SimpleNamespace(get=get))
        with SpriteInput(client,{'paths':{'vtt':'/vtt'}},config) as source:
            fingerprints.append(source.fingerprint)
            with source.thumbnail(2) as image:pixels.append(image.getpixel((0,0)))
    assert pixels[0] != pixels[1] and fingerprints[0] != fingerprints[1]


def test_transient_thumbnail_failure_allows_later_cues(tmp_path):
    from stash_ai.recognition.vtt_portraits import Sources
    config=cluster_config(tmp_path)
    media=MediaClient()
    source=SpriteInput(media,media.scene,config)
    calls=[]
    original=source.thumbnail
    def thumbnail(index):
        calls.append(index)
        if len(calls)==1:raise OSError('one transient thumbnail read')
        return original(index)
    source.thumbnail=thumbnail
    client=SimpleNamespace(scene=lambda _:media.scene)
    from stash_ai.runtime.cleanup import closing
    with closing(Sources(client,config,lambda *a:source)) as owner:
        def row(index):
            cue=source.cues[index]
            return {'input_sha':source.fingerprint,'cue':index,'seconds':cue.start,'source':cue.image,'rect':encoded(cue.rect)}
        with pytest.raises(OSError):owner.thumbnail('1',row(0))
        with owner.thumbnail('1',row(1)) as image:
            assert image.size == (160, 160)
        assert calls == [0, 1]


def test_corrupt_video_cache_reacquires_the_available_frame(tmp_path,monkeypatch):
    from tests.test_recognition_metadata import setup,Decoder
    from stash_ai.recognition.portraits import choose
    from tests.helpers.recognition import Visibility
    config,client,services=setup(tmp_path,monkeypatch)
    config=changed(config,portraits={'max_candidates':1})
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        person=state.connection.execute('SELECT * FROM people').fetchone()
        selected=state.connection.execute('SELECT id FROM scans').fetchone()[0]
        state.connection.execute("UPDATE portraits SET image=? WHERE status='published'",(b'corrupt cached JPEG',));state.connection.commit()
        Decoder.instances.clear()
        target={'person':person['id'],'performer':person['performer'],'scans':[{'scene':'1','scan':selected}]}
        winner=choose(state,client,config,person['scope'],target,{},lambda *a:Backend(),Decoder,visibility=Visibility())
        assert json.loads(winner['metadata'])['source_type']=='video'
        assert len(Decoder.instances)==1


def test_bad_newest_portrait_does_not_hide_verified_thumbnail(tmp_path,monkeypatch):
    from tests.test_recognition_vtt_portraits import setup
    from stash_ai.recognition.portrait_availability import cached_thumbnail
    config,client,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1')==0
    with FaceDatabase(config.database) as state:
        old=state.connection.execute("SELECT * FROM portraits WHERE status='published'").fetchone()
        metadata=json.loads(old['metadata'])
        state.connection.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
            ('broken-newest',old['scope'],old['person'],old['scene'],old['metadata'],b'bit rot',old['digest'],'selected'))
        state.connection.commit()
        row=state.connection.execute('SELECT o.*,s.input_sha FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id=?',(metadata['observation'],)).fetchone()
        target={'person':old['person'],'performer':metadata['performer']}
        assert hashlib.sha256(old['image']).hexdigest()==old['digest']
        reused=cached_thumbnail(state,config,old['scope'],target,row,'new-policy',dict(metadata))
        assert reused is not None and reused['image']==old['image']


def test_matching_digest_does_not_authorize_non_image_bytes(tmp_path,monkeypatch):
    from tests.test_recognition_vtt_portraits import setup
    config,client,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1') == 0
    bad=b'this is not an image'
    with FaceDatabase(config.database) as state:
        row=state.connection.execute("SELECT * FROM portraits WHERE status='published'").fetchone()
        state.connection.execute('UPDATE portraits SET image=?,digest=? WHERE id=?',(bad,hashlib.sha256(bad).hexdigest(),row['id']))
        state.connection.commit()
    # The fake server accepts the submitted bytes; a real server may reject them.
    assert run(config,services,'1') == 0
    assert bad not in client.images.values()


def test_source_cleanup_preserves_primary_failure_and_clears_cache(tmp_path,monkeypatch):
    from PIL import Image
    config=cluster_config(tmp_path)
    media=MediaClient()
    source=SpriteInput(media,media.scene,config)
    close=Image.Image.close
    def broken(image):
        close(image)
        raise OSError('sheet cleanup failed')
    frames=source.frames()
    next(frames)
    monkeypatch.setattr(Image.Image,'close',broken)
    with pytest.raises(RuntimeError,match='primary processing failure') as caught:
        frames.throw(RuntimeError('primary processing failure'))
    assert any('sheet cleanup failed' in note for note in caught.value.__notes__)
    assert not source.cache
