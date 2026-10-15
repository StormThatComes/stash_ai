from tests.test_recognition_video import video
from types import SimpleNamespace
import pytest
from tests.test_recognition_clustering import setup as cluster_config
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_run import run
import json


def test_transient_http_error_allows_a_later_successful_video_seek(tmp_path, video):
    import http.client
    from stash_ai.recognition.video import VideoFrames
    calls=[]
    class Failure:
        status_code=503
        headers={}
        def raise_for_status(self):raise OSError('temporary HTTP 503')
        def close(self):pass
    def get(*a,**kw):
        from tests.helpers.recognition import Response
        calls.append(True)
        return Failure() if len(calls)==1 else Response(video)
    client=SimpleNamespace(url='http://stash/',media=SimpleNamespace(get=get))
    with VideoFrames(client,{'paths':{'stream':'/stream'}},cluster_config(tmp_path)) as owner:
        connection=http.client.HTTPConnection('127.0.0.1',owner.server.server_port,timeout=2)
        try:
            connection.request('GET',owner.path)
            with pytest.raises(http.client.RemoteDisconnected):connection.getresponse()
        finally:connection.close()
        assert owner.remaining>0 and owner.error is None and owner.last_error is not None
        image,_=owner.frame(0.)
        with image:assert image.size==(160,160)
        assert len(calls)>=2


def test_video_bridge_omits_compressed_lengths_for_decoded_body(tmp_path):
    import gzip,http.client
    from stash_ai.recognition.video import VideoFrames
    from tests.helpers.recognition import Response
    data=b'video-bytes-'*100
    response=Response(data)
    response.headers={'Content-Length':str(len(gzip.compress(data))),'Content-Encoding':'gzip','Content-Type':'video/mp4'}
    client=SimpleNamespace(url='http://stash/',media=SimpleNamespace(get=lambda *a,**kw:response))
    with VideoFrames(client,{'paths':{'stream':'/stream'}},cluster_config(tmp_path)) as owner:
        connection=http.client.HTTPConnection('127.0.0.1',owner.server.server_port,timeout=2)
        try:
            connection.request('GET',owner.path)
            result=connection.getresponse()
            received=result.read()
            assert received == data
            assert result.getheader('Content-Length') is None
            assert result.getheader('Content-Encoding') is None
        finally:connection.close()


def test_valid_mpeg_program_stream_is_supported_without_relaxing_protocols(tmp_path,monkeypatch):
    import shutil,subprocess
    from tests.test_recognition_video import media_server
    from stash_ai.recognition.video import VideoFrames
    binary=shutil.which('ffmpeg')
    if not binary:pytest.skip('requires system FFmpeg')
    path=tmp_path/'source.mpg'
    subprocess.run([binary,'-hide_banner','-loglevel','error','-f','lavfi','-i','testsrc=size=160x160:rate=10',
                    '-t','1','-threads','1','-c:v','mpeg2video','-pix_fmt','yuv420p','-f','mpeg',str(path)],check=True,capture_output=True,timeout=10)
    config=cluster_config(tmp_path)
    with media_server(path.read_bytes()) as (client,_):
        with VideoFrames(client,{'paths':{'stream':'/stream'}},config) as owner:
            image,_=owner.frame(0.)
            with image:assert image.size==(160,160)


def test_video_fallback_preserves_original_decoder_diagnostics(tmp_path,monkeypatch,capsys):
    from tests.test_recognition_metadata import setup,Decoder
    config,client,services=setup(tmp_path,monkeypatch)
    reason='specific FFmpeg executable was not found'
    class Failed(Decoder):
        def frame(self,*a):raise OSError(reason)
    services.decoder=Failed
    assert run(config,services,'1') == 0
    output=capsys.readouterr().out
    with FaceDatabase(config.database) as db:
        for table,column in [('runs','summary'),('portraits','metadata')]:
            assert reason in ''.join(r[0] for r in db.connection.execute(f'SELECT {column} FROM {table}'))
    assert reason in output


def test_cached_metadata_sanitization_removes_nested_credentials():
    from stash_ai.recognition.diagnostics import metadata
    raw={'source':'http://stash/video?apikey=secret','exclusions':[{'quality_errors':{'visibility':'failed http://stash/model?apikey=secret'}}]}
    cleaned=metadata(raw,'secret')
    assert cleaned['source']=='http://stash/video'
    assert 'secret' not in json.dumps(cleaned)
    assert 'secret' in json.dumps(raw)


def test_compressed_range_is_refused_and_identity_encoding_is_requested(tmp_path):
    import http.client
    from tests.helpers.recognition import Response
    from stash_ai.recognition.video import VideoFrames
    requests = []
    def get(*args, **kwargs):
        requests.append(kwargs['headers'])
        response = Response(b'decoded bytes')
        response.status_code = 206
        response.headers = {'Content-Encoding': 'gzip', 'Content-Range': 'bytes 0-4/5', 'Content-Length': '5'}
        return response
    client = SimpleNamespace(url='http://stash/', media=SimpleNamespace(get=get))
    with VideoFrames(client, {'paths': {'stream': '/stream'}}, cluster_config(tmp_path)) as owner:
        connection = http.client.HTTPConnection('127.0.0.1', owner.server.server_port, timeout=2)
        try:
            connection.request('GET', owner.path, headers={'Range': 'bytes=0-'})
            with pytest.raises(http.client.RemoteDisconnected):
                connection.getresponse()
        finally:
            connection.close()
        assert isinstance(owner.error, ValueError)
        assert requests == [{'Accept-Encoding': 'identity', 'Range': 'bytes=0-'}]
