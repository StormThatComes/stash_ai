import pytest
import json
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_run import setup as run_setup, run
import sqlite3
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_clustering import setup as cluster_config, scan


def test_frame_prefetch_shutdown_is_bounded_for_noncooperative_source(tmp_path):
    import subprocess,sys
    script='''import sys,threading
from pathlib import Path
from stash_ai.recognition.prefetch import FramePrefetch
started=threading.Event()
class Source:
    def frames(self,start):
        started.set()
        threading.Event().wait()
        yield None
owner=FramePrefetch(Source(),0,1).launch()
assert started.wait(1)
Path(sys.argv[1]).write_text('closing an owned blocked frame source')
owner.close()
'''
    marker=tmp_path/'started'
    result=subprocess.run([sys.executable,'-c',script,str(marker)],capture_output=True,timeout=4)
    assert result.returncode != 0 and b'quarantined' in result.stderr
    assert marker.exists()


def test_failed_decoder_termination_does_not_close_busy_pipe(tmp_path):
    import subprocess,sys
    script='''import os,sys,subprocess,threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from stash_ai.recognition.config import load
from stash_ai.recognition import video
root=Path(sys.argv[1])
owner=video.VideoFrames(SimpleNamespace(url='http://stash/'),{'paths':{'stream':'/video'}},load(root=root,device='cpu'))
owner.server=SimpleNamespace(server_port=8000)
read_fd,write_fd=os.pipe()
reader=os.fdopen(read_fd,'rb')
class Pipe:
    def read(self,size):return reader.read(size)
    def close(self):
        (root/'closing-pipe').write_text('join returned; closing a busy buffered reader')
        reader.close()
def kill():raise OSError('termination unavailable')
def wait(**kw):raise subprocess.TimeoutExpired('owned decoder',kw.get('timeout',0))
process=SimpleNamespace(stdout=Pipe(),stderr=BytesIO(),poll=lambda:None,kill=kill,wait=wait)
video.subprocess.Popen=lambda *a,**kw:process
Thread=threading.Thread
class FailingSecond(Thread):
    starts=0
    def start(self):
        FailingSecond.starts+=1
        if FailingSecond.starts==2:raise RuntimeError('second reader start failed')
        return super().start()
    def join(self,timeout=None):return super().join(timeout=.03)
video.threading.Thread=FailingSecond
owner.frame(0)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,timeout=4)
    assert result.returncode != 0 and b'second reader start failed' in result.stderr
    assert not (tmp_path/'closing-pipe').exists()


def test_history_records_fatal_model_cleanup_failure(tmp_path, monkeypatch):
    from tests.helpers.recognition import Backend
    config, client, services, _ = run_setup(tmp_path, monkeypatch)
    class Model(Backend):
        def close(self): raise OSError('owned backend cleanup failed')
    services.backend = lambda *a: Model()
    with pytest.raises(OSError, match='cleanup failed'):
        run(config, services, '1')
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT status,error,summary FROM runs').fetchone()
        assert row['status'] == 'failed' and 'cleanup failed' in row['error']
        assert json.loads(row['summary'])['mutations']['cleanup_failures'] == 1


def test_rebuild_scratch_close_preserves_primary_failure(tmp_path, monkeypatch):
    import stash_ai.recognition.rebuild as rebuild
    config = cluster_config(tmp_path)
    with FaceDatabase(config.database) as db:
        selected, _ = scan(db, config, '1')
        connect = sqlite3.connect
        class Connection(sqlite3.Connection):
            def close(self):
                super().close()
                raise OSError('scratch close failed')
        monkeypatch.setattr(sqlite3, 'connect', lambda *a,**kw:connect(*a, **kw, factory=Connection))
        def fail(*a, **kw): raise RuntimeError('primary replay failure')
        monkeypatch.setattr(rebuild, 'cluster_scene', fail)
        with pytest.raises(RuntimeError, match='primary replay failure') as caught:
            rebuild_selected(db, [{'scene':'1','scan':selected}], config)
        assert any('scratch close failed' in note for note in caught.value.__notes__)
