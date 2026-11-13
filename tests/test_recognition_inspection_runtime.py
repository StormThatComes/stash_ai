import threading
import time

from stash_ai.recognition.inspection_runtime import Worker, Cache, check_cancelled
from stash_ai.recognition.identity_inspection import IdentityReader
from tests.test_recognition_identity_inspector import scene_fixture


def test_latest_job_does_not_wait_for_old_io_and_close_never_joins():
    started,release=threading.Event(),threading.Event()
    values=[]
    worker=Worker('test-inspector',workers=2)
    def slow():
        started.set();release.wait(3);check_cancelled();return 'old'
    worker.request(slow,values.append,values.append)
    assert started.wait(1)
    worker.request(lambda:'new',values.append,values.append)
    deadline=time.monotonic()+2
    while not values and time.monotonic()<deadline:
        worker.poll();time.sleep(.005)
    assert values==['new']
    start=time.monotonic();worker.close()
    assert time.monotonic()-start<.2
    release.set()
    worker.poll()
    assert values==['new']


def test_cache_is_bounded_and_returns_independent_values():
    cache=Cache(100,entries=2)
    cache.put('a',['value']);value=cache.get('a');value.append('changed')
    assert cache.get('a')==['value']
    cache.put('b','b');cache.put('c','c')
    assert cache.get('a') is None
    cache.put('huge','x'*1000)
    assert cache.get('huge') is None and cache.size<=100


def test_metadata_precedes_pixels_and_cursor_pages_keep_all_members(tmp_path,monkeypatch):
    from stash_ai.recognition import identity_inspection
    from stash_ai.recognition.database import FaceDatabase
    config,person=scene_fixture(tmp_path)
    reader=IdentityReader(config,identity_id=person)
    monkeypatch.setattr(identity_inspection,'PAGE_SIZE',2)
    rows=[];cursor=None
    while True:
        page,more=reader.page(kind='face',after=cursor)
        rows.extend(page)
        if not more:break
        cursor=page[-1]
    assert len(rows)==len({r['id'] for r in rows})==6
    monkeypatch.setattr(identity_inspection,'_image',lambda *_: (_ for _ in ()).throw(AssertionError('Decoded pixels')))
    assert reader.detail('face',rows[0]['id'],pixels=False)['image'][0] is None
    assert sum(r['count'] for r in reader.states('face'))==6
    with FaceDatabase(config.database) as db:
        db.connection.execute('DELETE FROM members WHERE observation=?',(rows[0]['id'],));db.connection.commit()
    assert sum(r['count'] for r in reader.states('face'))==5
