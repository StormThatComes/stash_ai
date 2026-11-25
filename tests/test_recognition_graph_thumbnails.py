from io import BytesIO
from threading import Event
import pytest
from PIL import Image
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_inspection import IdentityReader
from stash_ai.recognition.inspection_graph import build
from stash_ai.recognition import inspection_graph_thumbnails as thumbnails
from stash_ai.recognition.inspection_runtime import Cancelled, local
from tests.test_recognition_identity_report import seed


def fixture(tmp_path,count=4):
    config,person,_=seed(tmp_path,count)
    return config,IdentityReader(config,identity_id=person)


def test_load_progress_small_crops_and_readonly_database(tmp_path):
    config,reader=fixture(tmp_path)
    graph=build(reader);before=config.database.read_bytes();progress=[]
    result=thumbnails.load(config,graph,lambda *p:progress.append(p))
    assert result['total']==4 and result['missing']==0
    assert progress==[(i,4) for i in range(5)]
    assert len(result['images'])==4
    assert all(image.size==(37,43) for image in result['images'].values())
    assert config.database.read_bytes()==before


def test_large_crop_is_downsized_and_missing_or_corrupt_crop_does_not_abort(tmp_path):
    config,reader=fixture(tmp_path)
    output=BytesIO()
    with Image.new('RGB',(640,320)) as image:image.save(output,'PNG')
    with FaceDatabase(config.database) as state:
        ids=[r[0] for r in state.connection.execute('SELECT id FROM observations ORDER BY id')]
        for identifier,blob in zip(ids,[output.getvalue(),None,b'broken',None]):
            state.connection.execute('UPDATE observations SET crop=? WHERE id=?',(blob,identifier))
        state.connection.commit()
    result=thumbnails.load(config,build(reader))
    assert result['total']==4 and result['missing']==3
    assert next(iter(result['images'].values())).size==(64,32)
    assert any('unreadable' in caption for caption in result['captions'].values())
    assert any('not retained' in caption for caption in result['captions'].values())


def test_unavailable_reference_is_not_looked_up(tmp_path,monkeypatch):
    config,reader=fixture(tmp_path)
    graph=build(reader)
    graph['nodes']=[dict(id='face:outside',type='observation',kind='face',observation='outside',availability='unavailable_in_scope')]
    monkeypatch.setattr(thumbnails,'_image',lambda *a:pytest.fail('Unavailable scoped reference was queried'))
    result=thumbnails.load(config,graph)
    assert not result['images'] and result['missing']==1
    assert 'scope' in result['captions']['face:outside']


def test_revision_change_before_or_during_decode_is_rejected(tmp_path,monkeypatch):
    config,reader=fixture(tmp_path);graph=build(reader)
    current=graph['revision']
    monkeypatch.setattr(thumbnails,'stamp',lambda _:('changed',))
    with pytest.raises(ValueError,match='changed'):thumbnails.load(config,graph)
    revisions=iter([current,('changed',)])
    monkeypatch.setattr(thumbnails,'stamp',lambda _:next(revisions))
    with pytest.raises(ValueError,match='changed'):thumbnails.load(config,graph)


def test_cancellation_is_checked_between_images(tmp_path,monkeypatch):
    config,reader=fixture(tmp_path);graph=build(reader)
    event=Event();monkeypatch.setattr(local,'cancel',event,raising=False)
    counts=[]
    def progress(done,total):
        counts.append(done)
        if done==1:event.set()
    with pytest.raises(Cancelled):thumbnails.load(config,graph,progress)
    assert counts==[0,1]


def test_thumbnail_node_budget_is_enforced(tmp_path):
    config,reader=fixture(tmp_path);graph=build(reader)
    graph['nodes']=[dict(type='observation')]*(thumbnails.MAX_NODES+1)
    with pytest.raises(ValueError,match='budget'):thumbnails.load(config,graph)
