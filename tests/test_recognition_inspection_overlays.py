"""Preview overlays remain source-mapped, read-only and independent of models."""
import json

import pytest
from PIL import Image

from stash_ai.recognition.inspection_overlays import COLORS, render
from tests.test_recognition_identity_inspector import scene_fixture, vtt_fixture

FACE = [120,220,40,40,130,230,150,230,140,240,130,250,150,250,.99]
BODY = dict(bounds=[100,200,200,400],keypoints=[[150,320],[-1,-1],[200,260]]+[[-1,-1]]*14)


def test_preview_overlay_maps_linked_face_and_selected_pose_and_keeps_clean_pixels():
    image=Image.new('RGB',(50,100),(20,30,40))
    before=image.tobytes()
    geometry=dict(bounds=[100,200,200,400],notes=[],shapes=[
        dict(kind='body',selected=True,geometry=BODY),dict(kind='face',selected=False,geometry=FACE)])
    annotated,caption=render(image,geometry)
    assert annotated.size==image.size and image.tobytes()==before
    assert annotated.getpixel((25,60))==COLORS['body']
    assert annotated.getpixel((0,50))==COLORS['body']
    assert annotated.getpixel((15,15))==COLORS['face']
    assert annotated.getpixel((10,20))==COLORS['face']
    assert annotated.getpixel((47,30))==(20,30,40)  # Outside pose point was not clamped to the border.
    assert 'Selected body: green (1 points)' in caption
    assert 'Linked face: yellow (5 points)' in caption


@pytest.mark.parametrize('bounds',[None,[0,0,0,10],[0,0,float('nan'),10],[0,'bad',5,10]])
def test_preview_without_known_crop_transform_stays_clean(bounds):
    image=Image.new('RGB',(50,100),'black')
    annotated,caption=render(image,dict(bounds=bounds,shapes=[],notes=[]))
    assert annotated is image
    assert 'source crop bounds missing or invalid' in caption


def test_box_survives_missing_landmarks_and_subpixel_crop_edge():
    image=Image.new('RGB',(10,10),'black')
    geometry=dict(bounds=[0,0,100,100],notes=[],shapes=[
        dict(kind='body',selected=True,geometry=dict(bounds=[99,0,100,100],keypoints='unavailable'))])
    annotated,caption=render(image,geometry)
    assert annotated.getpixel((9,5))==COLORS['body']
    assert '(0 points)' in caption


def test_pose_skeleton_and_coincident_face_markers_remain_distinguishable():
    image=Image.new('RGB',(100,100),'black')
    points=[[50,20]]+[[-1,-1]]*16
    points[5:7]=[[30,40],[70,40]]
    face=[40,10,20,20,50,20,-1,-1,-1,-1,-1,-1,-1,-1,.99]
    geometry=dict(bounds=[0,0,100,100],notes=[],shapes=[
        dict(kind='face',selected=False,geometry=face),
        dict(kind='body',selected=True,geometry=dict(bounds=[5,5,95,95],keypoints=points))])
    annotated,_=render(image,geometry)
    assert annotated.getpixel((50,40))==COLORS['body']  # Shoulder connection.
    assert annotated.getpixel((50,20))==COLORS['face']
    assert annotated.getpixel((53,20))==COLORS['body']  # Outer pose marker remains visible.
    assert annotated.getpixel((10,10))==(0,0,0)  # No skeleton edge to a missing point.


def linked_fixture(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.identity_inspection import IdentityReader
    config,person=scene_fixture(tmp_path)
    reader=IdentityReader(config,identity_id=person)
    face=reader.page(kind='face',scene='2')[0][0]['id']
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute("UPDATE observations SET geometry=?,quality=json_set(quality,'$.crop_bounds',json(?)) WHERE id=?",
                   (json.dumps(FACE),json.dumps([100,200,200,400]),face))
        source=db.execute('SELECT scan,cue FROM observations WHERE id=?',(face,)).fetchone()
        db.execute('UPDATE body_observations SET face=?,scan=?,cue=?,geometry=? WHERE id=?',
                   (face,source['scan'],source['cue'],json.dumps(BODY),'accepted-body'))
        db.commit()
    return config,reader,face


def test_readers_expose_only_selected_and_unambiguous_linked_geometry(tmp_path,monkeypatch):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.scene_inspection import SceneReader,UnassignedReader
    config,reader,face=linked_fixture(tmp_path)
    before=config.database.read_bytes()
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *_: pytest.fail('Loaded inference model'))
    for kind,address in (('face',face),('body','accepted-body')):
        detail=reader.detail(kind,address)
        preview=reader.preview(kind,address)
        assert detail['geometry']==preview['geometry']
        shapes=detail['geometry']['shapes']
        assert {s['kind'] for s in shapes}=={'face','body'}
        assert next(s for s in shapes if s['selected'])['kind']==kind
    unassigned=UnassignedReader(config,SceneReader(config,'2').catalogue()['scan'])
    assert unassigned.preview('body','accepted-body')['geometry']==reader.preview('body','accepted-body')['geometry']
    assert not unassigned.preview('face','missing')['geometry']['shapes']
    assert config.database.read_bytes()==before
    with FaceDatabase(config.database) as state:
        db=state.connection
        row=dict(db.execute("SELECT * FROM body_observations WHERE id='accepted-body'").fetchone())
        row['id']='other-pose'
        db.execute('INSERT INTO body_observations VALUES ('+','.join('?' for _ in row)+')',tuple(row.values()))
        db.commit()
    geometry=reader.preview('face',face)['geometry']
    assert [s['kind'] for s in geometry['shapes']]==['face']
    assert 'Ambiguous linked pose' in geometry['notes'][0]


def test_linked_pose_prefers_current_contract_and_refuses_wrong_cue_face(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    config,reader,face=linked_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        row=dict(db.execute("SELECT * FROM body_observations WHERE id='accepted-body'").fetchone())
        row.update(id='historical-pose',contract='historical',geometry='{}')
        db.execute('INSERT INTO body_observations VALUES ('+','.join('?' for _ in row)+')',tuple(row.values()))
        db.commit()
    assert next(s for s in reader.preview('face',face)['geometry']['shapes'] if s['kind']=='body')['geometry']==BODY
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE body_observations SET cue=999 WHERE id='accepted-body'")
        state.connection.commit()
    assert [s['kind'] for s in reader.preview('body','accepted-body')['geometry']['shapes']]==['body']


def test_vtt_transform_is_supplied_after_pixel_verification_before_annotation(tmp_path):
    from stash_ai.recognition.inspection_crops import VttPreviews
    from stash_ai.recognition.identity_window import Inspector
    config,reader,client,media,face,backend,closed=vtt_fixture(tmp_path)
    before=config.database.read_bytes()
    previews=VttPreviews(config,client_factory=lambda:client)
    # Exercise exactly the worker path without needing a graphical desktop.
    inspector=object.__new__(Inspector)
    inspector.previews=previews
    try:
        result=inspector.load_detail(reader,'face',face)
        assert result['geometry']['bounds']==[0,0,160,160]
        assert 'Reconstructed from VTT' in result['annotated'][1]
        assert 'Selected face: yellow' in result['annotated'][1]
        raw,annotated=result['image'][0],result['annotated'][0]
        assert raw.size==annotated.size and raw.tobytes()!=annotated.tobytes()
        calls=list(media.calls)
        reference=inspector.load_reference(reader,'body','rejected-body',result['scope'])
        assert 'Selected body: green' in reference['annotated'][1]
        assert media.calls==calls  # Same VTT sprite cache, no new inference.
        assert backend.calls==0 and config.database.read_bytes()==before
    finally:
        previews.close()
    assert closed==[True]


def test_vtt_fractional_bounds_use_actual_rounded_crop_viewport(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.inspection_crops import VttPreviews
    config,reader,client,_,_,_,_=vtt_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE body_observations SET geometry=? WHERE id='rejected-body'",
                                 (json.dumps(dict(bounds=[10.4,20.2,80.8,130.6])),))
        state.connection.commit()
    result=reader.detail('body','rejected-body')
    previews=VttPreviews(config,client_factory=lambda:client)
    try:
        image,_=previews.image('body','rejected-body',result['scope'],result['image'],geometry=result['geometry'])
        assert image.size==(71,111)
        assert result['geometry']['bounds']==[10,20,81,131]
    finally:
        previews.close()
