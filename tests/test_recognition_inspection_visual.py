"""Context uses retained geometry and verified pixels, never inference or tracking."""
import json
from dataclasses import replace
from PIL import Image
import pytest
from tests.test_recognition_identity_inspector import vtt_fixture, scene_fixture
from stash_ai.recognition.inspection_crops import VttPreviews
from stash_ai.recognition.inspection_context import locate, cue_geometry
from stash_ai.recognition.inspection_overlays import render, point_info, COLORS
from stash_ai.recognition.database import FaceDatabase


def test_full_cue_keeps_all_retained_detections_and_original_pixel_check(tmp_path):
    config,reader,client,media,face,backend,_=vtt_fixture(tmp_path)
    preview=VttPreviews(config,client_factory=lambda:client)
    result=reader.detail('body','rejected-body')
    before=config.database.read_bytes()
    try:
        image,caption=preview.image('body','rejected-body',result['scope'],result['image'],full=True,geometry=result['geometry'])
        assert image.size==(160,160)
        assert result['geometry']['bounds']==[0,0,160,160]
        assert {(s['kind'],s['id']) for s in result['geometry']['shapes']}=={('face',face),('body','rejected-body')}
        assert result['geometry']['cue_count']==1 and 'source verified' in caption
        assert config.database.read_bytes()==before and backend.calls==0
        with FaceDatabase(config.database) as state:
            state.connection.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.pixel_digest','bad')")
            state.connection.commit()
        # A retained crop must not bypass full-frame source/signature checks.
        image,caption=preview.image('body','rejected-body',result['scope'],(Image.new('RGB',(2,2)),''),full=True)
        assert image is None and 'recorded signature' in caption
    finally:preview.close()


def test_nearby_cue_uses_vtt_order_and_actual_timestamps_without_tracking(tmp_path):
    from stash_ai.recognition.media import SpriteInput
    config,reader,client,_,face,backend,_=vtt_fixture(tmp_path)
    def source(*args):
        original=SpriteInput(*args)
        original.cues=list(original.cues)+[replace(original.cues[0],start=137.5)]
        thumbnail=original.thumbnail
        original.thumbnail=lambda i:thumbnail(0)
        return original
    preview=VttPreviews(config,client_factory=lambda:client,source_factory=source)
    result=reader.detail('face',face)
    try:
        image,caption=preview.image('face',face,result['scope'],result['image'],full=True,cue=1,geometry=result['geometry'])
        assert image is not None
        assert result['geometry']['cue_index']==1 and result['geometry']['cue_seconds']==137.5
        assert result['geometry']['shapes']==[]  # No guessed identity across cues.
        assert backend.calls==0
    finally:preview.close()


def test_context_navigation_is_scope_bound_and_handles_unassigned(tmp_path):
    config,person=scene_fixture(tmp_path)
    from stash_ai.recognition.scene_inspection import SceneReader
    scan=SceneReader(config,'2').catalogue()['scan']
    for kind,address in [('body','accepted-body'),('body','rejected-body'),('face','rejected-face')]:
        row=locate(config,kind,address,scan['scope'])
        assert row['id']==address
        result=row['reader'].detail(kind,address,pixels=False)
        assert result['scope']==scan['scope'] and result['image'][0] is None
    with pytest.raises(ValueError,match='scope/model'):locate(config,'face','rejected-face','wrong')
    with pytest.raises(ValueError,match='Kind'):locate(config,'pose','rejected-face',scan['scope'])


def test_layers_independent_and_point_info_does_not_invent_confidence():
    points=[[20,20]]+[[-1,-1]]*16
    points[5:7]=[[10,40],[50,40]]
    geometry=dict(bounds=[0,0,100,100],notes=[],shapes=[dict(kind='body',selected=True,
        geometry=dict(bounds=[5,5,95,95],keypoints=points,keypoint_scores=[.8]))])
    image=Image.new('RGB',(100,100),'black')
    annotated,_=render(image,geometry,dict(skeleton=True))
    assert annotated.getpixel((30,40))==COLORS['body']
    assert annotated.getpixel((20,20))==(0,0,0) and annotated.getpixel((5,5))==(0,0,0)
    assert 'nose' in point_info(geometry,image.size,20,20) and '0.8' in point_info(geometry,image.size,20,20)
    assert 'not retained' in point_info(geometry,image.size,10,40)
    _,caption=render(image,geometry,dict(masks=True))
    assert 'No usable retained mask' in caption
    geometry['shapes'][0]['geometry']['keypoints']=None
    assert not point_info(geometry,image.size,10,40)


def test_retained_mask_maps_to_viewport_without_modifying_raw_pixels():
    import base64
    import numpy as np
    mask=base64.b64encode(np.packbits(np.ones((10,10),dtype=bool)).tobytes()).decode()
    image=Image.new('RGB',(30,30),'black')
    geometry=dict(bounds=[0,0,30,30],notes=[],shapes=[dict(kind='body',selected=True,
        geometry=dict(bounds=[10,10,20,20],pose_refinement=dict(encoder_mask=mask)))])
    annotated,caption=render(image,geometry,dict(masks=True))
    assert annotated.getpixel((15,15))!=(0,0,0) and annotated.getpixel((5,5))==(0,0,0)
    assert image.getpixel((15,15))==(0,0,0) and 'No usable retained mask' not in caption
