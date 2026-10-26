from types import SimpleNamespace

import pytest
from PIL import Image

from stash_ai.recognition.person_audit import BOX_COLOR, POSE_COLOR, FACE_COLORS, export


@pytest.mark.parametrize('bounds,expected_box', [
    ([20,30,101,150], [60.8,0,320,384]),  # width 81: odd horizontal padding
    ([20,30,140,111], [0,60.8,384,320]),  # height 81: odd vertical padding
])
def test_person_overlays_use_source_and_exact_model_coordinates(tmp_path,bounds,expected_box):
    person=dict(bounds=bounds,keypoints=[[40,50]]+[[-1,-1]]*16)
    face=SimpleNamespace(landmarks=[[60,70]]*5)
    # Deliberately differs from source geometry: the overlay must use the
    # prepared model points, including canonicalization, not recompute them.
    points=[[.5,.5]]+[[-1,-1]]*16+[[.7,.4]]*5
    with Image.new('RGB',(180,180),(20,20,20)) as source,Image.new('RGB',(384,384),(30,30,30)) as model:
        before_source,before_model=source.tobytes(),model.tobytes()
        result=export(source,model,face,person,points,tmp_path/'sample')
        assert source.tobytes()==before_source and model.tobytes()==before_model
    assert result['annotations']['input_person_bounds']==pytest.approx(expected_box)
    assert result['annotations']['source_drawn_indices']==[0,17,18,19,20,21]
    assert result['annotations']['input_drawn_indices']==[0,17,18,19,20,21]
    with Image.open(tmp_path/result['source_file']) as source:
        assert source.getpixel((40,50))==POSE_COLOR
        assert source.getpixel((60,70))==FACE_COLORS[-1]
        assert source.getpixel((bounds[0],bounds[1]+10))==BOX_COLOR
        assert source.getpixel((0,0))==(20,20,20)
    with Image.open(tmp_path/result['annotated_input_file']) as annotated:
        assert annotated.getpixel((192,192))==POSE_COLOR
        assert annotated.getpixel((268,153))==FACE_COLORS[-1]
        assert annotated.getpixel((0,0))==(30,30,30)


def test_unavailable_association_does_not_invent_person_overlay(tmp_path):
    from stash_ai.recognition.person_collection import audit_input
    class Detector:
        def observation_quality(self,*args):
            return dict(identity_rejection='person_face_association_unavailable',person=None)
        def audit_input(self,*args):pytest.fail('No associated person to prepare')
    with Image.new('RGB',(100,100)) as image:
        result=audit_input(Detector(),image,None,tmp_path/'missing')
    assert result['identity_rejection']=='person_face_association_unavailable'
    assert not list(tmp_path.iterdir())


def test_audit_manifest_lists_all_person_images(tmp_path,monkeypatch):
    from tests.test_recognition_alignment_audit import config,Detector
    from tests.helpers.recognition import MediaClient
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.alignment_audit import audit_scene
    cfg=config(tmp_path)
    client=MediaClient(count=1)
    class PersonDetector(Detector):
        def audit_input(self,*a):pytest.fail('Only the export integration is exercised here')
    expected=dict(input_file='person-input.png',mask_file='person-mask.png',
                  source_file='person-source.png',annotated_input_file='person-input-annotated.png')
    monkeypatch.setattr('stash_ai.recognition.person_collection.audit_input',lambda *a:expected)
    with SpriteInput(client,client.scene,cfg) as source:
        result=audit_scene(source,PersonDetector(),cfg,'1',tmp_path)
    sample=result['samples'][0]
    assert sample['person_input']==expected
    assert all('scene-1/'+name in sample['images'] for name in expected.values())
