"""Foreground admission, conservative fallback and transient mask lifecycle."""
import base64
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.body_collection import collect, public_person, quality
from stash_ai.recognition.body_overlap import measure
from stash_ai.recognition.config import defaults, digest, plain


def person(bounds, mask=None, **refinement):
    left, top, right, bottom = bounds
    value = dict(bounds=list(bounds), keypoints=[[left+2,top+2]]*17, score=.99)
    if mask is not None:
        value['pose_refinement'] = dict(status='refined', bounds=list(bounds),
            overlap_mask=base64.b64encode(np.packbits(mask).tobytes()).decode(), **refinement)
    return value


def split_people():
    mask = np.indices((16,16))[1] < 8
    return [person([0,0,16,16], mask, separated=True),
            person([0,0,16,16], ~mask, separated=True)]


def test_disjoint_masks_survive_identical_boxes_and_the_actual_collection_gate():
    people = split_people()
    data = defaults()
    data['execution']['scan_cpu_workers'] = 1
    data['sapiensid']['collection'].update(min_person_pixels=1, min_sharpness=0, min_keypoints=1)
    items = []
    model = SimpleNamespace(frame_people=people, face_matches=[], person_geometry=lambda *a:('input',))
    pipeline = SimpleNamespace(collect_items=lambda db,image,rows:items.extend(rows))
    with Image.new('RGB',(16,16)) as image:
        collect(pipeline,None,model,SimpleNamespace(data=data),
                '00000000-0000-0000-0000-000000000001',0,0,image,[],[])
    records = [record[1] for record,value in items if record[0] is not None]
    assert len(records) == 2
    for record in records:
        assert record['identity_rejection'] is None
        assert record['quality']['overlap'] == 0
        assert record['quality']['box_overlap'] == 1
        assert record['quality']['overlap_basis'] == 'mask'
        assert not record['quality'].get('query_only')
        assert 'overlap_mask' not in record['person']['pose_refinement']
    assert all('overlap_mask' in p['pose_refinement'] for p in people)


def test_shifted_crop_coordinates_directional_area_and_exact_threshold():
    # A covers x=3..6 at y=3..6; B covers x=5..6 at y=4..7.
    mask_a = np.zeros((6,6),bool); mask_a[1:5,1:5] = True
    mask_b = np.zeros((6,6),bool); mask_b[:4,1:3] = True
    a, b = person([2,2,8,8],mask_a), person([4,4,10,10],mask_b)
    assert measure(a,[a,b])['overlap'] == 6/16
    assert measure(b,[a,b])['overlap'] == 6/8
    settings = dict(min_person_pixels=1,min_sharpness=0,min_keypoints=1,max_overlap=6/16)
    with Image.new('RGB',(12,12)) as image:
        assert quality(image,a,[a,b],settings)[1] is None
        settings['max_overlap'] = .37
        assert quality(image,a,[a,b],settings)[1] == 'person_overlap'


@pytest.mark.parametrize('missing', ['absent','empty','rejected','conflict','moved'])
@pytest.mark.parametrize('side', [0,1])
def test_unusable_mask_on_either_side_falls_back_to_boxes(missing,side):
    people = split_people(); details = people[side]['pose_refinement']
    if missing == 'absent': details.pop('overlap_mask')
    if missing == 'empty': details['overlap_mask'] = base64.b64encode(bytes(32)).decode()
    if missing == 'rejected': details['status'] = 'rejected'
    if missing == 'conflict': details['mask_conflict'] = True
    if missing == 'moved': details['bounds'] = [1,0,17,16]
    assert measure(people[0],people) == dict(overlap=1.,box_overlap=1.,overlap_basis='box')


def test_mixed_pairs_take_maximum_and_do_not_hide_a_maskless_peer():
    a,b = split_people()
    c = person([8,0,24,16])
    assert measure(a,[a,b,c]) == dict(overlap=.5,box_overlap=1.,overlap_basis='mixed')
    assert measure(a,[a,c,b]) == measure(a,[a,b,c])
    assert measure(a,[a]) == dict(overlap=0.,box_overlap=0.,overlap_basis='none')


def test_nonintersecting_boxes_need_no_mask_decode_and_corruption_is_not_silently_accepted():
    a,b = split_people(); b['pose_refinement']['overlap_mask'] = 'corrupt'
    with pytest.raises(ValueError): measure(a,[a,b])
    b['bounds'] = [16,0,32,16]
    assert measure(a,[a,b])['overlap'] == 0


def test_public_geometry_removes_both_masks_without_mutating_frame_evidence():
    a,b = split_people()
    a['pose_refinement']['encoder_mask'] = a['pose_refinement']['overlap_mask']
    public = public_person(a)
    assert not {'encoder_mask','overlap_mask'}.intersection(public['pose_refinement'])
    assert {'encoder_mask','overlap_mask'} <= a['pose_refinement'].keys()


@pytest.mark.parametrize('masked_crop', [False,True])
def test_contract_refreshes_old_overlap_but_keeps_legacy_banks_recognizable(masked_crop):
    from stash_ai.recognition.inference_contracts import pose as pose_contract
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract, compatible_contracts
    data=defaults(); pose=data['sapiensid']['pose']; pose['backend']='pmpose'
    pose['refinement'].update(enabled=True,masked_identity_crop=masked_crop)
    config=SimpleNamespace(data=data)
    previous=person_contract(config,legacy_mask_overlap=True)
    assert pose_contract(config)[1]['quality_mask'] == 'crop-bits-v1'
    collection=plain(data['sapiensid']['collection']);collection.pop('enabled')
    old_body=digest(['independent-person-v4-independent-torso',previous,collection])
    assert person_contract(config) != previous
    assert contract(config) != old_body
    assert compatible_contracts(config)[0] == contract(config)
    assert old_body in compatible_contracts(config)[1:]
    pose['refinement'].update(enabled=False,masked_identity_crop=False)
    assert 'quality_mask' not in pose_contract(config)[1]
    assert person_contract(config) == person_contract(config,legacy_mask_overlap=True)
    assert contract(config) == digest(['independent-person-v4-independent-torso',person_contract(config),collection])


def test_old_box_cues_are_recollected_and_only_scalar_overlap_is_persisted(tmp_path,monkeypatch):
    import json
    from tests.test_recognition_body_collection import Persons, settings
    from tests.helpers.recognition import Backend, MediaClient
    from stash_ai.recognition.body_observations import _contract, contract, coverage
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.scanning import scan_scene
    monkeypatch.setattr('stash_ai.recognition.sapiensid_v1.prepare_person_input',lambda *a:(np.zeros(1),)*3)
    config=settings(tmp_path)
    config.data['sapiensid']['collection']['max_overlap']=.65
    config.data['sapiensid']['pose']['backend']='pmpose'
    config.data['sapiensid']['pose']['refinement']['enabled']=True
    class MaskedPersons(Persons):
        def detect(self,image,faces=None):
            super().detect(image,faces)
            for index,p in enumerate(self.frame_people):
                mask=np.indices((160,160))[1]<80
                if index:mask=~mask
                p['pose_refinement']=person(p['bounds'],mask,separated=True)['pose_refinement']
            return faces
    client=MediaClient();model=MaskedPersons();backend=Backend();backend.detect=lambda image:[]
    collector=PersonCollector(config,{},lambda *a:model)
    try:
        with FaceDatabase(config.database) as state,SpriteInput(client,client.scene,config) as source:
            db=state.connection
            scan,_=scan_scene(state,'scope',client.scene,config,'ns',source,lambda:backend)
            legacy=_contract(config,'independent-person-v4-independent-torso')
            assert legacy!=contract(config)
            db.executemany('INSERT INTO body_cues VALUES (?,?,?)',[(scan,legacy,i) for i in range(3)])
            db.commit()
            collector.enrich(db,scan,source)
            assert (model.calls,model.embedded)==(3,6)
            assert coverage(db,config,[scan])['ready']==6
            for row in db.execute('SELECT geometry,quality FROM body_observations WHERE contract=?',(contract(config),)):
                assert 'overlap_mask' not in row['geometry']+row['quality']
                quality=json.loads(row['quality'])['quality']
                assert (quality['overlap'],quality['box_overlap'],quality['overlap_basis'])==(0.,1.,'mask')
            collector.enrich(db,scan,source)
            assert model.calls==3
    finally:
        collector.close()
