"""YAML must activate segmentation before semantic sharpness gates can be used."""
import re

import pytest
import yaml

from stash_ai.recognition.config import defaults, load
from stash_ai.recognition.config_layout import get, put, public_path, v2_document


def configured(tmp_path, data, version):
    path=tmp_path/'sharpness.yml'
    document=data if version==1 else v2_document(data)
    path.write_text(yaml.safe_dump(document))
    return load(path)


@pytest.mark.parametrize('version',[1,2])
@pytest.mark.parametrize('floor',[
    'detection.min_sharpness','clustering.min_auxiliary_sharpness','clustering.min_anchor_sharpness',
])
def test_face_floor_requires_visibility_and_names_the_yaml_paths(tmp_path,version,floor):
    data=defaults();data['detection'].update(min_sharpness=0,min_visible_face_fraction=0)
    put(data,floor,10)
    path=floor if version==1 else public_path(floor)
    with pytest.raises(ValueError,match=re.escape(path)) as raised:
        configured(tmp_path,data,version)
    visibility='detection.min_visible_face_fraction' if version==1 else 'face.quality.min_visible_face_fraction'
    assert visibility in str(raised.value)
    data['clustering']['min_anchor_visible_face_fraction']=.5
    assert get(configured(tmp_path,data,version).data,floor)==10


@pytest.mark.parametrize('version',[1,2])
@pytest.mark.parametrize('floor',['sapiensid.collection.min_sharpness','sapiensid.identities.anchor_min_sharpness'])
@pytest.mark.parametrize('broken',['backend','disabled','overlap'])
def test_body_floor_requires_pmpose_sam_on_every_person(tmp_path,version,floor,broken):
    data=defaults();body=data['sapiensid'];body['enabled']=True
    body['collection']['min_sharpness']=body['identities']['anchor_min_sharpness']=0
    put(data,floor,10)
    body['pose']['backend']='pmpose'
    body['pose']['refinement'].update(enabled=True,mode='all')
    if broken=='backend':
        body['pose']['backend']='vitpose_plus_large';body['pose']['refinement']['enabled']=False
    elif broken=='disabled':body['pose']['refinement']['enabled']=False
    else:body['pose']['refinement']['mode']='overlap'
    path=floor if version==1 else public_path(floor)
    with pytest.raises(ValueError,match=re.escape(path)) as raised:
        configured(tmp_path,data,version)
    prefix='sapiensid.pose' if version==1 else 'pose'
    assert prefix+'.refinement.mode: all' in str(raised.value)
    body['pose']['backend']='pmpose';body['pose']['refinement'].update(enabled=True,mode='all')
    assert configured(tmp_path,data,version).data['sapiensid']['enabled']


@pytest.mark.parametrize('version',[1,2])
def test_disabled_body_and_zero_floors_do_not_require_masks(tmp_path,version):
    data=defaults()
    assert not configured(tmp_path,data,version).data['sapiensid']['enabled']
    data['sapiensid']['enabled']=True
    data['sapiensid']['collection']['min_sharpness']=0
    data['sapiensid']['identities']['anchor_min_sharpness']=0
    data['detection'].update(min_sharpness=0,min_visible_face_fraction=0)
    assert configured(tmp_path,data,version).data['sapiensid']['pose']['backend']=='yolo'


@pytest.mark.parametrize('version',[1,2])
def test_portrait_sharpness_cannot_disable_its_faceocc_measurement(tmp_path,version):
    data=defaults();data['portraits']['weights']['visibility']=0
    with pytest.raises(ValueError,match='portraits.weights.visibility'):
        configured(tmp_path,data,version)
    data['portraits']['anchor_only']=True
    data['clustering']['min_anchor_visible_face_fraction']=.8
    assert configured(tmp_path,data,version).data['portraits']['anchor_only']


def test_partial_v2_defaults_supply_masks_and_explicit_opt_out_is_not_overwritten(tmp_path):
    path=tmp_path/'partial.yml';path.write_text('version: 2\n')
    pose=load(path).data['sapiensid']['pose']
    assert pose['backend']=='pmpose' and pose['refinement']['enabled'] and pose['refinement']['mode']=='all'
    for override in ('estimator: {backend: vitpose_plus_large}', 'refinement: {enabled: false}',
                     'refinement: {mode: overlap}'):
        path.write_text('version: 2\npose: {'+override+'}\n')
        with pytest.raises(ValueError,match='body.collection.min_sharpness'):
            load(path)
    path.write_text('version: 2\npose: {estimator: {backend: vitpose_plus_large}}\n'
                    'body: {collection: {min_sharpness: 0}}\n'
                    'matching: {body: {anchors: {min_sharpness: 0}}}\n')
    pose=load(path).data['sapiensid']['pose']
    assert pose['backend']=='vitpose_plus_large' and not pose['refinement']['enabled']


def test_explicit_incompatible_refinement_reports_the_public_backend_path(tmp_path):
    path=tmp_path/'bad.yml'
    path.write_text('version: 2\npose: {estimator: {backend: vitpose_plus_large}, refinement: {enabled: true}}\n')
    with pytest.raises(ValueError,match='pose.estimator.backend: pmpose'):
        load(path)
