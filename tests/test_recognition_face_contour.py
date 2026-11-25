"""Contour ownership, source coverage and occlusion have separate contracts."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stash_ai.recognition.face_contour import mesh_region, resolve
from stash_ai.recognition.visibility import (FaceVisibility, VisibilityGeometryError,
                                            prepare_measurement, measurement, device_measurements)
from tests.helpers.recognition import FACE, textured_image


def sample(*, region_mode='mesh'):
    import cv2
    with textured_image() as image:
        item = prepare_measurement(image, FACE, 0., region_mode=region_mode)
    inv = cv2.invertAffineTransform(np.asarray(item.affine).reshape(2, 3))
    anchors = (np.asarray(FACE.landmarks)+.5) @ inv[:, :2].T+inv[:, 2]
    points = np.zeros((478, 3))
    points[:, :2] = [128, 128]
    angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
    points[200:240, :2] = np.stack((128+70*np.cos(angles), 128+77*np.sin(angles)),axis=1)
    for indices, xy in zip(((33,133),(362,263),(1,),(61,),(291,)),anchors):
        points[list(indices),:2] = xy
    return item, points


def test_contour_ignores_iris_outliers_and_never_uses_logits():
    item, points = sample()
    points[468:, :2] = 10000
    region = mesh_region(points, 5., item)
    assert region.method == 'mesh' and region.total == region.mask.sum()
    assert region.support.sum() > region.total
    full = np.ones((1,1,256,256), np.float32)
    hidden = full.copy(); hidden[...,128:] = -1
    a = measurement(full, region.mask, 'cpu', face_pixels=region.total)
    b = measurement(hidden, region.mask, 'cpu', face_pixels=region.total)
    assert a['visible_face_fraction'] == 1.
    assert .45 < b['visible_face_fraction'] < .55
    assert a['visibility']['face_pixels'] == b['visibility']['face_pixels']


@pytest.mark.parametrize('damage', ['missing', 'other_face', 'collapsed', 'clipped'])
def test_rejected_mesh_is_unavailable_without_an_ellipse_score(damage):
    item, points = sample()
    presence = -1. if damage == 'missing' else 5.
    if damage == 'other_face': points[:,:2] += [60,0]
    if damage == 'collapsed': points[:,:2] = [128,128]
    if damage == 'clipped': points[200:240,0] = 300
    mesh = SimpleNamespace(region=lambda value: mesh_region(points,presence,value))
    with pytest.raises(VisibilityGeometryError):
        resolve(SimpleNamespace(mesh=mesh,region_requested='mesh',minimum_source_fraction=.85),item)


def test_padding_cannot_contribute_even_with_positive_logits():
    item, points = sample()
    item.source_size = (60,160)
    region = mesh_region(points,5.,item)
    assert 0 < region.mask.sum() < region.total
    assert np.all(region.mask <= region.support)
    score = measurement(np.ones((1,1,256,256)),region.mask,'cpu',face_pixels=region.total)
    assert 0 < score['visible_face_fraction'] < .85
    with pytest.raises(VisibilityGeometryError,match='real source'):
        resolve(SimpleNamespace(mesh=SimpleNamespace(region=lambda _:region),minimum_source_fraction=.85), item)


def test_separate_sharpness_mask_has_identical_numpy_and_torch_results():
    item, points = sample()
    region = mesh_region(points,5.,item)
    logits = np.ones((1,1,256,256),np.float32)
    cpu = measurement(logits,region.mask,'cpu',blob=item[0],face_pixels=region.total,
                      sharpness_input=item.sharpness_input,sharpness_support=region.support)
    device = device_measurements(torch.from_numpy(logits),[region.mask],'cpu',
        inputs=torch.from_numpy(item[0]),face_pixels=[region.total],
        sharpness_inputs=[item.sharpness_input],sharpness_supports=[region.support])[0]
    assert cpu['visibility'] == device['visibility']
    assert cpu['sharpness_pixels'] == device['sharpness_pixels']
    assert cpu['sharpness'] == pytest.approx(device['sharpness'],abs=.001)
    for key in ('focus_sharpness', 'motion_sharpness', 'sharpness_confidence'):
        assert cpu[key] == pytest.approx(device[key], abs=.001) if device[key] is not None else cpu[key] is None
    assert cpu['sharpness_split_method'] == device['sharpness_split_method'] == 'edge-spread-covariance-v2-equal-span'
    assert cpu['sharpness_split_reason'] == device['sharpness_split_reason']
    for key in ('edge_counts', 'tile_counts'):
        assert cpu['sharpness_split_diagnostics'][key] == device['sharpness_split_diagnostics'][key]


def test_model_owner_keeps_mesh_failure_local_to_observation():
    item, points = sample()
    model = object.__new__(FaceVisibility)
    model.model = lambda x: torch.ones_like(x[:,:1])
    model.mesh = SimpleNamespace(region=lambda value:mesh_region(points,5.,value))
    model.device='cpu'; model.minimum_source_fraction=.85; model.logit_threshold=0.
    result = model.measure_prepared_many([item])[0]
    assert result['visibility']['region_method']=='mesh'
    assert result['visible_face_fraction']==1.
    # A second prepared observation can fail while the first remains usable.
    with textured_image() as image:
        other = prepare_measurement(image, FACE, 0., region_mode='mesh')
    model.mesh.region=lambda value:mesh_region(points,-1. if value is other else 5.,item)
    results = model.measure_prepared_many([item,other])
    assert results[0]['visibility']['region_method']=='mesh'
    assert isinstance(results[1], VisibilityGeometryError)


def test_region_config_roundtrip_and_invalid_value(tmp_path):
    from stash_ai.recognition.config import load
    path=tmp_path/'config.yml'
    for name in ['mesh','ellipse']:
        path.write_text(f'version: 2\nface:\n  quality:\n    visibility_region: {name}\n')
        assert load(path).data['detection']['visibility_region']==name
    path.write_text('version: 2\nface:\n  quality:\n    visibility_region: mask\n')
    with pytest.raises(ValueError,match='visibility_region'):
        load(path)


def test_changing_region_mode_invalidates_retained_measurement_and_scan(tmp_path):
    from dataclasses import replace
    from stash_ai.recognition.config import load
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.visibility import CONTRACT_ID, cached
    config=load(root=tmp_path)
    ellipse=replace(config,data=dict(config.data,detection=dict(config.data['detection'],visibility_region='ellipse')))
    assert scan_identity(config,'ns') != scan_identity(ellipse,'ns')
    quality={'visible_face_fraction':.9,'visibility':{'contract':CONTRACT_ID,'region_requested':'ellipse'}}
    assert cached(quality,region='ellipse') and not cached(quality,region='mesh')


def test_inspector_compares_contour_and_ellipse_without_new_inference():
    from PIL import Image
    from stash_ai.recognition.inspection_face_visibility import FaceReplay
    item, points=sample(region_mode='ellipse')
    contour=mesh_region(points,5.,item)
    replay=FaceReplay(Image.new('RGB',(256,256)), 'Recorded', (0,0,256,256),
                      np.ones((256,256)), item[1], item[2], (1,0,0,0,1,0), contour=contour)
    first,caption=replay.render(0.,use_contour=True,show_points=True)
    assert f'{contour.total:,} contour pixels' in caption
    second,caption=replay.render(0.,use_contour=False)
    assert f'{item[2]:,} ellipse pixels' in caption
    assert first.tobytes()!=second.tobytes()
    first.close();second.close();replay.image.close()


def test_mesh_source_coverage_uses_its_own_denominator():
    _,points=sample()
    with textured_image() as original, original.crop((0,0,111,160)) as image:
        item=prepare_measurement(image,FACE,0.,region_mode='mesh')
    region=mesh_region(points,5.,item)
    contour_fraction=region.mask.sum()/region.total
    assert 0 < contour_fraction < 1
    model=SimpleNamespace(mesh=SimpleNamespace(region=lambda _:region),minimum_source_fraction=contour_fraction)
    result=resolve(model,item)
    assert result.method=='mesh'
    model.minimum_source_fraction=np.nextafter(contour_fraction,1.)
    with pytest.raises(VisibilityGeometryError):
        resolve(model,item)


def test_mesh_preparation_and_area_do_not_depend_on_old_ellipse(monkeypatch):
    from stash_ai.recognition import face_region
    from stash_ai.recognition.models import checked_detection
    def forbidden(*args, **kwargs):
        pytest.fail('Mesh must not fit or validate against the old ellipse.')
    monkeypatch.setattr(face_region,'fit',forbidden)
    item, points=sample()
    assert item[1] is None and item[2] == 0
    # A bad nose previously dragged the template or failed association outright.
    row=list(FACE.row); row[8] += 35; row[9] -= 20
    with textured_image() as image:
        changed=prepare_measurement(image,checked_detection(row),0.,region_mode='mesh')
    np.testing.assert_array_equal(item[0],changed[0])
    assert mesh_region(points,5.,changed).total == mesh_region(points,5.,item).total


@pytest.mark.parametrize('turns',[1,2,3])
def test_mesh_box_crop_rotates_without_mirroring(turns):
    from tests.test_recognition_visibility_rotation import quarter_turn
    with textured_image() as original:
        expected=prepare_measurement(original,FACE,0.,region_mode='mesh')
        rotated,face=quarter_turn(original,FACE,turns)
        with rotated:
            actual=prepare_measurement(rotated,face,0.,region_mode='mesh')
    np.testing.assert_array_equal(actual[0],expected[0])


def test_missing_mesh_model_or_metadata_cannot_select_ellipse():
    from stash_ai.recognition.visibility import VisibilityError
    item,_=sample(region_mode='ellipse')
    with pytest.raises(VisibilityError,match='fallback is disabled'):
        resolve(SimpleNamespace(region_requested='mesh',mesh=None),item)
    with pytest.raises(VisibilityGeometryError,match='no ellipse fallback'):
        resolve(SimpleNamespace(region_requested='mesh',mesh=object()),tuple(item))
    with pytest.raises(VisibilityGeometryError,match='box-based crop'):
        resolve(SimpleNamespace(region_requested='mesh',mesh=object()),item)


def test_old_fallback_contract_is_invalidated():
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.visibility import CONTRACT, cached
    old = dict(CONTRACT, version=10, crop='landmark-template-square-padding-profile-roll-aligned-v1',
               region='mesh-envelope-or-landmark-ellipse-v1')
    quality = dict(visible_face_fraction=.9, visibility=dict(contract=digest(old),
                   region_requested='mesh', region_method='ellipse'))
    assert not cached(quality, region='mesh')
