import numpy as np
from PIL import Image
import pytest
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.visibility import prepare_aligned, measurement
from stash_ai.recognition.inspection_face_visibility import legacy_roi, FaceReplay
from stash_ai.recognition.face_region import outline


def test_short_eye_line_uses_nose_and_mouth_axis_including_quarter_turn():
    points = np.array([[48,40],[49,42],[49,51],[45,65],[53,65]])
    for rotation in (np.eye(2),np.array([[0,-1],[1,0]])):
        xy = (points-50) @ rotation.T + 50
        face = checked_detection([10,10,80,80,*xy.ravel(),.99])
        with Image.new('RGB',(100,100)) as image:
            _,region,total,affine = prepare_aligned(image,face,0.)
            axis = np.array([affine[0],affine[3]])
            assert np.dot(axis/np.linalg.norm(axis), rotation[:,0]) > .99
            alternate, alt_total, old_outline = legacy_roi(face,image.size,affine)
            assert 0 < alt_total and alt_total == alternate.sum()
            logits = np.where(alternate,1.,-1.)
            replay = FaceReplay(image,'',face.box,logits,region,total,
                                (1,0,0,0,1,0),(alternate,alt_total,old_outline))
            for use_legacy in (True,False):
                overlay, caption = replay.render(0.,use_legacy)
                assert 'Legacy box ellipse 1.000' in caption
                overlay.close()


def tilted_face(angle, profile=1, center=(160,160)):
    """Independent full contour: 80x140 face with features on its profile edge."""
    c,s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    rotation = np.array([[c,-s],[s,c]])
    if profile:
        points = np.array([[26*profile-1,-28],[26*profile+1,-28],
                           [35*profile,-3],[26*profile-3,28],[26*profile+3,28]])
    else:
        points = np.array([[-20,-28],[20,-28],[0,-3],[-16,28],[16,28]])
    points = points @ rotation.T + center
    # Extents of the full facial contour after rotation; not a fixed square
    # box or an ellipse reconstructed from the implementation being tested.
    w,h = 2*np.sqrt((40*c)**2+(70*s)**2), 2*np.sqrt((40*s)**2+(70*c)**2)
    face = checked_detection([center[0]-w/2,center[1]-h/2,w,h,*points.ravel(),.99])
    return face, rotation


@pytest.mark.parametrize('angle', [-150,-135,-120,-60,-45,-30,0,30,45,60,90,135,180])
@pytest.mark.parametrize('profile', [-1,0,1])
def test_tilted_full_face_keeps_forehead_chin_and_both_cheeks(angle, profile):
    face, rotation = tilted_face(angle,profile)
    with Image.new('RGB',(320,320)) as image:
        blob,region,total,affine = prepare_aligned(image,face,1.)
        old,old_total,_ = legacy_roi(face,image.size,affine)
    inverse = np.linalg.inv([affine[:3],affine[3:],(0,0,1)])
    # These anatomical points are deliberately outside the central five-point
    # cloud. Both profile directions must preserve the far cheek as well.
    anatomical = np.array([[0,-60],[0,60],[-30,0],[30,0]]) @ rotation.T + 160
    pixels = np.c_[anatomical,np.ones(4)] @ inverse.T
    xx,yy = np.floor(pixels[:,:2]).astype(int).T
    assert region[yy,xx].all()
    rows,columns = np.indices((256,256)) + .5
    sx = affine[0]*columns+affine[1]*rows+affine[2]
    sy = affine[3]*columns+affine[4]*rows+affine[5]
    local = np.stack((sx-160,sy-160),axis=-1) @ rotation
    truth = (local[...,0]/40)**2+(local[...,1]/70)**2 <= 1
    logits = np.where(truth,1.,-1.).astype('float32')[None,None]
    score = measurement(logits,region,'cpu',face_pixels=total)['visible_face_fraction']
    assert score > .99
    assert np.count_nonzero(truth & region)/truth.sum() > .85
    if abs(angle) == 45:
        assert not old[yy[:2],xx[:2]].any()  # Reproduces the reported lost chin/forehead.
        assert measurement(logits,old,'cpu',face_pixels=old_total)['visible_face_fraction'] < .9
    # A truly covered lower face must still lower visibility. The region is
    # fixed before seeing logits and cannot shrink to the remaining skin.
    covered = logits.copy()
    covered[0,0,local[...,1]>5] = -1
    assert measurement(covered,region,'cpu',face_pixels=total)['visible_face_fraction'] < .6


@pytest.mark.parametrize('angle', [44.9,45.,45.1,134.9,135.,135.1])
def test_diagonal_box_ambiguity_does_not_collapse_or_expand_region(angle):
    from stash_ai.recognition.face_region import ellipse
    face,rotation = tilted_face(angle)
    center,axis,radii = ellipse(face)
    np.testing.assert_allclose(axis,rotation[:,0],atol=1e-10)
    np.testing.assert_allclose(radii,[36,67.2],rtol=.001)
    assert center == pytest.approx((160,160))


def test_compressed_profile_features_cannot_shrink_the_diagonal_face_to_their_cloud():
    from stash_ai.recognition.face_region import ellipse
    face,_ = tilted_face(45)
    points = np.asarray(face.landmarks)
    # Simulate compressed features while retaining the independent full box.
    points = (points-points.mean(axis=0))*.4 + points.mean(axis=0)
    face = checked_detection([*face.box,*points.ravel(),.99])
    _,_,(rx,ry) = ellipse(face)
    w,h = face.box[2:]
    assert (rx/.45)**2+(ry/.48)**2 == pytest.approx(w*w+h*h)
    assert ry/.48 >= rx/.45 - 1e-6


@pytest.mark.parametrize('threshold', [0.,-.2])
def test_previous_axis_aligned_measurements_are_not_reused(threshold):
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.visibility import CONTRACT, cached
    old = dict(CONTRACT,version=7,region='source-face-box-ellipse-0.45w-0.48h-inverse-mapped')
    if threshold:
        old['mask_threshold'] = 'logit>record.logit_threshold'
    quality = dict(visible_face_fraction=.9,
                   visibility=dict(contract=digest(old),logit_threshold=threshold))
    assert not cached(quality,threshold)


def test_foreground_outside_ellipse_is_diagnostic_and_does_not_change_score():
    face,_ = tilted_face(45)
    with Image.new('RGB',(320,320),(50,50,50)) as image:
        _,region,total,affine = prepare_aligned(image,face,1.)
        inverse = tuple(np.linalg.inv([affine[:3],affine[3:],(0,0,1)])[:2].ravel())
        logits = np.ones((256,256),dtype='float32')
        replay = FaceReplay(image,'',face.box,logits,region,total,inverse,
                            legacy_roi(face,image.size,affine),outline(face))
        normal,caption = replay.render(0.)
        raw,raw_caption = replay.render(0.,show_outside=True)
        # Point is within the model crop, but outside the face ellipse.
        assert normal.getpixel((160,230)) == image.getpixel((160,230))
        assert raw.getpixel((160,230)) != image.getpixel((160,230))
        assert raw_caption.startswith(caption) and 'Replay 1.000' in caption
        assert 'not counted' in raw_caption
        normal.close();raw.close()


@pytest.mark.parametrize('angle', [-60,-45,-30,30,45,60])
def test_partial_tilted_face_keeps_full_denominator_and_cpu_device_agreement(angle):
    import torch
    from stash_ai.recognition.visibility import device_measurements, VisibilityGeometryError
    face,_ = tilted_face(angle,center=(5,160))
    with Image.new('RGB',(320,320)) as image:
        blob,region,total,_ = prepare_aligned(image,face,0.)
        with pytest.raises(VisibilityGeometryError):
            prepare_aligned(image,face,1.)
    logits = np.ones((1,1,256,256),dtype='float32')
    cpu = measurement(logits,region,'cpu',face_pixels=total,blob=blob)
    device, = device_measurements(torch.from_numpy(logits),[region],'cpu',
                                 face_pixels=[total],inputs=torch.from_numpy(blob))
    assert cpu['visible_face_fraction'] == device['visible_face_fraction'] == region.sum()/total
    assert .5 < cpu['visible_face_fraction'] < .65
    assert cpu['visibility'] == device['visibility']
    assert cpu['sharpness_pixels'] == device['sharpness_pixels']
