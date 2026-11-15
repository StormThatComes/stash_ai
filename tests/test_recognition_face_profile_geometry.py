import numpy as np
from PIL import Image
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.visibility import prepare_aligned
from stash_ai.recognition.inspection_face_visibility import profile_roi, FaceReplay


def test_short_eye_line_uses_nose_and_mouth_axis_including_quarter_turn():
    points = np.array([[48,40],[49,42],[49,51],[45,65],[53,65]])
    for rotation in (np.eye(2),np.array([[0,-1],[1,0]])):
        xy = (points-50) @ rotation.T + 50
        face = checked_detection([10,10,80,80,*xy.ravel(),.99])
        with Image.new('RGB',(100,100)) as image:
            _,region,total,affine = prepare_aligned(image,face,0.)
            axis = np.array([affine[0],affine[3]])
            assert np.dot(axis/np.linalg.norm(axis), rotation[:,0]) > .99
            alternate, alt_total, outline = profile_roi(face,image.size,affine)
            assert 0 < alt_total < total and alt_total == alternate.sum()
            logits = np.where(alternate,1.,-1.)
            replay = FaceReplay(image,'',face.box,logits,region,total,
                                (1,0,0,0,1,0),(alternate,alt_total,outline))
            for use_profile in (True,False):
                overlay, caption = replay.render(0.,use_profile)
                assert 'Landmark ROI 1.000' in caption
                overlay.close()
