"""Selective full-frame RTMO-S second pass for the labelled contact benchmark."""
import math

import numpy as np

from stash_ai.recognition.crowd_context import pair
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces, associate


MODEL_URL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
             'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip')
MODEL_SHA = 'd0703d40d19f3921da51ae725402d5fdae4d2478c7442072d3101bd396f370d8'


class RTMO:
    def __init__(self, config):
        from stash_ai.recognition.sapiensid_downloads import install
        from stash_ai.recognition.sapiensid_loading import checked_bytes
        from stash_ai.recognition.runtime import session
        path = config.model_dir/'rtmo_s'/MODEL_SHA/'end2end.onnx'
        if not path.is_file():
            if not config.data['models']['download_missing']:
                raise FileNotFoundError(f'Missing RTMO model {path}. Enable models.download_missing to fetch it.')
            install(MODEL_URL,path,39617685,MODEL_SHA,byte_range=(783,36440719),archive_size=36442532)
        model = checked_bytes(path,MODEL_SHA,limit=39617685)
        self.session,self.device = session(config,model)
        self.input = self.session.get_inputs()[0].name

    def predict(self, image, settings):
        import cv2
        width,height = image.size
        scale = min(640/width,640/height)
        with image.convert('RGB') as rgb:
            resized = cv2.resize(np.asarray(rgb)[:,:,::-1],
                                 (max(1,int(width*scale)),max(1,int(height*scale))))
        pixels = np.full((640,640,3),114,dtype='float32')
        h,w = resized.shape[:2]
        pixels[:h,:w] = resized
        batch = np.ascontiguousarray(pixels.transpose(2,0,1)[None])
        boxes,poses = self.session.run(None,{self.input:batch})
        if (boxes.ndim!=3 or boxes.shape[0]!=1 or boxes.shape[-1]!=5 or
                poses.shape!=(1,boxes.shape[1],17,3) or not np.isfinite(boxes).all()
                or not np.isfinite(poses).all()):
            raise ValueError('Unexpected RTMO end-to-end detector or pose output.')
        candidates = []
        for detection,pose in zip(boxes[0],poses[0]):
            left,top,right,bottom,score = detection
            if score <= settings['score_threshold']:
                continue
            box = [max(0,math.floor(left/scale)),max(0,math.floor(top/scale)),
                   min(width,math.ceil(right/scale)),min(height,math.ceil(bottom/scale))]
            if box[2]>box[0] and box[3]>box[1]:
                candidates.append((box,float(score),pose))
        if not candidates:
            return [],[],np.empty((0,17,2)),np.empty((0,17))
        xywh = [[a,b,c-a,d-b] for (a,b,c,d),_,_ in candidates]
        keep = cv2.dnn.NMSBoxes(xywh,[s for _,s,_ in candidates],settings['score_threshold'],
                                settings['nms_threshold'])
        selected = [candidates[i] for i in np.asarray(keep,dtype=int).ravel()[:settings['max_people_per_frame']]]
        if not selected:
            return [],[],np.empty((0,17,2)),np.empty((0,17))
        bounds, scores, selected_poses = zip(*selected)
        poses = np.stack(selected_poses)
        return list(bounds),list(scores),poses[:,:,:2]/scale,poses[:,:,2]


def needs_retry(image,faces,people):
    if len(people)>2 or len(faces)>2:
        return False
    context = getattr(image,'_recognition_crowd',None)
    if len(people)<2:
        return len(faces)==2 or bool(context and context[4])
    (a,b,c,d),(e,f,g,h) = [p['bounds'] for p in people]
    intersection = max(0,min(c,g)-max(a,e))*max(0,min(d,h)-max(b,f))
    return intersection > .2*min((c-a)*(d-b),(g-e)*(h-f))


class ContactModel(SapiensIDFaces):
    def __init__(self,config,paths,stats):
        self.rtmo = None
        self.stats = stats
        super().__init__(config,paths,load_detector=False)

    def _activate(self,image,faces,people):
        super()._activate(image,faces,people)
        if not needs_retry(image,faces,self.frame_people):
            return faces
        self.stats['retry_cues'] += 1
        if self.rtmo is None:
            self.rtmo = RTMO(self.config)
            self.stats['rtmo_device'] = self.rtmo.device
        outputs = self.rtmo.predict(image,self.config.data['sapiensid']['pose'])
        found = self._people(*outputs)
        if pair(found) is None:
            return faces
        matches = associate(faces,found)
        if any(m is not None and n is None for m,n in zip(self.face_matches,matches)):
            return faces
        self.stats['replacement_cues'] += 1
        self.frame_people,self.face_matches = found,matches
        self._person_frame = image,{tuple(f.row):found[m] if m is not None else None for f,m in zip(faces,matches)}
        from stash_ai.recognition.crowd_context import remember
        remember(self,image,found)
        return faces

    def close(self):
        self.rtmo = None
        super().close()
