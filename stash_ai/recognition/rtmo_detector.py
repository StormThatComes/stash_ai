"""Experimental one-stage RTMO-S boxes and COCO17, without a second detector."""
import math
from types import SimpleNamespace
import numpy as np
from . import rtmo_registry as registry


def load(config):
    from .sapiensid_loading import local_path,checked_bytes
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .runtime import session
    settings=config.data['sapiensid']['pose']
    directory=local_path(config,settings['rtmo_directory']) if settings['rtmo_directory'] else config.model_dir/'rtmo_s'/registry.SHA256
    path=directory/'end2end.onnx'
    if not path.exists():
        _require_download(config,path)
        install(registry.URL,path,registry.SIZE,registry.SHA256,byte_range=registry.BYTE_RANGE,archive_size=registry.ARCHIVE_SIZE)
    runtime,device=session(config,checked_bytes(path,registry.SHA256,limit=registry.SIZE))
    return RTMO(runtime,device,settings)


def prepare(image):
    import cv2
    scale=min(640/image.width,640/image.height)
    with image.convert('RGB') as rgb:
        resized=cv2.resize(np.asarray(rgb)[:,:,::-1],(max(1,int(image.width*scale)),max(1,int(image.height*scale))))
    pixels=np.full((640,640,3),114,dtype=np.float32)
    pixels[:resized.shape[0],:resized.shape[1]]=resized
    return np.ascontiguousarray(pixels.transpose(2,0,1)),scale


def decode(boxes,poses,size,scale,conf,iou,maximum,*,torso_nms=False,keypoint_floor=.5):
    import cv2
    import torch
    if (boxes.ndim!=2 or boxes.shape[1]!=5 or poses.shape!=(len(boxes),17,3)
            or not np.isfinite(boxes).all() or not np.isfinite(poses).all()
            or np.any((boxes[:,4]<0)|(boxes[:,4]>1)) or np.any((poses[:,:,2]<0)|(poses[:,:,2]>1))):
        raise ValueError('Malformed RTMO end-to-end boxes or COCO17 scores.')
    width,height=size; candidates=[]
    for box,pose in zip(boxes,poses):
        a,b,c,d,score=box
        if score<=conf:continue
        bounds=[max(0,math.floor(a/scale)),max(0,math.floor(b/scale)),min(width,math.ceil(c/scale)),min(height,math.ceil(d/scale))]
        if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:continue
        candidates.append((bounds,float(score),pose))
    kept=[]
    if candidates:
        if torso_nms:
            from .crowd_context import pair
            people=[]
            for bounds,score,pose in candidates:
                xy=pose[:,:2]/scale
                valid=(pose[:,2]>keypoint_floor)&(xy[:,0]>=bounds[0])&(xy[:,0]<bounds[2])&(xy[:,1]>=bounds[1])&(xy[:,1]<bounds[3])
                people.append(dict(bounds=bounds,keypoints=np.where(valid[:,None],xy,-1).tolist()))
            for i in sorted(range(len(candidates)),key=lambda i:(-candidates[i][1],tuple(candidates[i][0]))):
                a,b,c,d=candidates[i][0]
                for j in kept:
                    e,f,g,h=candidates[j][0]
                    intersection=max(0,min(c,g)-max(a,e))*max(0,min(d,h)-max(b,f))
                    union=(c-a)*(d-b)+(g-e)*(h-f)-intersection
                    if intersection>iou*union and not pair([people[i],people[j]]):break
                else:
                    kept.append(i)
                    if len(kept)>=maximum:break
        else:
            xywh=[[a,b,c-a,d-b] for (a,b,c,d),_,_ in candidates]
            kept=np.asarray(cv2.dnn.NMSBoxes(xywh,[s for _,s,_ in candidates],conf,iou),dtype=int).ravel()[:maximum]
    chosen=[candidates[i] for i in kept]
    bounds=np.asarray([r[0] for r in chosen],dtype=np.float32).reshape(-1,4)
    scores=np.asarray([r[1] for r in chosen],dtype=np.float32)
    keypoints=np.asarray([r[2] for r in chosen],dtype=np.float32).reshape(-1,17,3)
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.from_numpy(bounds),conf=torch.from_numpy(scores)),
        keypoints=SimpleNamespace(xy=torch.from_numpy(keypoints[:,:,:2]/scale),conf=torch.from_numpy(keypoints[:,:,2].copy())))


class RTMO:
    task='pose'
    def __init__(self,session,device,settings):
        self.session,self.device,self.settings=session,device,settings
        inputs=session.get_inputs()
        if len(inputs)!=1 or inputs[0].type!='tensor(float)' or inputs[0].shape[1:]!=[3,640,640]:
            raise ValueError('RTMO requires the pinned float32 NCHW 640x640 graph.')
        self.input=inputs[0].name
        static=inputs[0].shape[0]
        if isinstance(static,int) and static!=1:raise ValueError('Unsupported fixed RTMO batch.')
        self.batch_size=1 if isinstance(static,int) else settings['batch_size']
    def predict(self,source,*,conf,iou,max_det,device=None,verbose=False,nms=True):
        images=source if isinstance(source,list) else [source]
        results=[]
        for start in range(0,len(images),self.batch_size):
            batch=images[start:start+self.batch_size]
            prepared=[prepare(image) for image in batch]
            outputs=self.session.run(None,{self.input:np.stack([p[0] for p in prepared])})
            if len(outputs)!=2:raise ValueError('Unexpected RTMO output count.')
            boxes,poses=outputs
            if boxes.ndim!=3 or boxes.shape[0]!=len(batch) or poses.shape!=(len(batch),boxes.shape[1],17,3):
                raise ValueError('Malformed RTMO output batch.')
            for image,(_,scale),box,pose in zip(batch,prepared,boxes,poses):
                results.append(decode(box,pose,image.size,scale,conf,iou,max_det,
                    torso_nms=self.settings['rtmo_torso_nms'],keypoint_floor=self.settings['keypoint_score_threshold']))
        return results
    def close(self):self.session=None


class SeededPose:
    """Reuse the actual one-stage output for optional mask-only SAM processing."""
    def predict_many(self,frames):
        result=[]
        for image,bounds in frames:
            original,xy,score=image._rtmo_points
            if bounds!=original:raise ValueError('RTMO mask input differs from its measured detector boxes.')
            result.append((xy.copy(),score.copy()))
        return result
    def predict(self,image,bounds):return self.predict_many([(image,bounds)])[0]
    def close(self):pass
