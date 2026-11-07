"""Stage-local caching of raw person boxes and top-down predictions."""
from types import SimpleNamespace
from .inference_cache import many,current


def detections(owner,images,calculate):
    if current() is None:return calculate(images)
    from .inference_contracts import detector
    def compute(batch):
        results=[]
        for value in calculate(batch):
            record=dict(boxes=value.boxes.xyxy.detach().cpu().numpy(),scores=value.boxes.conf.detach().cpu().numpy())
            if getattr(value,'keypoints',None) is not None:
                record.update(points=value.keypoints.xy.detach().cpu().numpy(),confidence=value.keypoints.conf.detach().cpu().numpy())
            import numpy as np
            scores,boxes=record['scores'],record['boxes']
            if (scores.ndim!=1 or boxes.shape!=(len(scores),4)
                    or any(not np.isfinite(array).all() for array in record.values())
                    or np.any((scores<0)|(scores>1))):
                raise ValueError('Person detector returned malformed/non-finite geometry; result was not cached.')
            results.append(record)
        return results
    values=many('person_detection',[detector(owner.config),owner.device],images,compute)
    import torch
    return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.from_numpy(v['boxes']),conf=torch.from_numpy(v['scores'])),
        keypoints=SimpleNamespace(xy=torch.from_numpy(v['points']),conf=torch.from_numpy(v['confidence'])) if 'points' in v else None) for v in values]


class Pose:
    def __init__(self,model,config,device):
        from .inference_contracts import pose
        self.model=model;self.contract=pose(config),str(device)
    def __getattr__(self,name):return getattr(self.model,name)
    def predict(self,image,bounds):return self.predict_many([(image,bounds)])[0]
    def predict_many(self,frames):
        def calculate(values):
            method=getattr(self.model,'predict_many',None)
            return method(values) if callable(method) else [self.model.predict(*value) for value in values]
        return many('pose',self.contract,list(frames),calculate)


class RawPMPose(Pose):
    """Cache raw first and masked passes by their exact pixels, boxes and masks."""
    def __init__(self,model,device):
        from .pmpose_registry import SHA256,INFERENCE
        self.model=model;self.contract=[SHA256,INFERENCE,str(device)]

    def predict(self,image,bounds,masks=None):
        return self.predict_many([(image,bounds,masks)])[0]

    def predict_many(self,frames):
        return many('raw_pmpose',self.contract,list(frames),self.model.predict_many)
