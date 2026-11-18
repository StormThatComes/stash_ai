"""Bounded post-NMS reserves shared by box-only and YOLO person detectors."""
import numpy as np


def select(result, iou, maximum, *, contact_max_iou=.94):
    import torch
    from types import SimpleNamespace
    boxes, scores = result.boxes.xyxy, result.boxes.conf
    xy=boxes.detach().cpu().numpy()
    confidence=scores.detach().cpu().numpy()
    areas=np.maximum(0,xy[:,2:]-xy[:,:2]).prod(axis=1)
    def overlaps(index):
        intersection=np.maximum(0,np.minimum(xy[index,2:],xy[:,2:])-np.maximum(xy[index,:2],xy[:,:2])).prod(axis=1)
        return intersection/np.maximum(areas[index]+areas-intersection,1e-12)
    primary=[];suppressed=set()
    for index in np.argsort(-confidence,kind='stable'):
        if int(index) in suppressed:continue
        primary.append(int(index))
        suppressed.update(np.flatnonzero(overlaps(index)>iou).tolist())
        if len(primary)>=maximum:break
    chosen, parents = list(primary), [-1]*len(primary)
    occupied = set(primary)
    for slot, index in enumerate(primary):
        overlap = overlaps(index)
        options = [j for j in range(len(xy)) if j not in occupied and iou < overlap[j] < contact_max_iou]
        if options:
            j = max(options,key=lambda j:float(confidence[j]))
            chosen.append(j); parents.append(slot); occupied.add(j)
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes[chosen],conf=scores[chosen]),
                           reserve_parents=torch.tensor(parents),keypoints=None)
