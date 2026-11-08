"""Compact ViTPose DARK decoding with the processor as the numerical fallback."""
import numpy as np


def compact(maps, boxes, processor, fallback):
    import torch
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale, transform_preds
    n, joints, height, width = maps.shape
    if joints != 17 or min(height,width) < 2 or n != sum(map(len,boxes)):
        raise ValueError('Invalid ViTPose heatmap dimensions.')
    maps = maps.float().reshape(n*joints,height*width)
    peaks, indices = maps.max(dim=1)
    x, y = indices % width, indices // width
    # DARK needs seven smoothed neighbours of the peak. SciPy uses
    # half-sample symmetric reflection and rounds to float32 after each axis.
    offsets = torch.arange(-5,6,device=maps.device)
    weights = torch.exp(-0.5 * (offsets.double()/0.8).square())
    weights /= weights.sum()
    def reflect(index, extent):
        folded = index % (2*extent)
        return torch.minimum(folded,2*extent-1-folded)
    def log_heat(dx,dy):
        sx = reflect((x+dx).clamp(0,width-1)[:,None]+offsets,width)
        sy = reflect((y+dy).clamp(0,height-1)[:,None]+offsets,height)
        addresses = sy[:,:,None]*width + sx[:,None,:]
        samples = maps.gather(1,addresses.flatten(1)).reshape(-1,11,11)
        vertical = (samples.double()*weights[None,:,None]).sum(1).float()
        filtered = (vertical.double()*weights).sum(1).float()
        return filtered.clamp(0.001,50).log()
    center, right, left = log_heat(0,0), log_heat(1,0), log_heat(-1,0)
    bottom, top = log_heat(0,1), log_heat(0,-1)
    cross = .5*(log_heat(1,1)-right-bottom+center+center-left-top+log_heat(-1,-1))
    xx = (right-2*center+left).double()+np.finfo(np.float32).eps
    yy = (bottom-2*center+top).double()+np.finfo(np.float32).eps
    cross = cross.double()
    det = xx*yy-cross.square()
    # A singular or ill-conditioned Hessian can amplify a one-ulp blur
    # difference; nonpositive peaks use the upstream padded-index semantics.
    unsafe = (peaks<=0) | ~torch.isfinite(peaks) | ~torch.isfinite(det) | (
        det.abs() < 1e-5*torch.maximum(xx.abs(),yy.abs()).clamp_min(1).square())
    if unsafe.any().item():
        return fallback()
    gx, gy = (.5*(right-left)).double(), (.5*(bottom-top)).double()
    coords = torch.stack((x-(yy*gx-cross*gy)/det,y-(xx*gy-cross*gx)/det),dim=-1).float()
    packed = torch.cat((coords,peaks[:,None]),dim=1).reshape(n,joints,3).cpu().numpy()
    points = np.empty((n,joints,2),dtype=np.float32)
    for i,box in enumerate(box for group in boxes for box in group):
        center,scale = box_to_center_and_scale(box,image_width=processor.size['width'],
                                                image_height=processor.size['height'])
        points[i] = transform_preds(packed[i,:,:2],center,scale,[height,width])
    if not np.isfinite(points).all():
        return fallback()
    return points, packed[:,:,2], np.broadcast_to(np.arange(joints),(n,joints))
