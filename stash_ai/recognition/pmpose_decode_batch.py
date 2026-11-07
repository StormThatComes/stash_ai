"""Batched DARK/UDP. Keep the reference path for numerically ambiguous peaks."""
import numpy as np


def decode_batch(maps,centers,scales):
    import torch
    import cv2
    import torch.nn.functional as F
    from .pmpose_geometry import decode
    n,j,h,w=maps.shape
    if (j,h,w)!=(17,64,48):raise ValueError('Invalid PMPose heatmap shape.')
    flat=maps.reshape(n*j,1,h,w)
    peaks,idx=flat.reshape(n*j,-1).max(1)
    # Separable, zero-padded convolution matches the padded OpenCV reference.
    # FP64 convolution avoids TF32/cuDNN precision changing peaks near ties.
    kernel=torch.as_tensor(cv2.getGaussianKernel(11,0),device=maps.device,dtype=torch.float64)
    blur=F.conv2d(flat.double(),kernel.reshape(1,1,1,11),padding=(0,5))
    blur=F.conv2d(blur,kernel.reshape(1,1,11,1),padding=(5,0)).reshape(n*j,h*w).float()
    blur*= (peaks/(blur.max(1).values+1e-12))[:,None]
    logs=blur.clamp(1e-3,50.).log()
    y,x=idx//w,idx%w
    def at(dy,dx):
        pos=((y+dy).clamp(0,h-1)*w+(x+dx).clamp(0,w-1))[:,None]
        return logs.gather(1,pos)[:,0]
    v=at(0,0);left,right,top,bottom=at(0,-1),at(0,1),at(-1,0),at(1,0)
    grad=torch.stack(((right-left)*.5,(bottom-top)*.5),1).double()
    cross=.5*(at(1,1)-right-bottom+2*v-left-top+at(-1,-1))
    hess=torch.stack((right-2*v+left,cross,cross,bottom-2*v+top),1).reshape(-1,2,2).double()
    hess+=torch.eye(2,dtype=torch.float64,device=maps.device)*np.finfo(np.float32).eps
    # Only ambiguous samples transfer heatmaps; ill-conditioned Hessians amplify
    # small differences between OpenCV and GPU blur kernels.
    det=torch.linalg.det(hess)
    fallback=((det.abs()<1e-5*hess.abs().amax((1,2)).clamp_min(1).square()) & (peaks>0)).view(n,j).any(1)
    offset=(torch.linalg.pinv(hess,rtol=1e-15)@grad.unsqueeze(2)).squeeze(2)
    xy=(torch.stack((x,y),1)-offset).float().double().reshape(n,j,2)
    scales=torch.as_tensor(scales,device=maps.device,dtype=torch.float64)
    centers=torch.as_tensor(centers,device=maps.device,dtype=torch.float64)
    xy=xy/xy.new_tensor((w-1,h-1))*scales[:,None]+centers[:,None]-scales[:,None]*.5
    xy=torch.where((peaks>0).view(n,j,1),xy,-1.).float().cpu().numpy()
    for i in fallback.nonzero().flatten().tolist():
        xy[i]=decode(maps[i].cpu().numpy(),centers[i].cpu().numpy(),scales[i].cpu().numpy())
    if not np.isfinite(xy).all():raise ValueError('Non-finite PMPose coordinates.')
    return xy
