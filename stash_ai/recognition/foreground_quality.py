"""Sharpness inside measured foreground, excluding artificial mask boundaries."""
import numpy as np


def laplacian(values, mask=None):
    values = np.asarray(values, dtype=np.float32)
    response = (4*values[1:-1,1:-1]-values[:-2,1:-1]-values[2:,1:-1]
                -values[1:-1,:-2]-values[1:-1,2:])
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ValueError('Sharpness mask must be boolean and match the measured pixels.')
        interior = (mask[1:-1,1:-1] & mask[:-2,1:-1] & mask[2:,1:-1]
                    & mask[1:-1,:-2] & mask[1:-1,2:])
        response = response[interior]
    return dict(sharpness=float(response.var()) if response.size else 0.,
                sharpness_pixels=int(response.size))


def face(blob, mask, method):
    """Measure the existing FaceOcc 256px input, without another image/model pass."""
    rgb = np.asarray(blob,dtype=np.float32)[0].transpose(1,2,0)
    rgb = (rgb*np.asarray([.229,.224,.225],dtype=np.float32)
           +np.asarray([.485,.456,.406],dtype=np.float32))*255
    gray = rgb @ np.asarray([.299,.587,.114],dtype=np.float32)
    return dict(laplacian(gray,mask),sharpness_method=method)


def face_geometry(image, detection):
    from .visibility import prepare, VisibilityGeometryError
    try:
        blob, region = prepare(image,detection,allow_partial=True)
    except VisibilityGeometryError:
        return dict(sharpness=0.,sharpness_pixels=0,sharpness_method='unavailable')
    return face(blob,region,'face_ellipse')


def device_faces(inputs, masks):
    """Same measurement on the model device; transfer only scalar diagnostics."""
    import torch
    rgb = (inputs.float()*inputs.new_tensor([.229,.224,.225])[None,:,None,None]
           +inputs.new_tensor([.485,.456,.406])[None,:,None,None])*255
    gray = (rgb*inputs.new_tensor([.299,.587,.114])[None,:,None,None]).sum(dim=1)
    response = (4*gray[:,1:-1,1:-1]-gray[:,:-2,1:-1]-gray[:,2:,1:-1]
                -gray[:,1:-1,:-2]-gray[:,1:-1,2:])
    interior = (masks[:,1:-1,1:-1] & masks[:,:-2,1:-1] & masks[:,2:,1:-1]
                & masks[:,1:-1,:-2] & masks[:,1:-1,2:])
    counts = interior.sum(dim=(1,2))
    # Center before squaring to avoid cancellation for low-texture faces.
    total = (response*interior).sum(dim=(1,2))
    means = total/counts.clamp_min(1)
    variance = (((response-means[:,None,None])**2)*interior).sum(dim=(1,2))/counts.clamp_min(1)
    values = torch.stack((variance,counts),dim=1).cpu().tolist()
    return [dict(sharpness=float(value),sharpness_pixels=int(count),sharpness_method='faceocc_mask')
            for value,count in values]
