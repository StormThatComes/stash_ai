"""Sharpness inside measured foreground, excluding artificial mask boundaries."""
import numpy as np
from contextvars import ContextVar
from contextlib import contextmanager

from .blur_quality import measure

_diagnostics=ContextVar('face_quality_diagnostics',default=False)


@contextmanager
def diagnostics(enabled):
    token=_diagnostics.set(enabled)
    try:yield
    finally:_diagnostics.reset(token)


def laplacian(values, mask=None, *, diagnostics=True):
    """Legacy diagnostic only; admission and ranking use measure()."""
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
    pixels=values[mask] if mask is not None else values.ravel()
    contrast=float(np.var(pixels)) if pixels.size else 0.
    variance=float(response.var()) if response.size else 0.
    return dict(sharpness_normalized=variance/max(contrast,1.), sharpness_contrast=contrast,
                sharpness_noise=(float(np.median(np.abs(response-np.median(response)))) if response.size else 0.) if diagnostics else None,
                sharpness=variance,
                sharpness_pixels=int(response.size))


def face(blob, mask, method, *, diagnostics=True):
    """Measure the existing FaceOcc 256px input, without another image/model pass."""
    rgb = np.asarray(blob,dtype=np.float32)[0].transpose(1,2,0)
    rgb = (rgb*np.asarray([.229,.224,.225],dtype=np.float32)
           +np.asarray([.485,.456,.406],dtype=np.float32))*255
    gray = rgb @ np.asarray([.299,.587,.114],dtype=np.float32)
    return dict(measure(gray,mask,diagnostics=diagnostics),sharpness_method=method)


def face_geometry(image, detection, *, diagnostics=False):
    from .visibility import prepare, VisibilityGeometryError
    try:
        blob, region = prepare(image,detection,allow_partial=True)
    except VisibilityGeometryError:
        return dict(sharpness=0.,sharpness_pixels=0,sharpness_method='unavailable')
    return face(blob,region,'face_ellipse',diagnostics=diagnostics)


def device_faces(inputs, masks, *, diagnostics=True):
    """Same measurement on the model device; transfer only scalar diagnostics."""
    import torch
    inputs = inputs.float()
    rgb = (inputs*inputs.new_tensor([.229,.224,.225])[None,:,None,None]
           +inputs.new_tensor([.485,.456,.406])[None,:,None,None])*255
    gray = (rgb*inputs.new_tensor([.299,.587,.114])[None,:,None,None]).sum(dim=1)
    from .blur_quality import batch, records
    summaries = batch(gray, masks, torch).cpu().tolist()
    return [dict(record, sharpness_method='faceocc_mask')
            for record in records(summaries, diagnostics=diagnostics)]
