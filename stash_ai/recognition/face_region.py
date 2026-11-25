"""Fit a whole-face template to interior landmarks, independently of FaceOcc."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Fit:
    center: tuple
    axis: tuple
    radii: tuple
    view: str
    residual: float
    ignored: str | None


@lru_cache(maxsize=1)
def _templates():
    """Anatomical priors in eye-to-mouth units, not measured visibility.

    Eyes/mouth corners are interior features: the ellipse extends above, below
    and behind them. Profile projection retains cheek width even when paired
    landmarks nearly coincide. Width variants accommodate facial proportions;
    yaw samples describe projected geometry, not calibrated head-pose angles.
    """
    points, radii, views, priors = [], [], [], []
    for yaw in (0., -.5, .5, -.75, .75, -.9, .9, -.98, .98, -1., 1.):
        q, projection = abs(yaw), np.sqrt(1-yaw*yaw)
        eye_half = .625*projection + .018*q
        mouth_half = .48*projection + .055*q
        eye_y = -.45 - .05*q
        for width in (.6, .8, 1., 1.2):
            xy = np.array([[.46*yaw-eye_half, eye_y], [.46*yaw+eye_half, eye_y],
                           [.625*yaw, .05*(1-q)],
                           [.46*yaw-mouth_half, eye_y+1], [.46*yaw+mouth_half, eye_y+1]])
            xy[:,0] *= width
            points.append(xy[:,0] + 1j*xy[:,1])
            radii.append(((1.2-.55*q)*width, 1.28-.08*q))
            views.append('frontal' if not yaw else ('profile' if q>=.9 else 'three-quarter')
                         + ('-left' if yaw<0 else '-right'))
            # Near a profile, paired points contain little cheek-width
            # information. Regularize that weak degree of freedom so subpixel
            # jitter cannot jump between very different cheek positions.
            priors.append(.0001*q*q + (.0001+.02*q**4)*(width-1)**2)
    # Ordinary fit plus each leave-one-out fit. The score still checks every
    # landmark, with a capped penalty for at most one outlier. This avoids an
    # eye/nose error dragging the whole region across the face.
    masks = np.vstack((np.ones(5), 1-np.eye(5)))
    weights = np.tile(masks, (len(points),1))
    shapes = np.repeat(np.asarray(points),6,axis=0)
    means = (shapes*weights).sum(axis=1)/weights.sum(axis=1)
    centered = shapes-means[:,None]
    denominators = (abs(centered)**2*weights).sum(axis=1)
    arrays = (shapes, weights, means, centered, denominators,
              np.repeat(np.asarray(radii),6,axis=0), np.repeat(np.asarray(priors),6))
    for value in arrays:
        value.setflags(write=False)
    return (*arrays, tuple(view for view in views for _ in range(6)))


def fit(face):
    """Robust rotation/translation/scale fit; no mask or box-derived center.

    Returned centers use image-edge coordinates (landmark pixel centers + 0.5),
    matching PIL affine sampling. A box is only a broad plausibility guard;
    changing a reasonable box cannot change an otherwise identical fit.
    """
    from .visibility import VisibilityGeometryError
    points = np.asarray(face.landmarks, dtype=np.float64)
    if points.shape != (5,2) or not np.isfinite(points).all():
        raise VisibilityGeometryError('Visibility alignment requires five finite landmarks.')
    distance = float(np.linalg.norm(points[1]-points[0]))
    if not np.isfinite(distance) or distance <= 1e-6:
        raise VisibilityGeometryError('Visibility alignment requires distinct eye landmarks.')
    shapes, weights, means, centered, denominators, radii, priors, views = _templates()
    observed = points[:,0] + 1j*points[:,1]
    origins = (observed*weights).sum(axis=1)/weights.sum(axis=1)
    gains = (weights*centered.conj()*(observed-origins[:,None])).sum(axis=1)/denominators
    scales = abs(gains)
    predictions = origins[:,None] + gains[:,None]*centered
    errors = abs(predictions-observed)/np.maximum(scales[:,None],1e-9)
    ordered = np.sort(errors,axis=1)
    # Four geometrically consistent points are required; no arbitrary ellipse
    # is substituted when the data cannot support a whole-face estimate.
    mean_error = (ordered[:,:4]**2).mean(axis=1)
    scores = mean_error + .1*np.minimum(ordered[:,4]**2,.3**2) + priors
    usable = (scales>=2) & np.isfinite(scores) & (ordered[:,3]<=.25) & (mean_error<=.15**2)
    scores = np.where(usable,scores,np.inf)
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise VisibilityGeometryError('Face ellipse requires four consistent landmarks.')
    gain, scale = gains[best], float(scales[best])
    origin = origins[best]-gain*means[best]
    center = (float(origin.real)+.5,float(origin.imag)+.5)
    rx,ry = map(float,radii[best]*scale)
    x,y,width,height = face.box
    # A loose check against corrupt/mismatched observations, never a clamp or
    # an optimization target. In particular, a shifted/oversized box does not
    # move or expand a supported face ellipse.
    if (max(rx,ry)>2*max(width,height)
            or np.hypot(center[0]-(x+width/2),center[1]-(y+height/2))>2*np.hypot(width,height)):
        raise VisibilityGeometryError('Landmark face ellipse is inconsistent with its detection.')
    worst = int(np.argmax(errors[best]))
    ignored = ('left_eye','right_eye','nose','left_mouth','right_mouth')[worst] if errors[best,worst]>.25 else None
    if views[best].startswith('profile') and ignored=='nose':
        raise VisibilityGeometryError('Profile ellipse requires a consistent nose to locate the cheek.')
    return Fit(center,(float(gain.real/scale),float(gain.imag/scale)),(rx,ry),
               views[best],float(np.sqrt((ordered[best,:4]**2).mean())),ignored)


def ellipse(face):
    value = fit(face)
    return value.center,value.axis,value.radii


def outline(face, *, fitted=None):
    """Render exactly the ellipse used by measurement, in source coordinates."""
    value = fit(face) if fitted is None else fitted
    center, horizontal, (rx, ry) = value.center,value.axis,value.radii
    c, s = horizontal
    angles = np.linspace(0, 2*np.pi, 129)
    points = np.stack((rx*np.cos(angles), ry*np.sin(angles)), axis=1)
    points = points @ np.array([[c, s], [-s, c]]) + center
    return [tuple(point) for point in points]
