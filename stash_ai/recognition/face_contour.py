"""Independent facial surface geometry; FaceOcc logits never choose its area."""
from dataclasses import dataclass

import numpy as np

from .visibility import VisibilityError, VisibilityGeometryError


@dataclass
class Region:
    mask: np.ndarray
    total: int
    support: np.ndarray
    method: str
    reason: str = ''
    outline: list | None = None
    landmarks: np.ndarray | None = None

    def record(self):
        return {'region_method': self.method, 'region_fallback_reason': self.reason}


def real_pixels(affine, size):
    y, x = np.indices((256, 256), dtype=np.float64) + .5
    a, b, c, d, e, f = affine
    return ((a*x+b*y+c >= 0) & (a*x+b*y+c < size[0])
            & (d*x+e*y+f >= 0) & (d*x+e*y+f < size[1]))


def prepare(image, face):
    """Box-centered upright input; no ellipse fit, mask or area is required."""
    from PIL import Image
    from .face_frame import pose_axis
    from .visibility import _constants
    (cosine, sine), _ = pose_axis(face.landmarks)
    x, y, width, height = face.box
    side = max(width, height) * 1.5
    scale = side / 256
    cx, cy = x+width/2, y+height/2
    affine = (scale*cosine, -scale*sine, cx-side/2*(cosine-sine),
              scale*sine, scale*cosine, cy-side/2*(sine+cosine))
    with image.convert('RGB') as rgb, rgb.transform((256,256), Image.Transform.AFFINE,
            affine, resample=Image.Resampling.BILINEAR) as crop:
        pixels = np.asarray(crop, dtype=np.float32)/255.
    mean, std, _, _ = _constants()
    blob = np.ascontiguousarray(((pixels-mean)/std).transpose(2,0,1)[None])
    return blob, None, 0, affine


def mesh_region(points, presence, item):
    """Project a bounded mesh surface envelope, including the nose in profiles.

    Keep xyz, but count projected pixels only once: the far side of a profile
    must not contribute a second facial area. The envelope is deliberately
    independent of segmentation, including any externally occluded surface.
    """
    import cv2
    points = np.asarray(points, dtype=np.float64).reshape(478, 3)
    if not np.isfinite(points).all() or not np.isfinite(presence) or presence < 0:
        raise VisibilityGeometryError('Mesh did not confirm a finite facial surface.')
    affine = np.asarray(item.affine).reshape(2, 3)
    inverse = cv2.invertAffineTransform(affine)
    expected = (np.asarray(item.face.landmarks)+.5) @ inverse[:, :2].T + inverse[:, 2]
    xy = points[:, :2]
    anchors = np.array([(xy[33]+xy[133])/2, (xy[362]+xy[263])/2,
                        xy[1], xy[61], xy[291]])
    # Match paired features in the upright crop, not the source x-axis. Avoid
    # dividing by the near-zero interocular distance of a profile.
    for pair in (slice(0, 2), slice(3, 5)):
        anchors[pair] = anchors[pair][np.argsort(anchors[pair, 0])]
        expected[pair] = expected[pair][np.argsort(expected[pair, 0])]
    scale = np.linalg.norm(expected[3:].mean(axis=0)-expected[:2].mean(axis=0))
    errors = np.linalg.norm(anchors-expected, axis=1)/max(scale, 1.)
    # Associate to the selected detection, allowing one faulty interior point.
    # These points do not prescribe the contour's shape, center or area.
    inliers = np.sort(errors)[:4]
    if np.mean(inliers) > .22 or np.max(inliers) > .45:
        raise VisibilityGeometryError('Mesh disagrees with the selected face landmarks.')
    # Iris landmarks do not define the skin contour. A projected envelope also
    # avoids the crossing oval polygon of a nearly side-on face.
    hull = cv2.convexHull(xy[:468].astype('float32')).reshape(-1, 2)
    if (hull < 1).any() or (hull > 255).any():
        raise VisibilityGeometryError('Mesh contour extends outside the model crop.')
    mask = np.zeros((256, 256), np.uint8)
    cv2.fillConvexPoly(mask, np.rint((hull-.5)*256).astype('int32'), 1, shift=8)
    total = int(mask.sum())
    # Reject collapsed or implausibly enlarged surfaces, independently of logits.
    if not .02*256**2 <= total <= .85*256**2:
        raise VisibilityGeometryError('Mesh contour has implausible facial area.')
    real = real_pixels(item.affine, item.source_size)
    # Sharpness may use nearby visible skin outside the reference contour, but
    # never all positive logits in the crop (which can include another face).
    support = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool) & real
    mask = mask.astype(bool) & real
    outline = np.vstack((hull, hull[:1])) @ affine[:, :2].T + affine[:, 2]
    source_points = xy @ affine[:, :2].T + affine[:, 2]
    return Region(mask, total, support, 'mesh', outline=outline.tolist(), landmarks=source_points)


class Mesh:
    """One verified mesh-only ORT session owned by the visibility model worker."""
    def __init__(self, config, paths):
        from .models import ASSETS, ensure_assets, verified_bytes
        from .runtime import runtime, session
        from .expression_onnx import check_io
        from .portrait_threads import native_threads
        if 'expression_mesh' not in paths:
            paths = ensure_assets(config, keys=['expression_mesh'])
        blob = verified_bytes(paths['expression_mesh'], ASSETS['expression_mesh'])
        self.session, self.device = session(config, blob, runtime_info=runtime(config),
                                           cpu_fallback=False, cpu_threads=native_threads(config))
        check_io(self.session, 'input_12', (None, 256, 256, 3),
                 {'Identity': (None, 1, 1, 1434), 'Identity_1': (None, 1, 1, 1), 'Identity_2': (None, 1)})

    def region(self, item):
        # Same upright pixels, different normalization/layout; no second crop.
        blob = item[0][0].transpose(1, 2, 0)
        rgb = blob*np.array([.229, .224, .225], np.float32)+np.array([.485, .456, .406], np.float32)
        values, presence = self.session.run(['Identity', 'Identity_1'],
                                            {'input_12': np.ascontiguousarray(rgb[None])})
        if values.shape != (1, 1, 1, 1434) or presence.shape != (1, 1, 1, 1):
            raise ValueError('Face contour model returned incompatible outputs.')
        return mesh_region(values, float(presence.ravel()[0]), item)

    def close(self):
        self.session = None


def resolve(model, item):
    mesh = getattr(model, 'mesh', None)
    mode = getattr(model, 'region_requested', 'mesh' if mesh is not None else 'ellipse')
    if mode == 'mesh':
        if mesh is None:
            raise VisibilityError('Mesh visibility model is unavailable; ellipse fallback is disabled.')
        if not hasattr(item, 'face'):
            raise VisibilityGeometryError('Mesh visibility requires prepared face geometry; no ellipse fallback.')
        if getattr(item, 'region_mode', None) != 'mesh':
            raise VisibilityGeometryError('Mesh visibility requires its box-based crop, not an ellipse input.')
        region = mesh.region(item)
    else:
        if item[1] is None or not item[2]:
            raise VisibilityGeometryError('Ellipse visibility requires an explicitly prepared ellipse.')
        region = Region(item[1], item[2], item[1], 'ellipse')
    count = int(region.mask.sum())
    if not count or count/region.total < model.minimum_source_fraction:
        raise VisibilityGeometryError('Insufficient face region coverage in the real source.',
                                      source_pixels=count, face_pixels=region.total)
    return region
