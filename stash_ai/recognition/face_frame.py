"""Shared roll frame for profile visibility and pose quality."""
import numpy as np


def axis(points,width,height):
    return _axis(points, max(2., .16 * min(width, height)),
                 max(3., .15 * min(width, height)))


def pose_axis(points):
    """Pose must use intrinsic landmark distances, never an axis-aligned box.

    Eye-to-mouth span is about 40% of face height, so .4 of that span replaces
    the ROI frame's .16 box-side threshold. Both this length and eye separation
    stay constant under roll. Pixel floors retain the existing degeneracy guard.
    """
    points = np.asarray(points, dtype=float)
    from .visibility import VisibilityGeometryError
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise VisibilityGeometryError('Pose alignment requires five finite landmarks.')
    span = float(np.linalg.norm(points[3:].mean(axis=0) - points[:2].mean(axis=0)))
    return _axis(points, max(2., .4 * span), 3., tolerance=1e-9)


def _axis(points, minimum, minimum_span, *, tolerance=0.):
    # A rigid rotation can change the last floating-point bits of a length.
    # Pose comparisons need a small deadband so equality cannot switch frames.
    def above(value, boundary):
        return value > boundary + tolerance * max(abs(value), abs(boundary), 1.)
    from .visibility import VisibilityGeometryError
    points=np.asarray(points,dtype=float)
    eye=points[1]-points[0]
    distance=float(np.linalg.norm(eye))
    if not above(distance,1e-6):
        raise VisibilityGeometryError('Distinct eye landmarks are required.')
    axis = eye / distance
    # At a narrow eye baseline, a one-pixel error can rotate the entire crop.
    # Use the longer vertical facial axis only when both nose and mouth agree.
    supported_axis = False
    if above(minimum,distance):
        eyes = points[:2].mean(axis=0)
        mouth = points[3:].mean(axis=0)
        down = mouth - eyes
        length = np.linalg.norm(down)
        if above(length,minimum_span):
            down /= length
            nose_height = np.dot(points[2]-eyes, down) / length
            if above(nose_height,.12) and above(.90,nose_height):
                horizontal = np.array([down[1], -down[0]])
                weight = min(1., 2 * (1 - distance / minimum))
                mixed = (1-weight)*axis + weight*horizontal
                if above(np.linalg.norm(mixed),.25):
                    axis = mixed / np.linalg.norm(mixed)
                    supported_axis = True
    if not above(distance,1.) and not supported_axis:
        raise VisibilityGeometryError('Visibility alignment requires distinct eye landmarks or a supported profile axis.')
    return axis, supported_axis
