"""Shared roll frame for profile visibility and pose quality."""
import numpy as np


def axis(points,width,height):
    from .visibility import VisibilityGeometryError
    points=np.asarray(points,dtype=float)
    eye=points[1]-points[0]
    distance=float(np.linalg.norm(eye))
    if distance<=1e-6:
        raise VisibilityGeometryError('Distinct eye landmarks are required.')
    axis = eye / distance
    # At a narrow eye baseline, a one-pixel error can rotate the entire crop.
    # Use the longer vertical facial axis only when both nose and mouth agree.
    minimum = max(2., .16 * min(width, height))
    supported_axis = False
    if distance < minimum:
        eyes = points[:2].mean(axis=0)
        mouth = points[3:].mean(axis=0)
        down = mouth - eyes
        length = np.linalg.norm(down)
        if length > max(3., .15 * min(width, height)):
            down /= length
            nose_height = np.dot(points[2]-eyes, down) / length
            if .12 < nose_height < .90:
                horizontal = np.array([down[1], -down[0]])
                weight = min(1., 2 * (1 - distance / minimum))
                mixed = (1-weight)*axis + weight*horizontal
                if np.linalg.norm(mixed) > .25:
                    axis = mixed / np.linalg.norm(mixed)
                    supported_axis = True
    if distance <= 1 and not supported_axis:
        raise VisibilityGeometryError('Visibility alignment requires distinct eye landmarks or a supported profile axis.')
    return axis, supported_axis
