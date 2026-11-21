"""Box-supported face ellipse, independent of the predicted visible foreground."""
import numpy as np


def ellipse(face):
    """Return source center, horizontal axis and radii of the whole facial region.

    A detector box is axis-aligned, so rotating its width/height directly inflates
    oblique profiles. Recover the upright ellipse dimensions from its projected
    extents instead. At 45 degrees those extents cannot determine the aspect
    ratio: use the eye-to-mouth height (about 40% of face height) as a bounded
    prior. The box still determines the total squared size and the center;
    closely spaced profile landmarks never shrink the ROI onto the features.
    """
    from .face_frame import axis
    from .visibility import VisibilityGeometryError
    points = np.asarray(face.landmarks, dtype=np.float64)
    if points.shape != (5,2) or not np.isfinite(points).all():
        raise VisibilityGeometryError('Visibility alignment requires five finite landmarks.')
    distance = float(np.linalg.norm(points[1]-points[0]))
    if not np.isfinite(distance) or distance <= 1e-6:
        raise VisibilityGeometryError('Visibility alignment requires distinct eye landmarks.')
    x, y, width, height = face.box
    horizontal, _ = axis(points, width, height)
    c, s = horizontal
    vertical = np.array([-s, c])
    span = abs(float((points[3:].mean(axis=0) - points[:2].mean(axis=0)) @ vertical))
    total = width * width + height * height
    # W² + H² = w² + h²; W² - H² = cos(2*roll) * (w² - h²).
    # Regularization vanishes at quarter turns and remains finite at 45°.
    difference_axis = c*c - s*s
    strength = .01 * (1 - difference_axis*difference_axis)**2
    # A compressed eye/mouth projection is not evidence that the forehead and
    # chin disappeared: the uncertain prior must never make height < width.
    prior = np.clip(total - 2 * (2.5 * span)**2, -.6 * total, 0.)
    difference = (difference_axis * (width*width - height*height) + strength*prior)
    difference /= difference_axis*difference_axis + strength
    # Contradictory boxes/landmarks must not collapse a facial axis. Preserve
    # exact detector dimensions at quarter turns, including elongated boxes.
    if strength > 1e-12:
        difference = np.clip(difference, -.8 * total, .8 * total)
    upright_width = np.sqrt((total + difference) / 2)
    upright_height = np.sqrt((total - difference) / 2)
    return ((x + width/2, y + height/2), horizontal,
            (float(.45 * upright_width), float(.48 * upright_height)))


def outline(face):
    """Render exactly the ellipse used by measurement, in source coordinates."""
    center, horizontal, (rx, ry) = ellipse(face)
    c, s = horizontal
    angles = np.linspace(0, 2*np.pi, 129)
    points = np.stack((rx*np.cos(angles), ry*np.sin(angles)), axis=1)
    points = points @ np.array([[c, s], [-s, c]]) + center
    return [tuple(point) for point in points]
