"""Audit-style overlays from retained source geometry, without model inference."""
import math

from .alignment_audit import _png
from .models import checked_detection
from .person_audit import BOX_COLOR, FACE_COLORS, POSE_COLOR, _draw


def export(image, path, geometry, bounds, *, pose=False, keypoints=None):
    """Map source measurements into an unaligned retained crop and annotate a copy."""
    if bounds is None:
        return dict(status='missing_crop_bounds', file=None)
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4
            or any(type(v) is not int or v < 0 for v in bounds)
            or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
        return dict(status='invalid_crop_bounds', file=None)
    try:
        face = checked_detection(geometry)
        points = list(face.landmarks)
        if pose:
            if not isinstance(keypoints, (list, tuple)) or len(keypoints) != 17:
                return dict(status='missing_pose_geometry', file=None)
            if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in keypoints):
                return dict(status='invalid_pose_geometry', file=None)
            points = [(float(x), float(y)) for x, y in keypoints] + points
    except (TypeError, ValueError, OverflowError):
        return dict(status='invalid_geometry', file=None)

    left, top, right, bottom = bounds
    sx, sy = image.width / (right-left), image.height / (bottom-top)
    # Test in source coordinates first: a missing (-1,-1) or an outside point
    # must never become an apparently valid measurement at a crop border.
    points = [((x-left)*sx, (y-top)*sy)
              if math.isfinite(x) and math.isfinite(y) and left <= x < right and top <= y < bottom
              else (-1., -1.) for x, y in points]
    if pose:
        box = (0., 0., float(image.width), float(image.height))
    else:
        x, y, width, height = face.box
        box = ((x-left)*sx, (y-top)*sy, (x+width-left)*sx, (y+height-top)*sy)
    target = path.with_name(path.stem + '-annotated.png')
    with image.copy() as annotated:
        drawn = _draw(annotated, box, points, face_only=not pose, labels=pose)
        _png(annotated, target)
    return dict(status='exported', file=target.name,
        source='retained_source_geometry', coordinate_system='retained_crop_pixels',
        crop_bounds=list(bounds), source_to_crop_scale=[sx, sy], box=list(box),
        box_color=list(BOX_COLOR), pose_color=list(POSE_COLOR) if pose else None,
        face_colors=[list(c) for c in FACE_COLORS], index_base=0,
        pose_indices=list(range(17)) if pose else [],
        face_indices=list(range(17, 22)) if pose else list(range(5)),
        point_order=('COCO17 pose followed by five face-detector landmarks' if pose else
                     'Five face-detector landmarks'), drawn_indices=drawn,
        note='Measured source points mapped to retained pixels, not canonicalized model input. Missing and outside points are omitted.')
