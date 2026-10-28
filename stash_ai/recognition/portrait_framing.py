"""Integer portrait geometry, including head-aware framing of retained people."""
from fractions import Fraction
import math


def aspect_size(width, height, ratio):
    """Fit the ratio without upscaling; use exact integer multiples when possible."""
    width, height = max(1, math.floor(width+1e-7)), max(1, math.floor(height+1e-7))
    fraction = Fraction(str(ratio)).limit_denominator(10000)
    unit_width, unit_height = fraction.numerator, fraction.denominator
    scale = min(width // unit_width, height // unit_height)
    if scale:
        return unit_width * scale, unit_height * scale
    # Arbitrary ratios and tiny sources cannot always be exact on a pixel grid.
    if width / height > ratio:
        width = max(1, min(width, round(height * ratio)))
    else:
        height = max(1, min(height, round(width / ratio)))
    return width, height


def matches_aspect(size, ratio):
    width, height = size
    fraction = Fraction(str(ratio)).limit_denominator(10000)
    if width >= fraction.numerator and height >= fraction.denominator:
        return width*fraction.denominator == height*fraction.numerator
    return width == max(1, round(height*ratio)) or height == max(1, round(width/ratio))


def _head(size, geometry):
    """COCO nose/eyes/ears plus clearance, mapped from source to retained pixels."""
    if not isinstance(geometry, dict):
        return None
    bounds, points = geometry.get('bounds'), geometry.get('keypoints')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or not isinstance(points, (list, tuple)):
        return None
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds):
        return None
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        return None
    width, height = size
    head = []
    for point in points[:5]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = point
        if (all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in point)
                and left <= x < right and top <= y < bottom):
            head.append(((x-left)*width/(right-left), (y-top)*height/(bottom-top)))
    if not head:
        return None
    xs, ys = zip(*head)
    # Landmarks describe facial features, not the skull/hair outline.
    clearance = max(4., .6*max(max(xs)-min(xs), max(ys)-min(ys)), .08*min(size))
    return ((max(0., min(xs)-clearance), max(0., min(ys)-clearance),
             min(width, max(xs)+clearance), min(height, max(ys)+clearance)),
            (min(xs), min(ys), max(xs), max(ys)))


def _origin(preferred, length, maximum, protected=None):
    if protected is not None:
        low, high = protected
        if high-low <= length:
            preferred = max(high-length, min(preferred, low))
        else:
            preferred = (low+high-length)/2
    return max(0, min(round(preferred), maximum-length))


def person_bounds(size, geometry, ratio):
    width, height = size
    if ratio is None:
        return [0, 0, width, height]
    crop_width, crop_height = aspect_size(width, height, ratio)
    head = _head(size, geometry)
    left, top = _origin((width-crop_width)/2, crop_width, width), 0
    # Sacrifice clearance before measured features when the whole head cannot fit.
    for region in head or ():
        left = _origin(left, crop_width, width, (region[0], region[2]))
        top = _origin(top, crop_height, height, (region[1], region[3]))
    return [left, top, left+crop_width, top+crop_height]
