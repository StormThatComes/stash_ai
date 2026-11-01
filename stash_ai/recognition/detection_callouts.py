"""Display-only face translations for full-cue detection audits."""
import math

from .person_audit import BOX_COLOR


GAP = 12
LEADER_COLOR = (*BOX_COLOR, 90)


def _valid(point, size):
    x, y = point
    return math.isfinite(x) and math.isfinite(y) and 0 <= x < size[0] and 0 <= y < size[1]


def _shift(bounds, dx, dy):
    left, top, right, bottom = bounds
    return [left + dx, top + dy, right + dx, bottom + dy]


def _overlaps(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def layout_faces(image, faces, people, labels):
    """Reserve room for box, points and labels; use a right gutter if crowded.

    Keep the source at (0, 0) and its original scale. Only valid source points
    can become display points: translating (-1, -1) must not resurrect it.
    """
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image)
    width, height = image.size
    bounds = []
    for face in faces:
        x, y, w, h = face.box
        # Match source clipping before moving a partially visible face box.
        bounds.append([max(0, min(width - 1, x)), max(0, min(height - 1, y)),
                       max(0, min(width - 1, x + w)), max(0, min(height - 1, y + h))])
    occupied = list(bounds)
    # Protect the measured pose points and their index labels, not entire bodies.
    for person in people:
        for point in person['keypoints']:
            if _valid(point, image.size):
                x, y = point
                occupied.append([x - 4, y - 12, x + 20, y + 5])
    for face in faces:
        for point in face.landmarks:
            if _valid(point, image.size):
                x, y = point
                occupied.append([x - 4, y - 4, x + 4, y + 4])
    layouts = []
    canvas_width, canvas_height = image.size
    gutter_x, gutter_y, column_width = width + GAP, GAP, 0
    for face, box, label in zip(faces, bounds, labels):
        valid = [point for point in face.landmarks if _valid(point, image.size)]
        text_box = painter.textbbox((0, 0), label)
        # The shared renderer places point indices at (+4, -8), face labels
        # above the box, and uses 2px point radii. Include all of that footprint.
        area = [min([box[0] - 3] + [p[0] - 3 for p in valid]),
                min([box[1] - 16] + [p[1] - 10 for p in valid]),
                max([box[2] + 3, box[0] + text_box[2] - text_box[0] + 3] +
                    [p[0] + 18 for p in valid]),
                max([box[3] + 3] + [p[1] + 5 for p in valid])]
        span_x = math.ceil(area[2] - area[0] + GAP)
        span_y = math.ceil(area[3] - area[1] + GAP)
        candidates = [(span_x, 0), (-span_x, 0), (0, -span_y), (0, span_y),
                      (span_x, -span_y), (-span_x, -span_y),
                      (span_x, span_y), (-span_x, span_y)]
        offset = None
        for dx, dy in candidates:
            target = _shift(area, dx, dy)
            if (target[0] >= 0 and target[1] >= 0 and target[2] < width and target[3] < height
                    and not any(_overlaps(target, other) for other in occupied)):
                offset = (dx, dy)
                break
        if offset is None:
            # Pack overflow annotations into columns beside the unscaled cue.
            area_width = math.ceil(area[2] - area[0]) + 1
            area_height = math.ceil(area[3] - area[1]) + 1
            if gutter_y > GAP and gutter_y + area_height > height:
                gutter_x += column_width + GAP
                gutter_y, column_width = GAP, 0
            offset = (math.ceil(gutter_x - area[0]), math.ceil(gutter_y - area[1]))
            gutter_y += area_height + GAP
            column_width = max(column_width, area_width)
        dx, dy = offset
        target = _shift(area, dx, dy)
        occupied.append(target)
        if target[2] >= width:
            canvas_width = max(canvas_width, math.ceil(target[2]) + GAP)
        if target[3] >= height:
            canvas_height = max(canvas_height, math.ceil(target[3]) + GAP)
        layouts.append(dict(offset=list(offset), bounds=_shift(box, dx, dy),
                            landmarks=[[x + dx, y + dy] if _valid((x, y), image.size) else [-1, -1]
                                       for x, y in face.landmarks]))
    return layouts, (canvas_width, canvas_height)


def _dashed(painter, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if not length:
        return
    for distance in range(0, math.ceil(length), 7):
        stop = min(distance + 3, length)
        painter.line([(start[0] + dx * distance / length, start[1] + dy * distance / length),
                      (start[0] + dx * stop / length, start[1] + dy * stop / length)],
                     fill=LEADER_COLOR, width=1)


def draw_leaders(image, faces, layouts, source_size):
    """Draw faint origins/leaders first, so pose measurements stay on top."""
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image, 'RGBA')
    for face, layout in zip(faces, layouts):
        for original, shifted in zip(face.landmarks, layout['landmarks']):
            if not _valid(original, source_size):
                continue
            _dashed(painter, original, shifted)
            x, y = original
            painter.ellipse((x - 2, y - 2, x + 2, y + 2), outline=LEADER_COLOR, width=1)
