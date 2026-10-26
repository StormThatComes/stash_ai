"""Visualize measured person geometry on copies of source and model pixels."""
import math

BOX_COLOR = (255, 220, 0)
POSE_COLOR = (255, 100, 255)
FACE_COLORS = ((0, 240, 255), (0, 240, 255), (255, 80, 80),
               (150, 255, 80), (150, 255, 80))


def _draw(image, bounds, points):
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    painter.rectangle((max(0, left), max(0, top), min(image.width-1, right),
                       min(image.height-1, bottom)), outline=BOX_COLOR, width=2)
    drawn = []
    for index, (x, y) in enumerate(points):
        # The adapter replaces rejected/missing measurements with (-1,-1).
        # Do not clamp them into fabricated points at the image border.
        if not (math.isfinite(x) and math.isfinite(y) and
                0 <= x < image.width and 0 <= y < image.height):
            continue
        color = POSE_COLOR if index < 17 else FACE_COLORS[index-17]
        painter.ellipse((x-2, y-2, x+2, y+2), fill=color)
        label = str(index)
        box = painter.textbbox((0, 0), label)
        position = (max(0, min(x+4, image.width-(box[2]-box[0])-2)),
                    max(0, min(y-8, image.height-(box[3]-box[1])-2)))
        painter.text(position, label, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
        drawn.append(index)
    return drawn


def export(image, model_image, face, person, normalized_points, path):
    from .alignment_audit import _png
    source_path = path.with_name(path.name+'-person-source.png')
    input_path = path.with_name(path.name+'-person-input-annotated.png')
    source_points = list(person['keypoints']) + list(face.landmarks)
    with image.convert('RGB') as source:
        source_drawn = _draw(source, person['bounds'], source_points)
        _png(source, source_path)

    # Match prepare_person_input exactly, including odd square-pad remainders.
    left, top, right, bottom = person['bounds']
    width, height = right-left, bottom-top
    side = max(width, height)
    padx, pady = (side-width)//2, (side-height)//2
    sx, sy = model_image.width/side, model_image.height/side
    input_bounds = [padx*sx, pady*sy, (padx+width)*sx, (pady+height)*sy]
    points = [(x*model_image.width, y*model_image.height) for x, y in normalized_points]
    with model_image.copy() as annotated:
        input_drawn = _draw(annotated, input_bounds, points)
        _png(annotated, input_path)
    return dict(source_file=source_path.name, annotated_input_file=input_path.name,
        annotations=dict(index_base=0, person_bounds_color=list(BOX_COLOR),
            pose_color=list(POSE_COLOR), face_colors=[list(c) for c in FACE_COLORS],
            pose_indices=list(range(17)), face_indices=list(range(17, 22)),
            source_point_order='COCO17 pose followed by five face-detector landmarks',
            input_point_order='SapiensID-canonicalized pose followed by five face-detector landmarks',
            source_drawn_indices=source_drawn, input_drawn_indices=input_drawn,
            input_person_bounds=input_bounds,
            note='Source shows measured pose; input shows exact prepared keypoints. Missing points are omitted.'))
