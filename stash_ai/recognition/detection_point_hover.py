"""Source-coordinate hit targets shared by recorded phases and the final preview."""
from dataclasses import dataclass
import math

from .detection_point_details import valid_xy
from .pose_anchor_rules import POINTS


@dataclass(frozen=True)
class Target:
    x: float
    y: float
    text: str
    rejected: bool = False


def locations(person, size):
    for k, point in enumerate(person.points):
        rejected = not valid_xy(point)
        if rejected:
            if k >= len(person.point_reasons) or not person.point_reasons[k] or k >= len(person.raw_points):
                continue
            point = person.raw_points[k]
        if valid_xy(point) and point[0] < size[0] and point[1] < size[1]:
            yield k, point, rejected


def value(values, index):
    return f'{values[index]:.4f}' if index < len(values) and math.isfinite(values[index]) else 'N/A'


def targets(people, title, size):
    result = []
    for person in people:
        probabilities = dict(person.probabilities)
        for k, (x, y), rejected in locations(person, size):
            name = POINTS[k] if k < len(POINTS) else f'point {k}'
            lines = [f'{person.label} | {name} (#{k})', f'Phase: {title}', f'Position: ({x:.2f}, {y:.2f})',
                     f'Score ({person.score_kind}): {value(person.scores, k)}']
            for key in ('presence', 'visibility'):
                measured = value(probabilities.get(key, ()), k)
                if person.probability_kind != 'measured':
                    measured = ('not measured' + (f' ({person.probability_kind}: {measured})'
                                if measured != 'N/A' else ''))
                lines.append(f'{key.capitalize()}: {measured}')
            if 'flip_agreement' in probabilities:
                lines.append(f"Flip agreement: {value(probabilities['flip_agreement'], k)}")
            state = person.states[k] if k < len(person.states) else 'available'
            lines.append('State: ' + ('filtered / ' + state if rejected else state))
            if k < len(person.point_reasons):
                lines.extend(person.point_reasons[k])
            if person.note:
                lines.append('Pose: ' + person.note)
            result.append(Target(x, y, '\n'.join(lines), rejected))
    return tuple(result)


def face_targets(record):
    result = []
    for face in record['faces']:
        source = face.get('landmarks', ())
        for k, point in enumerate(face.get('display_landmarks', source)):
            if not valid_xy(point):
                continue
            xy = source[k] if k < len(source) else point
            text = (f"F{face['face_index']} | face landmark #{k}\nPhase: Final result\n"
                    f'Position: ({xy[0]:.2f}, {xy[1]:.2f})\nPoint score: N/A\n'
                    f"Face detection score: {face['score']:.4f}\nPresence: not measured\nVisibility: not measured")
            result.append(Target(*point, text))
    return tuple(result)


def nearest(points, x, y, zoom, radius=8):
    distances = sorted(((math.hypot(p.x-x, p.y-y)*zoom, i) for i, p in enumerate(points)))
    if not distances or distances[0][0] > radius:
        return ()
    limit = min(radius, distances[0][0] + .5)
    return tuple(points[i] for distance, i in distances if distance <= limit)


def draw_rejected(image, people, size):
    from PIL import ImageDraw
    painter = ImageDraw.Draw(image)
    for person in people:
        for k, (x, y), rejected in locations(person, size):
            if rejected:
                painter.ellipse((x-4, y-4, x+4, y+4), outline=(210, 210, 210), width=1)
                painter.text((x+5, max(0, y-9)), str(k), fill=(210, 210, 210),
                             stroke_width=1, stroke_fill='black')
