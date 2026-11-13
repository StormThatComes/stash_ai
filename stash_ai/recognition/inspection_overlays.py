"""Selected and linked geometry on preview copies; no inference or pixel writes."""
import json
import math

from PIL import ImageDraw

from .body_observations import available, contract as body_contract
from .person_signal import contract as person_contract

COLORS = {'face': (255, 220, 0), 'body': (80, 230, 120)}
SKELETON = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def _json(value, default):
    try:
        result = json.loads(value)
        return result if isinstance(result, type(default)) else default
    except (ValueError, TypeError):
        return default


def _unique(rows, current):
    # The query orders the current contract first. Never combine different poses
    # from historical detector contracts into one seemingly measured skeleton.
    if len(rows) == 1 or len(rows) > 1 and rows[0]['contract'] == current and rows[1]['contract'] != current:
        return rows[0]
    return None


def read(db, config, selector, kind, address):
    table = 'observations' if kind == 'face' else 'body_observations'
    columns = 'o.scan,o.cue,o.geometry,o.quality' + (',o.face' if kind == 'body' else '')
    row = db.execute(f'SELECT {columns} FROM {table} o JOIN scans s ON s.id=o.scan '
        'WHERE o.id=? AND s.scope=? AND s.namespace=?', (address, selector['scope'], selector['namespace'])).fetchone()
    if row is None:
        return dict(bounds=None, shapes=[], notes=['Geometry unavailable in this scope and model.'])
    quality = _json(row['quality'], {})
    notes, shapes = [], []
    if kind == 'face':
        face = _json(row['geometry'], [])
        shapes.append(dict(kind='face', selected=True, geometry=face))
        bounds = quality.get('crop_bounds')
        rows = []
        current = body_contract(config)
        if available(db):
            rows = db.execute('SELECT geometry,contract FROM body_observations WHERE face=? AND scan=? AND cue=? '
                'ORDER BY (contract=?) DESC,id LIMIT 2', (address,row['scan'],row['cue'],current)).fetchall()
        linked = _unique(rows, current)
        if linked is not None:
            shapes.append(dict(kind='body', selected=False, geometry=_json(linked['geometry'], {})))
        elif not rows and db.execute("SELECT 1 FROM sqlite_master WHERE name='person_evidence'").fetchone():
            current = person_contract(config)
            rows = db.execute('SELECT metadata,contract FROM person_evidence WHERE observation=? '
                'ORDER BY (contract=?) DESC,contract LIMIT 2', (address,current)).fetchall()
            linked = _unique(rows, current)
            if linked is not None:
                shapes.append(dict(kind='body', selected=False, geometry=_json(linked['metadata'], {}).get('person') or {}))
        if linked is not None and linked['contract'] != current:
            notes.append('Linked pose uses a historical collection contract.')
        if rows and linked is None:
            notes.append('Ambiguous linked pose contracts; pose overlay omitted.')
    else:
        body = _json(row['geometry'], {})
        bounds = body.get('bounds')
        shapes.append(dict(kind='body', selected=True, geometry=body))
        face = db.execute('SELECT geometry FROM observations WHERE id=? AND scan=? AND cue=?',
                          (row['face'],row['scan'],row['cue'])).fetchone()
        if face is not None:
            shapes.append(dict(kind='face', selected=False, geometry=_json(face['geometry'], [])))
    return dict(bounds=bounds, shapes=shapes, notes=notes)


def _numbers(value, count):
    return isinstance(value, (list,tuple)) and len(value) == count and all(
        type(v) in (int,float) and math.isfinite(v) for v in value)


def render(image, geometry):
    """Return an annotated copy and a caption; source bounds must be known."""
    if image is None:
        return image, ''
    bounds = geometry.get('bounds')
    if not _numbers(bounds,4) or not (0 <= bounds[0] < bounds[2] and 0 <= bounds[1] < bounds[3]):
        return image, 'Geometry unavailable: source crop bounds missing or invalid.'
    left, top, right, bottom = bounds
    sx, sy = image.width/(right-left), image.height/(bottom-top)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    shown = []
    # Smaller face markers on top of larger pose markers keep coincident points
    # distinguishable. Selection is indicated by box width and caption.
    for shape in sorted(geometry['shapes'], key=lambda s: (s['kind'] == 'face', s['selected'])):
        kind, data = shape['kind'], shape['geometry']
        if kind == 'face':
            if not isinstance(data,list) or len(data) < 4 or not _numbers(data[:4],4) or min(data[2:4]) <= 0:
                continue
            x,y,w,h = data[:4]
            box, points = [x,y,x+w,y+h], [data[i:i+2] for i in range(4,min(14,len(data)),2)]
        else:
            if not isinstance(data,dict):
                continue
            box, points = data.get('bounds'), data.get('keypoints') or []
            if not isinstance(points,(list,tuple)) or len(points) != 17:
                points = []
        if not _numbers(box,4) or not (box[0] < box[2] and box[1] < box[3]):
            continue
        clipped = (max(left,box[0]),max(top,box[1]),min(right,box[2]),min(bottom,box[3]))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            continue
        mapped = [((p[0]-left)*sx,(p[1]-top)*sy) if _numbers(p,2) and
                  left <= p[0] < right and top <= p[1] < bottom else None for p in points]
        color = COLORS[kind]
        if kind == 'body' and len(mapped) == 17:
            for a,b in SKELETON:
                if mapped[a] is not None and mapped[b] is not None:
                    draw.line((mapped[a],mapped[b]), fill=color, width=1)
        draw.rectangle((min(image.width-1,max(0,(clipped[0]-left)*sx)),min(image.height-1,max(0,(clipped[1]-top)*sy)),
                        min(image.width-1,(clipped[2]-left)*sx),min(image.height-1,(clipped[3]-top)*sy)),
                       outline=color, width=2 if shape['selected'] else 1)
        for point in mapped:
            if point is not None:
                x,y = point
                radius = 2 if kind == 'face' else 3
                draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=color)
        shown.append(f"{'Selected' if shape['selected'] else 'Linked'} {kind}: {'yellow' if kind=='face' else 'green'} ({sum(p is not None for p in mapped)} points)")
    caption = '; '.join(shown) if shown else 'No retained geometry intersects this crop.'
    return annotated, caption + ((' · '+' '.join(geometry['notes'])) if geometry['notes'] else '')
