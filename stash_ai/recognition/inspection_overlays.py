"""Selected and linked geometry on preview copies; no inference or pixel writes."""
import json
import math

from PIL import Image, ImageDraw

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


def render(image, geometry, layers=None):
    """Return an annotated copy and a caption; source bounds must be known."""
    if image is None:
        return image, ''
    bounds = geometry.get('bounds')
    if not _numbers(bounds,4) or not (0 <= bounds[0] < bounds[2] and 0 <= bounds[1] < bounds[3]):
        return image, 'Geometry unavailable: source crop bounds missing or invalid.'
    left, top, right, bottom = bounds
    layers=layers or dict(face_boxes=True,body_boxes=True,landmarks=True,keypoints=True,skeleton=True,masks=False)
    sx, sy = image.width/(right-left), image.height/(bottom-top)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    shown = [];masks_shown=0
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
        if shape.get('context') and not shape['selected']:color=tuple(c//2 for c in color)
        if kind=='body' and layers.get('masks'):
            packed=data.get('pose_refinement',{}).get('encoder_mask')
            if packed:
                try:
                    from .person_geometry import unpack_mask
                    w,h=round(box[2]-box[0]),round(box[3]-box[1])
                    if not 0<w*h<=4000000:raise ValueError('Mask too large')
                    with Image.fromarray(unpack_mask(packed,w,h).astype('uint8')*70) as mask:
                        with mask.transform(image.size,Image.Transform.AFFINE,(1/sx,0,left-box[0],0,1/sy,top-box[1])) as alpha:
                            annotated.paste(COLORS['body'],(0,0,*image.size),alpha)
                            masks_shown+=1
                except (ValueError,TypeError,OverflowError):pass
        if kind == 'body' and len(mapped) == 17 and layers.get('skeleton'):
            for a,b in SKELETON:
                if mapped[a] is not None and mapped[b] is not None:
                    draw.line((mapped[a],mapped[b]), fill=color, width=1)
        if layers.get('face_boxes' if kind=='face' else 'body_boxes'):
            draw.rectangle((min(image.width-1,max(0,(clipped[0]-left)*sx)),min(image.height-1,max(0,(clipped[1]-top)*sy)),
                        min(image.width-1,(clipped[2]-left)*sx),min(image.height-1,(clipped[3]-top)*sy)),
                       outline=color, width=2 if shape['selected'] else 1)
        for point in mapped if layers.get('landmarks' if kind=='face' else 'keypoints') else ():
            if point is not None:
                x,y = point
                radius = 2 if kind == 'face' else 3
                draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=color)
        shown.append(f"{'Selected' if shape['selected'] else ('Context' if shape.get('context') else 'Linked')} {kind}: {'yellow' if kind=='face' else 'green'} ({sum(p is not None for p in mapped)} points)")
    caption = '; '.join(shown) if shown else 'No retained geometry intersects this crop.'
    if layers.get('masks') and not masks_shown:caption+=' · No usable retained mask in this viewport.'
    return annotated, caption + ((' · '+' '.join(geometry['notes'])) if geometry['notes'] else '')


def point_info(geometry,size,x,y):
    from .pose_anchor_rules import POINTS
    bounds=geometry.get('bounds')
    if not _numbers(bounds,4) or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:return ''
    sx,sy=size[0]/(bounds[2]-bounds[0]),size[1]/(bounds[3]-bounds[1])
    found=[]
    for shape in geometry['shapes']:
        data=shape['geometry'];face=shape['kind']=='face'
        if face and not isinstance(data,list) or not face and not isinstance(data,dict):continue
        points=[data[i:i+2] for i in range(4,min(14,len(data)),2)] if face else data.get('keypoints') or []
        if not isinstance(points,(list,tuple)):continue
        for i,p in enumerate(points[:5] if face else points[:17]):
            if not _numbers(p,2) or min(p)<0:continue
            d=((p[0]-bounds[0])*sx-x)**2+((p[1]-bounds[1])*sy-y)**2
            if d<=64:
                scores=[] if face else data.get('keypoint_scores')
                confidence=scores[i] if isinstance(scores,(list,tuple)) and i<len(scores) else None
                found.append((d,f'{"Face landmark "+str(i) if face else POINTS[i]} · source ({p[0]:g}, {p[1]:g}) · confidence: {confidence if confidence is not None else "not retained"}'))
    return min(found,key=lambda p:p[0])[1] if found else ''
