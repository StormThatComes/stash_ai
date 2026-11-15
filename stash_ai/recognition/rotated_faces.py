"""One pose-guided SCRFD retry for a measured, sideways head without a face."""
import json
import math
import uuid

from PIL import Image

from .body_observations import contract, enabled as collect_bodies, repair_projections
from .database import encoded
from .models import checked_detection
from stash_ai.runtime.cleanup import closing


def enabled(config):
    return (config.data['detection'].get('pose_guided_rotation', False) and collect_bodies(config)
            and config.data['models']['detector'].startswith('scrfd_'))


def head_crop(person, size):
    points = person.get('keypoints', ())
    if len(points)<3:
        return None
    nose, left, right = points[:3]
    vis = person.get('keypoint_visibility')
    if (vis is not None and min(vis[:3])<.5 or
            any(len(p)!=2 or not all(math.isfinite(v) and v>=0 for v in p) for p in (nose,left,right))):
        return None
    dx,dy = left[0]-right[0], left[1]-right[1]
    eyes = math.hypot(dx,dy)
    if eyes<4 or abs(dy)<1.73*abs(dx) or math.dist(nose,left)>2*eyes or math.dist(nose,right)>2*eyes:
        return None
    cx,cy = (sum(p[i] for p in (nose,left,right))/3 for i in (0,1))
    radius = 2.6*eyes
    box = (max(0,math.floor(cx-radius)),max(0,math.floor(cy-radius)),
           min(size[0],math.ceil(cx+radius)),min(size[1],math.ceil(cy+radius)))
    if min(box[2]-box[0],box[3]-box[1])<8:
        return None
    return box, dy>0


def restore(face, box, ccw):
    face = checked_detection(getattr(face,'row',face))
    l,t,r,b = box
    def point(x,y,edge=False):
        offset = 0 if edge else 1
        return (r-offset-y,t+x) if ccw else (l+y,b-offset-x)
    x,y,w,h = face.box
    a,c = point(x,y,True),point(x+w,y+h,True)
    marks = [v for xy in face.landmarks for v in point(*xy)]
    return checked_detection([min(a[0],c[0]),min(a[1],c[1]),h,w,*marks,face.score])


def recover(db, scan, source, config, acquire, visibility, release):
    if not enabled(config):
        return
    from .scan_pipeline import process
    from .scanning import geometry_contract, assess, rejection, crop_bytes, safe_url
    from .sapiensid_backend import associate
    from .cold_payloads import put
    from .pixel_storage import compact
    expected = contract(config)
    from .dependencies import supported, version, store
    durable = supported(db)
    def certify():
        if durable:
            with db:
                store(db,'rotated-head-v1',scan,expected,version(db,'raw-body',scan),expected)
    if durable and db.execute('SELECT 1 FROM evidence_fingerprints WHERE kind=? AND key=? AND policy=? AND version=? AND digest=?',
                              ('rotated-head-v1',scan,expected,version(db,'raw-body',scan),expected)).fetchone():
        return
    planned = {}
    for row in db.execute("""SELECT id,cue,geometry FROM body_observations WHERE scan=? AND contract=?
            AND face IS NULL AND json_extract(quality,'$.rotated_face_checked') IS NULL ORDER BY cue,id""",(scan,expected)):
        cue = source.cues[row['cue']]
        geometry = json.loads(row['geometry'])
        crop = head_crop(geometry,cue.rect[2:])
        if crop:
            planned.setdefault(row['cue'],{})[row['id']] = crop
    if not planned:
        certify()
        return
    from .backend import prepared_frame
    with closing(acquire(),name='close rotated head face backend') as backend:
        try:
            frames = source.frames_indices(sorted(planned)) if hasattr(source,'frames_indices') else source.frames(min(planned))
            completed = set()
            with closing(frames,name='close rotated head VTT source') as stream:
                for index,cue,image in stream:
                    with image:
                        if index not in planned:
                            continue
                        rows = list(db.execute('SELECT id,geometry FROM body_observations WHERE scan=? AND contract=? AND cue=? ORDER BY id',
                                               (scan,expected,index)))
                        people = [json.loads(row['geometry']) for row in rows]
                        faces = [checked_detection(json.loads(raw)) for raw, in db.execute(
                            'SELECT geometry FROM observations WHERE scan=? AND cue=? ORDER BY id',(scan,index))]
                        with db:
                            for i,row in enumerate(rows):
                                if row['id'] not in planned[index]:
                                    continue
                                box,ccw = planned[index][row['id']]
                                db.execute("UPDATE body_observations SET quality=json_set(quality,'$.rotated_face_checked',1) WHERE id=?",(row['id'],))
                                if len(faces)>=config.data['detection']['max_faces_per_frame']:
                                    continue
                                with image.crop(box) as tile, tile.transpose(Image.Transpose.ROTATE_90 if ccw else Image.Transpose.ROTATE_270) as upright:
                                    found = backend.detect(upright)
                                if len(found)!=1:
                                    continue
                                face = restore(found[0],box,ccw)
                                # An existing face at this location or a contested assignment vetoes recovery.
                                x,y,w,h = face.box
                                if any(min(x+w,a+c)>max(x,a) and min(y+h,b+d)>max(y,b) for a,b,c,d in (f.box for f in faces)):
                                    continue
                                if associate(faces+[face],people)[-1]!=i:
                                    continue
                                with prepared_frame(backend,image):
                                    records = list(process(image,[face],backend,visibility,config,cue,assess=assess,
                                        rejection=rejection,crop_bytes=crop_bytes,geometry_contract=geometry_contract))
                                for _,face,quality,reason,vector,crop in records:
                                    if reason is not None:
                                        continue
                                    oid = str(uuid.uuid5(uuid.UUID(scan),encoded(['rotated-head-v1',index,row['id']])))
                                    quality.update(rotated_head_source=row['id'],rotated_head_degrees=90 if ccw else -90)
                                    shot = db.execute("SELECT json_extract(quality,'$.quality.view_shot') FROM body_observations WHERE id=?",(row['id'],)).fetchone()[0]
                                    if shot is not None:
                                        quality['view_shot'] = shot
                                    db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                        (oid,scan,index,cue.start,safe_url(cue.image),encoded(list(cue.rect)),encoded(list(face.row)),
                                         compact(db,encoded(quality)),None,backend.dimensions,vector,put(db,crop)))
                                    db.execute("UPDATE body_observations SET face=?,quality=json_set(quality,'$.face_status','accepted') WHERE id=?",
                                               (oid,row['id']))
                                    faces.append(face)
                            repair_projections(db,config,scan,cue=index)
                        completed.add(index)
            if completed != planned.keys():
                raise ValueError('The VTT source did not supply every requested sideways head cue.')
        finally:
            release()
    certify()
