"""Bounded pose-guided SCRFD recovery, including one-eye profiles."""
import json
import math
import uuid

from PIL import Image

from .body_observations import contract, enabled as collect_bodies, repair_projections
from .database import encoded
from .models import checked_detection
from stash_ai.runtime.cleanup import closing

CONTRACT = 'rotated-head-v4-profile-repair'


def enabled(config):
    return (config.data['detection'].get('pose_guided_rotation', False) and collect_bodies(config)
            and config.data['models']['detector'].startswith('scrfd_'))


def reliable_head_points(person, size):
    points = person.get('keypoints', ())
    confidence=person.get('keypoint_scores')
    visibility=person.get('keypoint_visibility')
    states=person.get('keypoint_states')
    details = person.get('pose_refinement', {})
    borrowed = {i for i, _ in details.get('temporal_prompts', ())}
    if details.get('face_prompt'):
        borrowed.add(details['face_prompt'][0])
    from .contact_upper_body import _conflicts
    borrowed.update(_conflicts(person))
    bounds = person.get('bounds', (0,0,*size))
    def reliable(i,p):
        return (i not in borrowed and len(p)==2 and all(math.isfinite(v) and v>=0 for v in p)
                and 0<=p[0]<size[0] and 0<=p[1]<size[1]
                and bounds[0]<=p[0]<bounds[2] and bounds[1]<=p[1]<bounds[3]
                and (confidence is None or i<len(confidence) and math.isfinite(confidence[i]) and confidence[i]>=.5)
                and (visibility is None or i<len(visibility) and math.isfinite(visibility[i]) and visibility[i]>=.5)
                and (states is None or i<len(states) and states[i] not in ('conflicted','occluded','unavailable','recovered')))
    return {i:p for i,p in enumerate(points[:5]) if reliable(i,p)}


def head_crop(person, size):
    points = reliable_head_points(person,size)
    usable = list(points.values())
    if len(usable)<2:
        return None
    def uncertain():
        span=max(math.dist(a,b) for a in usable for b in usable)
        if span<4:return None
        cx,cy=(sum(p[i] for p in usable)/len(usable) for i in (0,1))
        radius=2.6*span
        box = (max(0,math.floor(cx-radius)),max(0,math.floor(cy-radius)),
               min(size[0],math.ceil(cx+radius)),min(size[1],math.ceil(cy+radius)))
        # A missing far eye does not mean an upright head. Retain uncertainty
        # and verify bounded quarter-turn candidates against the original pose.
        return (box, None) if min(box[2]-box[0],box[3]-box[1])>=8 else None
    if not all(i in points for i in range(3)):
        return uncertain()
    nose, left, right = (points[i] for i in range(3))
    dx,dy = left[0]-right[0], left[1]-right[1]
    eyes = math.hypot(dx,dy)
    if eyes<4 or math.dist(nose,left)>2*eyes or math.dist(nose,right)>2*eyes:
        return uncertain()
    cx,cy = (sum(p[i] for p in (nose,left,right))/3 for i in (0,1))
    radius = 2.6*eyes
    box = (max(0,math.floor(cx-radius)),max(0,math.floor(cy-radius)),
           min(size[0],math.ceil(cx+radius)),min(size[1],math.ceil(cy+radius)))
    if min(box[2]-box[0],box[3]-box[1])<8:
        return None
    if abs(dy)>=1.73*abs(dx):
        return box, dy>0
    return box, 180 if nose[1] < (left[1]+right[1])/2 else 0


def restore(face, box, ccw):
    if type(ccw) is not bool and not (type(ccw) is int and ccw in (0,180)):
        raise ValueError('Face crop rotation must be a known quarter turn.')
    face = checked_detection(getattr(face,'row',face))
    l,t,r,b = box
    def point(x,y,edge=False):
        offset = 0 if edge else 1
        if type(ccw) is int:
            return (r-offset-x,b-offset-y) if ccw==180 else (l+x,t+y)
        return (r-offset-y,t+x) if ccw else (l+y,b-offset-x)
    x,y,w,h = face.box
    a,c = point(x,y,True),point(x+w,y+h,True)
    marks = [v for xy in face.landmarks for v in point(*xy)]
    return checked_detection([min(a[0],c[0]),min(a[1],c[1]),(w if type(ccw) is int else h),(h if type(ccw) is int else w),*marks,face.score])


def detect_head(backend, image, box, turn, people, owner, faces):
    from .quality import alignment_measurements
    from .sapiensid_backend import associate
    from .face_landmark_repair import same_box
    measured = []
    for person in people:
        points = reliable_head_points(person,image.size)
        head = [points.get(i,[-1,-1]) for i in range(5)]
        measured.append(dict(person,keypoints=head+list(person['keypoints'][5:])))
    options = []
    rotations = (0, True, 180, False) if turn is None else (turn,)
    for ccw in rotations:
        with image.crop(box) as tile, (tile.copy() if type(ccw) is int and ccw==0 else tile.transpose(
                Image.Transpose.ROTATE_180 if ccw==180 else Image.Transpose.ROTATE_90 if ccw else Image.Transpose.ROTATE_270)) as upright:
            found = backend.detect(upright)
        if len(found)!=1:
            continue
        face = restore(found[0],box,ccw)
        quality = alignment_measurements(face,image)
        if not quality['alignment_valid'] or any(same_box(face,other) or duplicate_face(face,other) for other in faces):
            continue
        if associate(faces+[face],measured)[-1]!=owner:
            continue
        options.append((quality['alignment_error'],-face.score,face.row,face,ccw))
    if not options or any(not duplicate_face(a[3],b[3]) for i,a in enumerate(options) for b in options[i+1:]):
        return None
    selected = min(options,key=lambda value:value[:3])
    return selected[3],selected[4]


def recover(db, scan, source, config, acquire, visibility, release):
    if not enabled(config):
        return
    from .scan_pipeline import process
    from .scanning import geometry_contract, assess, rejection, crop_bytes, safe_url
    from .cold_payloads import put
    from .pixel_storage import compact
    expected = contract(config)
    from .dependencies import supported, version, store
    durable = supported(db)
    def certify():
        if durable:
            with db:
                store(db,CONTRACT,scan,expected,version(db,'raw-body',scan),expected)
    if durable and db.execute('SELECT 1 FROM evidence_fingerprints WHERE kind=? AND key=? AND policy=? AND version=? AND digest=?',
                              (CONTRACT,scan,expected,version(db,'raw-body',scan),expected)).fetchone():
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
                                recovered = detect_head(backend,image,box,ccw,people,i,faces)
                                if recovered is None:
                                    continue
                                face,ccw = recovered
                                with prepared_frame(backend,image):
                                    records = list(process(image,[face],backend,visibility,config,cue,assess=assess,
                                        rejection=rejection,crop_bytes=crop_bytes,geometry_contract=geometry_contract))
                                for _,face,quality,reason,vector,crop in records:
                                    if reason is not None:
                                        continue
                                    oid = str(uuid.uuid5(uuid.UUID(scan),encoded([CONTRACT,index,row['id']])))
                                    quality.update(rotated_head_source=row['id'],rotated_head_degrees=ccw if type(ccw) is int else 90 if ccw else -90)
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


def duplicate_face(face,other):
    x,y,w,h=face.box; a,b,c,d=other.box
    intersection=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
    if intersection/max(1,min(w*h,c*d))<.5:return False
    import numpy as np
    distance=np.median(np.linalg.norm(np.asarray(face.landmarks)-np.asarray(other.landmarks),axis=1))
    return distance < .4*min(w,h,c,d)
