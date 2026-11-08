"""Revisit ambiguous retained boxes using independent subsequent VTT torso pairs."""
import json
from collections import defaultdict
from contextlib import closing

from . import body_observations
from .body_observations import contract, save_records
from .crowd_context import pair, follows
from .config import digest
from .database import encoded
from .dependencies import supported, version


def plans(db, scan, expected):
    groups = defaultdict(list)
    for row in db.execute("""SELECT cue,id,seconds,geometry,rejected,
            json_extract(quality,'$.quality.view_shot') AS shot
            FROM body_observations WHERE scan=? AND contract=? ORDER BY cue,id""",(scan,expected)):
        groups[row['cue']].append(row)
    clean = {}
    for cue,rows in groups.items():
        if len(rows)==2 and all(r['rejected'] is None for r in rows):
            clean[cue] = pair([json.loads(r['geometry']) for r in rows])
    for cue, in db.execute('SELECT cue FROM body_cues WHERE scan=? AND contract=? ORDER BY cue',(scan,expected)):
        if cue in groups or not follows(clean.get(cue-1),clean.get(cue+1)):
            continue
        before,after = groups[cue-1],groups[cue+1]
        shot = before[0]['shot']
        if (shot is None or any(row['shot']!=shot for row in (*before,*after)) or
                not 0<after[0]['seconds']-before[0]['seconds']<=24):
            continue
        faces = list(map(tuple,db.execute('SELECT id,geometry,rejected FROM observations WHERE scan=? AND cue=? '
                                         'ORDER BY id',(scan,cue))))
        stamp = digest([[(r['id'],r['geometry'],r['seconds'],r['shot']) for r in (*before,*after)],faces])
        yield cue,[],shot,clean[cue-1],stamp
    for cue,rows in groups.items():
        conflict = any(r['rejected']=='person_mask_conflict' for r in rows)
        if len(rows)!=1 and not (len(rows)==2 and conflict):
            continue
        shot = rows[0]['shot']
        if shot is None:
            continue
        for later in range(cue+1,cue+5):
            near = groups.get(later,())
            if not near or any(r['shot']!=shot for r in near) or not 0<near[0]['seconds']-rows[0]['seconds']<=12:
                break
            far = groups.get(later+1,())
            if (far and all(r['shot']==shot for r in far) and 0<far[0]['seconds']-near[0]['seconds']<=12
                    and follows(clean.get(later),clean.get(later+1))):
                boxes = [json.loads(r['geometry']).get('bounds') for r in rows]
                if not all(boxes):
                    break
                left,top = min(b[0] for b in boxes),min(b[1] for b in boxes)
                right,bottom = max(b[2] for b in boxes),max(b[3] for b in boxes)
                if all(left<=x<right and top<=y<bottom for x,y in clean[later][0]):
                    evidence = [(r['id'],r['geometry'],r['rejected']) for group in (rows,near,far) for r in group]
                    faces = list(map(tuple, db.execute('SELECT id,geometry,rejected FROM observations '
                                     'WHERE scan=? AND cue=? ORDER BY id',(scan,cue))))
                    yield cue, rows, shot, clean[later], digest([evidence,faces])
                break


def recover(collector, db, scan, source):
    config = collector.config
    if not body_observations.enabled(config) or config.data['sapiensid']['pose']['backend']=='yolo':
        return
    expected = contract(config)
    key = 'future-crowd:'+scan
    audited = supported(db)
    def generation():
        return [expected,version(db,'raw-body',scan),version(db,'raw-observations',scan)] if audited else None
    saved = db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
    prior = json.loads(saved[0]) if saved else {}
    if audited and prior.get('generation')==generation():
        return
    attempted = prior.get('attempted',{}) if prior.get('contract')==expected else {}
    wanted = {cue:(rows,shot,support,stamp) for cue,rows,shot,support,stamp in plans(db,scan,expected)
              if attempted.get(str(cue))!=stamp}
    if wanted:
        collector._load()
        indices = sorted(wanted)
        frames = source.frames_indices(indices) if hasattr(source,'frames_indices') else source.frames(indices[0])
        with closing(frames):
            for cue,timing,image in frames:
                with image:
                    if cue in wanted:
                        rows,shot,support,stamp = wanted[cue]
                        _retry(collector.model,db,config,scan,cue,image,rows,shot,support,seconds=timing.start)
                        attempted[str(cue)] = stamp
    with db:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,encoded(dict(
            contract=expected, generation=generation(),attempted=attempted))))


def _retry(model,db,config,scan,cue,image,rows,shot,support,*,seconds=None):
    from .body_collection import collect
    from .body_perspectives import frame_signature
    from .crowded_retry import retry_people
    from .models import checked_detection
    from .person_pipeline import PreparedCollector
    from .sapiensid_backend import associate
    neighbours = []
    if rows:
        bounds = [json.loads(r['geometry'])['bounds'] for r in rows]
        seconds = rows[0]['seconds']
    else:
        import numpy as np
        from .perspective_shots import change
        expected = contract(config)
        signature = np.frombuffer(bytes.fromhex(frame_signature(image)),dtype='uint8').astype(float).reshape(-1,3)
        bounds = []
        for side in (cue-1,cue+1):
            people = []
            for stored,geometry,quality in db.execute('SELECT seconds,geometry,quality FROM body_observations '
                    'WHERE scan=? AND contract=? AND cue=? ORDER BY id',(scan,expected,side)):
                q = json.loads(quality).get('quality',{})
                if (not q.get('frame_signature') or q.get('view_shot')!=shot or
                        not 0 < (seconds-stored)*(cue-side) <= 12):
                    return
                other = np.frombuffer(bytes.fromhex(q['frame_signature']),dtype='uint8').astype(float).reshape(-1,3)
                if change(other,signature,True)!='continuous':
                    return
                person = json.loads(geometry)
                people.append(person)
                bounds.append(person['bounds'])
            neighbours.append(pair(people))
    box = [min(b[0] for b in bounds),min(b[1] for b in bounds),max(b[2] for b in bounds),max(b[3] for b in bounds)]
    faces = list(db.execute('SELECT id,geometry,rejected FROM observations WHERE scan=? AND cue=? ORDER BY id',
                            (scan,cue)))
    detections = [checked_detection(json.loads(r['geometry'])) for r in faces]
    image._recognition_cue_signature = cue,shot,frame_signature(image)
    image._recognition_crowd = scan,cue,shot,seconds,support
    recovered = retry_people(model,image,detections,[dict(bounds=box)])
    if len(recovered)!=2 or not all(p.get('crowded_box_retry') for p in recovered):
        return
    recovered_pair = pair([dict(p,crowded_box_retry=False) for p in recovered])
    if not all(follows(previous,recovered_pair) for previous in (support,*neighbours)):
        return
    for person in recovered:
        person['future_torso_retry'] = True
    model.frame_people = recovered
    model.face_matches = associate(detections,recovered)
    records = []
    with closing(PreparedCollector(config,model,writer=lambda _,batch:records.extend(batch))) as pipeline:
        collect(pipeline,db,model,config,scan,cue,seconds,image,faces,detections)
        pipeline.flush()
    ready = {oid for oid,metadata,feature,_ in records if '_body' in metadata and feature is not None}
    if len(ready)!=2:
        return
    obsolete = sorted({r['id'] for r in rows} - ready)
    with db:
        for oid in obsolete:
            db.execute('DELETE FROM body_members WHERE observation=?',(oid,))
            db.execute('DELETE FROM body_origins WHERE observation=?',(oid,))
            db.execute('DELETE FROM body_observations WHERE id=?',(oid,))
        save_records(db,config,records)
