"""Prepare all detected people once, optionally sharing their evidence with a face."""
import hashlib
import numpy as np

from .body_observations import contract, identifier


def _pixels(image,bounds,mask=None):
    left,top,right,bottom=bounds
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb, rgb.convert('L') as gray:
        values=np.asarray(gray,dtype=np.float32)
        lap=(4*values[1:-1,1:-1]-values[:-2,1:-1]-values[2:,1:-1]
             -values[1:-1,:-2]-values[1:-1,2:])
        if mask is not None:
            interior = (mask[1:-1,1:-1] & mask[:-2,1:-1] & mask[2:,1:-1]
                        & mask[1:-1,:-2] & mask[1:-1,2:])
            lap = lap[interior]
        sharpness=float(lap.var()) if lap.size else 0.
        pixel_digest=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
        with gray.resize((9,8)) as thumb:
            small=np.asarray(thumb)
            bits=(small[:,1:]>small[:,:-1]).flatten()
            appearance_hash=format(sum(int(v)<<i for i,v in enumerate(bits)),'016x')
    return sharpness,pixel_digest,appearance_hash


def quality(image,person,people,settings):
    from .inference_cache import many
    bounds=person['bounds'];left,top,right,bottom=bounds
    packed = person.get('pose_refinement',{}).get('encoder_mask')
    if packed:
        from .person_geometry import encoder_crop, unpack_mask
        width,height = right-left,bottom-top
        mask = unpack_mask(packed,width,height)
        with encoder_crop(image,bounds,packed) as crop:
            sharpness,pixel_digest,appearance_hash = _pixels(crop,(0,0,width,height),mask)
    else:
        sharpness,pixel_digest,appearance_hash=many('body_quality','body-pixels-v1',[(image,bounds)],
            lambda values:[_pixels(pixels,box) for pixels,box in values])[0]
    overlap=0.
    for other in people:
        if other is person:continue
        a,b,c,d=other['bounds']
        intersection=max(0,min(right,c)-max(left,a))*max(0,min(bottom,d)-max(top,b))
        overlap=max(overlap,intersection/((right-left)*(bottom-top)))
    from .person_geometry import used_keypoints
    detected=sum(x>=0 and y>=0 for x,y in person['keypoints'])
    points=used_keypoints(person)
    border=sum((left==0,top==0,right==image.width,bottom==image.height))
    measured=dict(pixel_digest=pixel_digest,appearance_hash=appearance_hash,width=right-left,height=bottom-top,sharpness=sharpness,
                  keypoints=points,detected_keypoints=detected,overlap=overlap,border_edges=border)
    if packed:
        measured['foreground_hash'] = appearance_hash
    reason=('person_pixels' if min(right-left,bottom-top)<settings['min_person_pixels'] else
            'person_sharpness' if sharpness<settings['min_sharpness'] else
            'person_keypoints' if points<settings['min_keypoints'] else
            'person_overlap' if overlap>settings['max_overlap'] else None)
    return measured,reason


def public_person(person):
    """Keep frame-local encoder pixels out of retained observation metadata."""
    if person is None or 'encoder_mask' not in person.get('pose_refinement', {}):
        return person
    details = {k:v for k,v in person['pose_refinement'].items() if k != 'encoder_mask'}
    return dict(person, pose_refinement=details)


def collect(pipeline,db,model,config,scan,cue,seconds,image,rows,faces, *, expected=None, captured=None):
    expected=expected or contract(config)
    if captured is None:
        people, matches = model.frame_people, model.face_matches
    else:
        people, matches, measurements = captured
    from .sapiensid_backend import association_choices
    choices=association_choices(faces,people)
    settings=config.data['sapiensid']['collection']
    if captured is None and len(people)>1 and config.data['execution']['scan_cpu_workers']>1:
        measurements=list(pipeline.stage.ordered(lambda pixels,person:quality(pixels,person,people,settings),image,people,extra=16384))
    elif captured is None:measurements=[quality(image,person,people,settings) for person in people]
    if people and image is not None:
        from .body_perspectives import frame_signature, orientation
        captured = getattr(image, '_recognition_cue_signature', None)
        signature = captured[2] if captured and captured[0]==cue else frame_signature(image)
        for person, (measurement, _) in zip(people, measurements):
            measurement['frame_signature'] = signature
            measurement['frame_size'] = list(image.size)
            if captured and captured[0]==cue and captured[1] is not None:
                measurement['view_shot'] = captured[1]
            measurement['detected_people'] = len(people)
            measurement['orientation'] = orientation(person)
    # A single pose containing distinct accepted faces is still an unresolved pair.
    distinct_faces = merged_pair = False
    if len(people)==1 and len(faces)>=2:
        left,top,right,bottom = people[0]['bounds']
        heads = [(f.box[0]+f.box[2]/2,f.box[1]+f.box[3]/2,min(f.box[2:]))
                 for f,row in zip(faces,rows) if row['rejected'] is None]
        for i,(x,y,size) in enumerate(heads):
            for u,v,other in heads[i+1:]:
                if np.hypot(x-u,y-v) >= max(4.,.7*max(size,other)):
                    distinct_faces = True
                    merged_pair |= left<=x<right and top<=y<bottom and left<=u<right and top<=v<bottom
    items=[]
    # Keep unavailable face projections for the existing hybrid/report path.
    for row,match in zip(rows,matches):
        if row['rejected'] is None and (match is None or merged_pair):
            items.append(([row['id'],dict(person=None,identity_rejection='person_face_association_unavailable'),None,None],None))
    for index,person in enumerate(people):
        associated=next((i for i,m in enumerate(matches) if m==index),None)
        face=faces[associated] if associated is not None else None
        row=rows[associated] if associated is not None else None
        measured,reason=measurements[index]
        ambiguous=associated is None and any(index in candidates for candidates in choices)
        refinement = person.get('pose_refinement', {})
        separated = refinement.get('status') == 'refined' and refinement.get('separated')
        if separated and (ambiguous or reason == 'person_overlap'):
            measured['query_only_reason'] = 'unassigned_face' if ambiguous else 'separated_overlap'
        if ambiguous and not separated:
            reason = 'person_face_association_ambiguous'
        if reason == 'person_overlap' and separated:
            reason = None
            measured['query_only'] = True
        if ambiguous and separated:
            measured['query_only'] = True
        if person.get('future_torso_retry'):
            measured.update(query_only=True,query_only_reason='future_torso_retry')
        if distinct_faces:
            measured['expected_people'] = 2
        if merged_pair:
            measured.update(query_only=True, query_only_reason='unresolved_partner')
            # Neither face owns a merged body. Keep the independent face observations.
            row = face = None
        if refinement.get('mask_conflict'):
            reason = 'person_mask_conflict'
        face_accepted=row is not None and row['rejected'] is None
        # Rejected face geometry must not invent head landmarks for a body query.
        usable_face=face if face_accepted else None
        metadata=dict(person=public_person(person),quality=measured,identity_rejection=reason,
            face_status='accepted' if face_accepted else 'rejected' if row is not None else 'absent',
            _body=dict(scan=scan,cue=cue,seconds=seconds,contract=expected,
                       face=row['id'] if row is not None else None,
                       face_accepted=face_accepted,rejected=reason))
        record=[identifier(scan,cue,person,expected),metadata,None,None]
        value=model.person_geometry(image,person,usable_face) if reason is None else None
        if value is not None and refinement.get('encoder_mask'):
            value = (*value, refinement['encoder_mask'])
        items.append((record,value))
    marker=[None,dict(_body_cue=(scan,expected,cue)),None,None]
    pipeline.collect_items(db,image,items+[ (marker,None) ])
