"""Prepare all detected people once, optionally sharing their evidence with a face."""
import hashlib
import numpy as np

from .body_observations import contract, identifier


def _pixels(image,bounds):
    left,top,right,bottom=bounds
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb, rgb.convert('L') as gray:
        values=np.asarray(gray,dtype=np.float32)
        lap=(4*values[1:-1,1:-1]-values[:-2,1:-1]-values[2:,1:-1]
             -values[1:-1,:-2]-values[1:-1,2:])
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


def collect(pipeline,db,model,config,scan,cue,seconds,image,rows,faces):
    expected=contract(config)
    people=model.frame_people
    matches=model.face_matches
    from .sapiensid_backend import association_choices
    choices=association_choices(faces,people)
    settings=config.data['sapiensid']['collection']
    if len(people)>1 and config.data['execution']['scan_cpu_workers']>1:
        measurements=list(pipeline.stage.ordered(lambda pixels,person:quality(pixels,person,people,settings),image,people,extra=16384))
    else:measurements=[quality(image,person,people,settings) for person in people]
    if people and image is not None:
        from .body_perspectives import frame_signature, orientation
        captured = getattr(image, '_recognition_cue_signature', None)
        signature = captured[2] if captured and captured[0]==cue else frame_signature(image)
        for person, (measurement, _) in zip(people, measurements):
            measurement['frame_signature'] = signature
            if captured and captured[0]==cue and captured[1] is not None:
                measurement['view_shot'] = captured[1]
            measurement['detected_people'] = len(people)
            measurement['orientation'] = orientation(person)
    items=[]
    # Keep unavailable face projections for the existing hybrid/report path.
    for row,match in zip(rows,matches):
        if row['rejected'] is None and match is None:
            items.append(([row['id'],dict(person=None,identity_rejection='person_face_association_unavailable'),None,None],None))
    for index,person in enumerate(people):
        associated=next((i for i,m in enumerate(matches) if m==index),None)
        face=faces[associated] if associated is not None else None
        row=rows[associated] if associated is not None else None
        measured,reason=measurements[index]
        ambiguous=associated is None and any(index in candidates for candidates in choices)
        refinement = person.get('pose_refinement', {})
        separated = refinement.get('status') == 'refined' and refinement.get('separated')
        if ambiguous and not separated:
            reason = 'person_face_association_ambiguous'
        if reason == 'person_overlap' and separated:
            reason = None
            measured['query_only'] = True
        if ambiguous and separated:
            measured['query_only'] = True
        if refinement.get('mask_conflict'):
            reason = 'person_mask_conflict'
        if refinement.get('encoder_mask'):
            from .person_geometry import encoder_crop
            with encoder_crop(image, person['bounds'], refinement['encoder_mask']) as foreground:
                _, measured['pixel_digest'], measured['foreground_hash'] = _pixels(
                    foreground, (0, 0, foreground.width, foreground.height))
            measured['appearance_hash'] = measured['foreground_hash']
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
