"""Prepare all detected people once, optionally sharing their evidence with a face."""
import hashlib
import numpy as np

from .body_observations import contract, identifier


def _pixels(image,bounds,mask=None, *, measure=True, encoder_mask=None):
    left,top,right,bottom=bounds
    with image.crop(bounds) as crop, crop.convert('RGB') as rgb, rgb.convert('L') as gray:
        values=np.asarray(gray,dtype=np.float32)
        from .blur_quality import measure as measure_sharpness
        measured=None
        if measure:
            from .model_input_quality import body
            measured=body(rgb,mask,encoder_mask)
            measured['portrait_sharpness']=measure_sharpness(values,mask)['sharpness']
        pixel_digest=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
        with gray.resize((9,8)) as thumb:
            small=np.asarray(thumb)
            bits=(small[:,1:]>small[:,:-1]).flatten()
            appearance_hash=format(sum(int(v)<<i for i,v in enumerate(bits)),'016x')
    return measured,pixel_digest,appearance_hash


def quality(image,person,people,settings):
    from .inference_cache import many
    bounds=person['bounds'];left,top,right,bottom=bounds
    from .mask_usage import uses
    packed = person.get('pose_refinement',{}).get('encoder_mask') if uses(person, 'body_encoder_input') else None
    from .body_overlap import foreground_mask
    foreground = foreground_mask(person)
    masked_sharpness = uses(person, 'sharpness')
    mask = foreground[0] if foreground is not None and masked_sharpness else None
    # Match encoder pixels while excluding synthetic replacement outlines;
    # the sharpness switch independently selects the measured region.
    if mask is not None or packed:
        from .person_geometry import unpack_mask
        encoder_mask = unpack_mask(packed,right-left,bottom-top) if packed else None
        sharp,pixel_digest,appearance_hash = _pixels(image,bounds,mask,encoder_mask=encoder_mask)
    else:
        sharp,pixel_digest,appearance_hash=many('body_quality','body-pixels-v5-split-blur',[(image,bounds)],
            lambda values:[_pixels(pixels,box) for pixels,box in values])[0]
    source_pixel_digest = pixel_digest
    if packed:
        from .person_geometry import encoder_crop
        with encoder_crop(image,bounds,packed) as crop:
            _,pixel_digest,appearance_hash = _pixels(crop,(0,0,right-left,bottom-top),measure=False)
    if masked_sharpness and mask is None:
        sharp=dict(sharp,crop_sharpness=sharp['sharpness'],sharpness=0.,sharpness_normalized=0.,
                   portrait_sharpness=0.,sharpness_pixels=0, focus_sharpness=None, motion_sharpness=None,
                   sharpness_confidence=0., sharpness_split_status='unavailable')
    sharpness=sharp['sharpness']
    sharp['sharpness_method']=('sam_mask' if mask is not None else 'unavailable') if masked_sharpness else 'person_crop'
    from .body_overlap import admission_overlap, measure
    overlaps=measure(person,people,report_above=settings['max_overlap'])
    overlap=admission_overlap(overlaps)
    from .person_geometry import used_keypoints
    detected=sum(x>=0 and y>=0 for x,y in person['keypoints'])
    points=used_keypoints(person)
    border=sum((left==0,top==0,right==image.width,bottom==image.height))
    details = person.get('pose_refinement', {})
    if 'mask_applied' in details:
        details['mask_applied'].update(sharpness=mask is not None,
            body_encoder_input=bool(packed), overlap=uses(person, 'overlap') and foreground is not None)
    measured=dict(pixel_digest=pixel_digest,appearance_hash=appearance_hash,width=right-left,height=bottom-top,
                  keypoints=points,detected_keypoints=detected,border_edges=border,**sharp,**overlaps)
    if packed:
        measured['foreground_hash'] = appearance_hash
        # Matching uses the masked input digest. VTT inspection verifies the original pixels.
        measured['source_pixel_digest'] = source_pixel_digest
    from .sharpness_gates import failure, active
    blur_failure = failure(sharp, settings)
    reason=('person_pixels' if min(right-left,bottom-top)<settings['min_person_pixels'] else
            'person_sharpness_mask_unavailable' if active(settings) and not sharp['sharpness_pixels'] else
            'person_'+blur_failure if blur_failure else
            'person_keypoints' if points<settings['min_keypoints'] else
            'person_overlap' if overlap is not None and overlap>settings['max_overlap'] else None)
    return measured,reason


def foreground_mask_available(person):
    from .body_overlap import foreground_mask
    return foreground_mask(person) is not None


def public_person(person):
    """Keep frame-local quality/encoder masks out of observation metadata."""
    masks = {'encoder_mask', 'overlap_mask'}
    if person is None or not masks.intersection(person.get('pose_refinement', {})):
        return person
    details = {k:v for k,v in person['pose_refinement'].items() if k not in masks}
    return dict(person, pose_refinement=details)


def isolated_torso(person, people):
    from .track_geometry import torso_points
    points = torso_points(person)
    if len(people)<2 or len(points)<3:
        return False
    def inside(point, box):
        x,y = point
        return box[0]<=x<=box[2] and box[1]<=y<=box[3]
    if not all(inside(point, person['bounds']) for point in points.values()):
        return False
    for peer in people:
        if peer is person:
            continue
        foreign = torso_points(peer)
        if (len(foreign)<3 or any(inside(p, peer['bounds']) for p in points.values())
                or any(inside(p, person['bounds']) for p in foreign.values())):
            return False
    return True


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
                measurement['transition_verified'] = bool(len(captured)>4 and captured[4])
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
    contact_masks = {}
    # Keep unavailable face projections for the existing hybrid/report path.
    for row,match in zip(rows,matches):
        if row['rejected'] is None and (match is None or merged_pair):
            items.append(([row['id'],dict(person=None,identity_rejection='person_face_association_unavailable'),None,None],None))
    for index,person in enumerate(people):
        associated=next((i for i,m in enumerate(matches) if m==index),None)
        face=faces[associated] if associated is not None else None
        row=rows[associated] if associated is not None else None
        measured,reason=measurements[index]
        if 'overlap_peer_index' in measured:
            peer = people[measured['overlap_peer_index']]
            measured['overlap_peer_observation'] = identifier(scan, cue, peer, expected)
        ambiguous=associated is None and any(index in candidates for candidates in choices)
        refinement = person.get('pose_refinement', {})
        measured['embedding_foreground']='masked' if refinement.get('encoder_mask') else 'full'
        from .mask_usage import uses, admission_conflict
        if not refinement.get('encoder_mask'):
            measured['embedding_mask_reason'] = ('disabled' if not uses(person, 'body_encoder_input')
                else 'mask_not_separated' if foreground_mask_available(person) else 'mask_unavailable')
        separated = uses(person, 'body_admission') and refinement.get('separated') and (
            refinement.get('mask_valid') or refinement.get('status') in ('refined','masked'))
        if separated and (ambiguous or reason == 'person_overlap'):
            measured['query_only_reason'] = 'unassigned_face' if ambiguous else 'separated_overlap'
        if ambiguous and not separated:
            if reason is None and not merged_pair and isolated_torso(person,people):
                measured.update(query_only=True,query_only_reason='independent_torso_unassigned_face')
            else:
                reason = reason or 'person_face_association_ambiguous'
        if reason == 'person_overlap' and separated:
            reason = None
            measured['query_only'] = True
        if ambiguous and separated:
            measured['query_only'] = True
        if refinement.get('contact_pose_recovered'):
            measured.update(query_only=True,query_only_reason='contact_pose_recovered')
        if person.get('future_torso_retry'):
            measured.update(query_only=True,query_only_reason='future_torso_retry')
        if distinct_faces:
            measured['expected_people'] = 2
        if merged_pair:
            measured.update(query_only=True, query_only_reason='unresolved_partner')
            # Neither face owns a merged body. Keep the independent face observations.
            row = face = None
        if admission_conflict(person):
            reason = 'person_mask_conflict'
        if 'mask_applied' in refinement:
            refinement['mask_applied']['body_admission'] = bool(separated or admission_conflict(person))
        face_accepted=row is not None and row['rejected'] is None
        if face_accepted and len(people) > 1:
            from .contact_face_evidence import measure as contact_evidence
            measured['contact_face_evidence'] = contact_evidence(face, person, people,
                face_id=row['id'], faces=faces, mask_cache=contact_masks)
            proof = measured['contact_face_evidence']
            for peer in [*proof['peers'], *proof.get('upper_body', {}).get('peers', ())]:
                peer['observation'] = identifier(scan, cue, people[peer['person_index']], expected)
        # Rejected face geometry must not invent head landmarks for a body query.
        usable_face=face if face_accepted else None
        metadata=dict(person=public_person(person),quality=measured,identity_rejection=reason,
            face_status='accepted' if face_accepted else 'rejected' if row is not None else 'absent',
            _body=dict(scan=scan,cue=cue,seconds=seconds,contract=expected,
                       face=row['id'] if row is not None else None,
                       face_accepted=face_accepted,rejected=reason))
        if foreground_mask_available(person) and config.data['sapiensid']['storage']['retain_body_masks']:
            from .retained_body_masks import compress_mask
            metadata['_quality_mask'] = compress_mask(refinement['overlap_mask'])
        record=[identifier(scan,cue,person,expected),metadata,None,None]
        value=model.person_geometry(image,person,usable_face) if reason is None else None
        if value is not None and refinement.get('encoder_mask'):
            value = (*value, refinement['encoder_mask'])
        items.append((record,value))
    marker=[None,dict(_body_cue=(scan,expected,cue)),None,None]
    pipeline.collect_items(db,image,items+[ (marker,None) ])
