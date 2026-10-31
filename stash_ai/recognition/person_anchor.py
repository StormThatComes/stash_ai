"""Source-size admission for body anchors, shared by both SapiensID routes."""
import json
import math

PIXEL_PARAMETER='sapiensid.identities.anchor_min_person_pixels'


def pixels(metadata):
    """Shorter source crop side; never infer resolution from resized images."""
    try:
        record=json.loads(metadata) if isinstance(metadata,str) else metadata
        quality=record.get('quality',{})
        if 'width' in quality or 'height' in quality:
            width,height=quality.get('width'),quality.get('height')
        else:
            # Older face-associated evidence retained bounds before collection
            # started storing explicit width/height quality measurements.
            bounds=(record.get('person') or {}).get('bounds',())
            if len(bounds)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bounds):return None
            width,height=bounds[2]-bounds[0],bounds[3]-bounds[1]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (width,height)):return None
        return min(width,height)
    except (ValueError,TypeError,AttributeError):return None


def pixel_gate(metadata,settings):
    minimum=settings['anchor_min_person_pixels'];measured=pixels(metadata)
    reason=(None if minimum is None else 'body_anchor_pixels_missing' if measured is None else
            'body_anchor_pixels' if measured<minimum else None)
    return dict(pixels=measured,minimum=minimum,eligible=reason is None,reason=reason)


def anchor_gate(metadata,settings,*,collection=None):
    """Measured pose and sharpness must qualify every body reference route."""
    gate=pixel_gate(metadata,settings)
    from .pose_anchor_rules import evaluate
    rules=settings.get('anchor_keypoint_rules', {})
    regional=evaluate(rules,None)
    try:
        record=json.loads(metadata) if isinstance(metadata,str) else metadata
        quality=record.get('quality') or {}
        points=quality.get('keypoints');sharpness=quality.get('sharpness')
        valid=lambda value:type(value) in (int,float) and math.isfinite(value) and value>=0
        person=record.get('person') or {}
        if 'keypoints' in person:
            from .person_geometry import body_points
            anatomical=body_points(person,anatomical=regional['enabled'])
            effective=sum(min(point)>=0 for point in anatomical)
            points=min(points,effective) if valid(points) else effective
            regional=evaluate(rules,anatomical)
        reason=record.get('identity_rejection') or gate['reason']
        if reason is None and collection is not None:
            # Older face projections discarded collection rejection reasons.
            # Re-evaluate measured quality before admitting those retained rows.
            source_pixels=pixels(record);overlap=quality.get('overlap')
            reason=('person_pixels' if source_pixels is not None and source_pixels<collection['min_person_pixels'] else
                    'person_sharpness' if valid(sharpness) and sharpness<collection['min_sharpness'] else
                    'person_keypoints' if valid(points) and points<collection['min_keypoints'] else
                    'person_overlap' if valid(overlap) and overlap>collection['max_overlap'] else None)
        if reason is None:
            reason=('body_anchor_keypoints_missing' if not valid(points) else
                    'body_anchor_keypoints' if points<settings['anchor_min_keypoints'] else
                    regional['reason'] if not regional['eligible'] else
                    'body_anchor_sharpness_missing' if not valid(sharpness) else
                    'body_anchor_sharpness' if sharpness<settings['anchor_min_sharpness'] else None)
    except (ValueError,TypeError,AttributeError,KeyError,OverflowError):
        points=sharpness=None;reason='body_anchor_quality_missing'
    return dict(gate,keypoints=points,sharpness=sharpness,keypoint_rules=regional,
                eligible=reason is None,reason=reason)
