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


def anchor_gate(metadata,settings):
    """Measured pose and sharpness must qualify every body reference route."""
    gate=pixel_gate(metadata,settings)
    try:
        record=json.loads(metadata) if isinstance(metadata,str) else metadata
        quality=record.get('quality') or {}
        points=quality.get('keypoints');sharpness=quality.get('sharpness')
        valid=lambda value:type(value) in (int,float) and math.isfinite(value) and value>=0
        reason=gate['reason']
        if reason is None:
            reason=('body_anchor_keypoints_missing' if not valid(points) else
                    'body_anchor_keypoints' if points<settings['anchor_min_keypoints'] else
                    'body_anchor_sharpness_missing' if not valid(sharpness) else
                    'body_anchor_sharpness' if sharpness<settings['anchor_min_sharpness'] else None)
    except (ValueError,TypeError,AttributeError):
        points=sharpness=None;reason='body_anchor_quality_missing'
    return dict(gate,keypoints=points,sharpness=sharpness,eligible=reason is None,reason=reason)
