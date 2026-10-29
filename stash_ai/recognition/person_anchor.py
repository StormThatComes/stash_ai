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
