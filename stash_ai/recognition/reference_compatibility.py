"""Reapply current quality gates without equating a scan cache key with a model.

Same-policy scans keep their existing admission. Cross-policy evidence must carry
current pose/alignment contracts and all quality measurements required today.
No image is inferred again and no rejected observation is resurrected.
"""
import json
import math
from types import SimpleNamespace

from .config import digest
from .models import checked_detection
from .quality import rejection
from .scanning import geometry_contract
from .visibility import cached as visibility_cached


def fallback_sql(db,config,namespace,*,observation='o'):
    policy=config.data['detection']
    name='reference_quality_'+digest([namespace,policy,'retained-quality-v1'])[:24]
    if not any(row[0]==name for row in db.execute('PRAGMA function_list')):
        def admissible(raw,geometry,rect):
            try:
                quality=json.loads(raw)
                face=checked_detection(json.loads(geometry))
                rectangle=json.loads(rect)
                if (quality.get('pose_contract')!='eye-frame-v2' or
                    quality.get('alignment_valid') is not True or
                    quality.get('geometry_contract')!=geometry_contract(face,SimpleNamespace(rect=rectangle)) or
                    face.score<=policy['score_threshold']):
                    return 0
                for key in ('pixels','sharpness','pose_deviation'):
                    value=quality.get(key)
                    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                        return 0
                if any(type(quality.get(key)) is not bool for key in ('complete','landmarks_inside')):
                    return 0
                if rejection(quality,policy) is not None:
                    return 0
                if policy['min_visible_face_fraction']>0:
                    return int(visibility_cached(quality) and
                               quality['visible_face_fraction']>=policy['min_visible_face_fraction'])
                return 1
            except (ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError):
                return 0
        db.create_function(name,3,admissible,deterministic=True)
    return f'{name}({observation}.quality,{observation}.geometry,{observation}.rect)'
