"""Bounded reads and versioned, per-scene endpoint representative selections."""
import hashlib
import json
from itertools import islice

from .body_identity import vector
from .database import encoded
from .dependencies import supported, version
from .derived_cache import current


def representatives_for_view(db, group, owner, scan, kind, stamp, settings, face, scores):
    # Keep small addresses for this scan only. Never sort embedding blobs.
    candidates = [(oid,raw) for _,oid,raw,_,_ in group]
    cache = current()
    policy = 'body-appearance-endpoint-views-v2'
    if cache is not None and supported(db):
        signature = hashlib.sha256(encoded([owner,scan,kind,stamp,version(db,'scan',scan),
                                            version(db,'raw-body',scan),candidates]).encode()).hexdigest()
        saved = cache.get(policy,signature)
        if saved is not None:
            return saved
    else:
        cache = None
    result, first, novel = [], None, None
    distance = 0.
    alternatives = []
    items = iter(candidates)
    while batch := list(islice(items,32)):
        blobs = {oid:(blob,cue,raw_q) for oid,blob,cue,raw_q in db.execute("SELECT id,embedding,cue,quality FROM body_observations WHERE id IN "
            "(SELECT value FROM json_each(?)) AND embedding IS NOT NULL",(encoded([oid for oid,_ in batch]),))}
        for oid,raw in batch:
            roots = json.loads(raw)
            if oid not in blobs or len(roots)!=2 or roots[0]==roots[1] or not all(face(scan,r,owner) for r in roots):
                continue
            blob,cue,raw_q = blobs[oid]
            pixel=json.loads(raw_q).get('quality',{}).get('pixel_digest') or oid
            value = vector({'embedding':blob})
            if first is None:
                result.append((0,oid,roots))
                first = value
            else:
                delta = 1.-scores.face_similarity(first,value)
                if delta<=1.-settings['match_threshold']:
                    continue
                for example,ref,refcue,refpixel,spread in alternatives:
                    if (refcue!=cue and refpixel!=pixel and spread>distance and
                            scores.face_similarity(value,example)>=settings['match_threshold']):
                        novel,distance = ref,spread
                if len(alternatives)<settings.get('max_references',32):
                    alternatives.append((value,(1,oid,roots),cue,pixel,delta))
    if novel and distance>1.-settings['match_threshold']:
        result.append(novel)
    if cache is not None:
        cache.put(policy,signature,result)
    return result
