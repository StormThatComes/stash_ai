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
    policy = 'body-appearance-endpoint-views-v1'
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
    items = iter(candidates)
    while batch := list(islice(items,32)):
        blobs = dict(db.execute("SELECT id,embedding FROM body_observations WHERE id IN "
            "(SELECT value FROM json_each(?)) AND embedding IS NOT NULL",(encoded([oid for oid,_ in batch]),)))
        for oid,raw in batch:
            roots = json.loads(raw)
            if oid not in blobs or len(roots)!=2 or roots[0]==roots[1] or not all(face(scan,r,owner) for r in roots):
                continue
            value = vector({'embedding':blobs[oid]})
            if first is None:
                result.append((0,oid,roots))
                first = value
            else:
                delta = 1.-scores.face_similarity(first,value)
                if delta>distance:
                    novel,distance = (1,oid,roots),delta
    if novel and distance>1.-settings['match_threshold']:
        result.append(novel)
    if cache is not None:
        cache.put(policy,signature,result)
    return result
