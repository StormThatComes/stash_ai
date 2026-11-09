"""Conservative fragment nomination inside a scene selected for rear replay."""
import base64
from collections import defaultdict

import numpy as np

from .body_identity import vector
from .config import digest
from .dependencies import version
from .derived_cache import current
from .profiling import count


WIDTH = 16


def _bounds(db, scan, stamp, budget):
    fragments = {}
    retained = 0
    rows = db.execute("""SELECT b.cue,b.embedding,m.person,json_extract(m.proof,'$.rooted_track.owner') AS root
        FROM body_observations b JOIN body_members m ON m.observation=b.id
        WHERE b.scan=? AND m.policy=? AND b.rejected IS NULL ORDER BY b.cue,b.id""",(scan,stamp))
    for row in rows:
        block = row['cue']//WIDTH
        balls,owners,roots = fragments.setdefault(block,([],set(),set()))
        for owner,group in ((row['person'],owners),(row['root'],roots)):
            if owner and owner not in group:
                group.add(owner)
                retained += 256+len(owner)*4
        if row['embedding'] is not None:
            q = vector(row).astype('float64')
            distances = [float(np.linalg.norm(q-center)) for center,_ in balls]
            if len(balls)<4 and (not distances or min(distances)>.45):
                balls.append([q,0.])
                retained += q.nbytes*3
            else:
                i = int(np.argmin(distances))
                balls[i][1] = max(balls[i][1],distances[i])
        if retained>budget:
            return None
    return [[block,sorted(owners),sorted(roots),
             [[base64.b64encode(q.astype('<f4').tobytes()).decode(),r] for q,r in balls]]
            for block,(balls,owners,roots) in fragments.items()]


def select(db,scan,stamp,pending):
    """Return None for full replay. Keep every cue of a potentially moving old identity."""
    store = current()
    if store is None or not pending.local_clocks:
        return None
    maximum = db.execute('SELECT max(cue) FROM body_observations WHERE scan=?',(scan,)).fetchone()[0]
    if maximum is None or maximum<WIDTH:
        return None
    scene_key = db.execute('SELECT json_array(scope,namespace,scene) FROM scans WHERE id=?',(scan,)).fetchone()[0]
    key = digest(['rear-fragments-v1',scan,stamp,version(db,'raw-body',scan),
                  version(db,'scene',scene_key),version(db,'raw-observations',scan)])
    kind = 'rear-fragment-bounds'
    fragments = store.get(kind,key)
    if fragments is None:
        budget = pending.search.config.data['sapiensid']['execution']['reference_bytes']//8
        fragments = _bounds(db,scan,stamp,budget)
        if fragments is None:
            return None
        store.put(kind,key,fragments)
    selected = set()
    owner_blocks = defaultdict(list)
    owners_by_block = {}
    endpoints = getattr(pending.search,'endpoints',None)
    for block,owners,roots,balls in fragments:
        owners_by_block[block] = owners
        for owner in owners:
            owner_blocks[owner].append(block)
        if endpoints and any(p in endpoints.banks for p in roots):
            selected.add(block)
        elif any(pending.gallery.nearby(np.frombuffer(base64.b64decode(raw),dtype='<f4'),
                                        pending.radius+radius+.003) for raw,radius in balls):
            selected.add(block)
    # Co-present competitors are in one block. Atomic moves and splits also
    # depend on all other blocks occupied by the same original identity.
    queue = list(selected)
    visited = set()
    while queue:
        for owner in owners_by_block[queue.pop()]:
            if owner in visited:
                continue
            visited.add(owner)
            for peer in owner_blocks[owner]:
                if peer not in selected:
                    selected.add(peer)
                    queue.append(peer)
    count('rear_fragments_examined',len(fragments))
    count('rear_fragments_skipped',len(fragments)-len(selected))
    return None if len(selected)==len(fragments) else sorted(selected)


def rank(pending, requests):
    """Memoize gallery comparisons only. Physical occupancy and global vetoes are replayed."""
    import hashlib
    gallery = pending.gallery
    def exact():
        return gallery.rank_many(requests) if hasattr(gallery,'rank_many') else [gallery.rank(*r) for r in requests]
    store = current()
    if not requests or store is None or not pending.local_clocks:
        return exact()
    center = np.mean([req[0] for req in requests],axis=0,dtype=np.float64)
    radius = max(gallery.search_radius(settings)+np.linalg.norm(vec-center)
                 for vec,_,settings,_,_ in requests)
    nearby = sorted(gallery.nearby(center,float(radius)+.003))
    if not nearby:
        return [[] for _ in requests]
    queries = [(hashlib.sha256(vec.tobytes()).hexdigest(),scan,settings['match_threshold'],
                settings['ambiguity_margin'],settings['attach_min_support'],pixel,kind)
               for vec,scan,settings,pixel,kind in requests]
    key = digest(['rear-fragment-rank-v1',queries,[pending.banks[i] for i in nearby]])
    cached = store.get('rear-fragment-ranks',key)
    if cached is not None:
        count('rear_fragment_queries_reused',len(requests))
        return cached
    result = exact()
    store.put('rear-fragment-ranks',key,result)
    return result
