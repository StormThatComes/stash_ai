"""Content-defined blocks keep unchanged nominations reusable after insertion."""
from hashlib import sha256
import pickle
import numpy as np

from .config import digest
from .disk_body_modes import Matrix
from .parallel_numeric import map_bounded


def endpoint_version(identity):
    # Disk-backed modes are immutable, but their temporary paths change each run.
    # Hash the exact ordered samples and vote weights, never the filename.
    class Fingerprinter(pickle.Pickler):
        def persistent_id(self, value):
            if not isinstance(value, Matrix):
                return None
            h=sha256()
            for start in range(0,len(value),32):
                block=slice(start,start+32)
                h.update(value[block].tobytes())
                h.update(value.weights(block).tobytes())
            return ("mode",value.shape,h.digest())
    stream=sha256()
    class Writer:
        def write(self, value):
            stream.update(value)
            return memoryview(value).nbytes
    Fingerprinter(Writer(),protocol=5).dump(identity)
    return stream.hexdigest()


def _blocks(lookup):
    start=0
    for i,identity in enumerate(lookup.identities,1):
        # After a new ID splits one block, subsequent boundary IDs still align.
        if (i-start>=64 or i==len(lookup.identities) or
            sha256(identity.person.encode()).digest()[0]<8):
            faces=lookup.faces[start:i]
            bodies=lookup.persons[start:i]
            bank=lookup.identities[start:i]
            labels=[(v.person,bool(len(v.faces)),bool(len(v.persons))) for v in bank]
            version=digest([labels,sha256(faces).hexdigest(),sha256(bodies).hexdigest()])
            yield version,faces,bodies,labels
            start=i


def _nominations(payload):
    left,right,diagonal,face_floor,body_floor=payload
    _,faces,bodies,labels=left
    _,other_faces,other_bodies,others=right
    return _selected(payload,faces@other_faces.T,bodies@other_bodies.T)


def _selected(payload,face,body):
    left,right,diagonal,face_floor,body_floor=payload
    labels,others=left[3],right[3]
    mask=(face+1e-6>=face_floor)|(body+1e-6>=body_floor)
    if diagonal:mask=np.triu(mask,1)
    return [(labels[i][0],others[j][0],float(face[i,j]) if labels[i][1] and others[j][1] else None,
             float(body[i,j]) if labels[i][2] and others[j][2] else None) for i,j in zip(*np.nonzero(mask))]


def nominations(lookup, store, face_floor, body_floor, config=None):
    from .nomination_bounds import ball, upper
    from .nomination_cuda import select
    blocks=list(_blocks(lookup))
    if not blocks:return
    dimensions=lookup.faces.shape[1]+lookup.persons.shape[1]
    required=(len(blocks)+64)*dimensions*8+len(lookup.identities)*512
    if required>lookup.budget-lookup.working_bytes:
        yield from lookup.pairs(face_floor=face_floor,person_floor=body_floor)
        return
    from .memory_size import Reservation
    from .resource_errors import ResourceLimit
    bound_budget=Reservation(lookup.budget-lookup.working_bytes,label='hybrid-block-bounds')
    try:
        bound_budget.set(required)
    except ResourceLimit:
        yield from lookup.pairs(face_floor=face_floor,person_floor=body_floor)
        return
    bounds=[(ball(block[1]),ball(block[2])) for block in blocks]
    device=select(config,lookup,blocks)
    def compute(payload):
        if device is not None:
            left,right,_,ff,bf=payload
            measured=device.measured_tile(left,right,ff,bf)
            if measured is not None:return _selected(payload,*measured)
        return _nominations(payload)
    # A completed catalogue certifies empty comparisons as well as hits. Only
    # positive old block pairs need individual tile records. If one was evicted
    # or damaged it is recomputed before this run can certify a successor.
    catalogue_key=digest(['hybrid-block-catalogue-v1',face_floor,body_floor])
    previous=store.get('hybrid-catalogue',catalogue_key) if store else None
    positions={block[0]:i for i,block in enumerate(blocks)}
    previous_ids=set(previous[0]) if previous else set()
    retained={}
    if previous:
        for left,right in previous[1]:
            if left in positions and right in positions:
                i,j=sorted((positions[left],positions[right]))
                retained.setdefault(i,[]).append(j)
    changed=[i for i,block in enumerate(blocks) if block[0] not in previous_ids]
    changed_set=set(changed)
    positive=[]
    # Keep both the temporary list and the serialized record bounded.
    catalogue_limit=min(store.budget//8,4*1024**2,lookup.budget-lookup.working_bytes) if store else 0
    catalogue_ok=len(blocks)*256<catalogue_limit
    def items():
        from bisect import bisect_left
        for i,first in enumerate(blocks):
            if i in changed_set:
                neighbours=range(i,len(blocks))
            else:
                neighbours=sorted((*retained.get(i,()),*changed[bisect_left(changed,i):]))
            for j in neighbours:
                count('hybrid_block_pairs_visited')
                second=blocks[j]
                key=digest([first[0],second[0],i==j,face_floor,body_floor])
                payload=(first,second,i==j,face_floor,body_floor)
                if (upper(bounds[i][0],bounds[j][0])+1e-6<face_floor and
                    upper(bounds[i][1],bounds[j][1])+1e-6<body_floor):
                    count('hybrid_blocks_excluded')
                    continue
                yield key,payload,store.get("hybrid-nominations",key) if store else None
    from .profiling import count
    for (key,payload,hit),result in map_bounded(compute,items(),
            payload=lambda item:item[1],cost=lambda item:65536+512*len(item[1][0][3])*len(item[1][1][3]),
            cached=lambda item:item[2],kind="incremental_hybrid_tiles"):
        if hit is None:
            count('hybrid_centroid_pairs_recomputed',len(payload[0][3])*len(payload[1][3]))
            if store:store.put("hybrid-nominations",key,result)
        if result and catalogue_ok:
            if (len(positive)+len(blocks)+1)*256>catalogue_limit:
                catalogue_ok=False
                positive.clear()
            else:positive.append([payload[0][0],payload[1][0]])
        yield from result
    if catalogue_ok:
        store.put('hybrid-catalogue',catalogue_key,[[block[0] for block in blocks],positive])
