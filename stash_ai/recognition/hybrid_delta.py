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
            return len(value)
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
    face=faces@other_faces.T
    body=bodies@other_bodies.T
    mask=(face+1e-6>=face_floor)|(body+1e-6>=body_floor)
    if diagonal:mask=np.triu(mask,1)
    return [(labels[i][0],others[j][0],float(face[i,j]) if labels[i][1] and others[j][1] else None,
             float(body[i,j]) if labels[i][2] and others[j][2] else None) for i,j in zip(*np.nonzero(mask))]


def nominations(lookup, store, face_floor, body_floor):
    blocks=list(_blocks(lookup))
    def items():
        for i,first in enumerate(blocks):
            for j in range(i,len(blocks)):
                second=blocks[j]
                key=digest([first[0],second[0],i==j,face_floor,body_floor])
                payload=(first,second,i==j,face_floor,body_floor)
                yield key,payload,store.get("hybrid-nominations",key)
    from .profiling import count
    for (key,payload,hit),result in map_bounded(_nominations,items(),
            payload=lambda item:item[1],cost=lambda item:65536+512*len(item[1][0][3])*len(item[1][1][3]),
            cached=lambda item:item[2],kind="incremental_hybrid_tiles"):
        if hit is None:
            count('hybrid_centroid_pairs_recomputed',len(payload[0][3])*len(payload[1][3]))
            store.put("hybrid-nominations",key,result)
        yield from result

