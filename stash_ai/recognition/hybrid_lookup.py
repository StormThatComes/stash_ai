"""Bounded all-competitor nomination using exact means of unnormalized centroids.

The dot product of two mean vectors equals the mean cross-sample cosine. It
nominates candidates only: original witnesses still determine every merge gate.
No ANN recall assumptions, top-k pruning, or N-by-N score matrix are involved.
"""
import numpy as np


class Lookup:
    def __init__(self,identities,budget):
        self.identities=tuple(identities)
        self.faces=self.persons=None
        if not identities:return
        face_dim=identities[0].faces.shape[1]
        person_dim=identities[0].persons.shape[1]
        # Account for source arrays, their backing/copy buffers and the float64
        # centroid bank before allocating it. Python/SQLite overhead is separate.
        required=2*sum(i.faces.nbytes+i.persons.nbytes for i in identities)
        required+=2*sum(g.nbytes for i in identities for g in (i.guard_faces,i.guard_persons) if g is not None)
        required+=2*sum(mode.nbytes for i in identities for mode in i.body_modes)
        required+=2*sum(f.nbytes+b.nbytes for i in identities for _,f,b in i.query_cores)
        required+=len(identities)*(face_dim+person_dim)*8
        required+=2*64*64*8
        if required>budget:
            raise ValueError('SapiensID reference_bytes budget exceeded by lookup working set.')
        from .memory_size import Reservation
        self.reservation=Reservation(budget)
        self.reservation.set(required)
        self.faces=np.empty((len(identities),face_dim),dtype='float64')
        self.persons=np.empty((len(identities),person_dim),dtype='float64')
        for index,item in enumerate(identities):
            self.faces[index]=item.faces.mean(axis=0,dtype='float64') if len(item.faces) else 0
            self.persons[index]=item.persons.mean(axis=0,dtype='float64') if len(item.persons) else 0

    def pairs(self, *, face_floor=None, person_floor=None, incident=None):
        count=len(self.identities)
        active=None if incident is None else np.array([i.person in incident for i in self.identities])
        for start in range(0,count,64):
            for other in range(start,count,64):
                if active is not None and not (active[start:start+64].any() or active[other:other+64].any()):continue
                faces=self.faces[start:start+64] @ self.faces[other:other+64].T
                persons=self.persons[start:start+64] @ self.persons[other:other+64].T
                rows=np.arange(start,start+len(faces))[:,None]
                columns=np.arange(other,other+faces.shape[1])[None,:]
                mask=rows<columns
                if active is not None:mask &= active[rows] | active[columns]
                if face_floor is not None or person_floor is not None:
                    nominated=np.zeros_like(mask)
                    if face_floor is not None:nominated |= faces+1e-6>=face_floor
                    if person_floor is not None:nominated |= persons+1e-6>=person_floor
                    mask &= nominated
                for i,j in zip(*np.nonzero(mask)):
                    left,right=self.identities[start+i],self.identities[other+j]
                    yield (left.person,right.person,
                        float(faces[i,j]) if len(left.faces) and len(right.faces) else None,
                        float(persons[i,j]) if len(left.persons) and len(right.persons) else None)
