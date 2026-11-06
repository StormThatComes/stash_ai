"""Bounded all-competitor nomination using exact means of unnormalized centroids.

The dot product of two mean vectors equals the mean cross-sample cosine. It
nominates candidates only: original witnesses still determine every merge gate.
No ANN recall assumptions, top-k pruning, or N-by-N score matrix are involved.
"""
import numpy as np
from .disk_body_modes import Matrix,resident_bytes


class Lookup:
    def __init__(self,identities,budget):
        self.budget=budget
        self.working_bytes=0
        self.identities=tuple(identities)
        self.faces=self.persons=None
        if not identities:return
        face_dim=identities[0].faces.shape[1]
        person_dim=identities[0].persons.shape[1]
        # Account for source arrays, their backing/copy buffers and the float64
        # centroid bank before allocating it. Python/SQLite overhead is separate.
        required=2*sum(i.faces.nbytes+i.persons.nbytes for i in identities)
        required+=2*sum(g.nbytes for i in identities for g in (i.guard_faces,i.guard_persons) if g is not None)
        required+=2*sum(resident_bytes(mode) for i in identities for mode in i.body_modes)
        required+=2*sum(f.nbytes+b.nbytes for i in identities for _,f,b in i.query_cores)
        required+=len(identities)*(face_dim+person_dim)*8
        required+=2*64*64*8
        if any(isinstance(m,Matrix) for i in identities for m in i.body_modes):required+=4*1024**2
        if required>budget:
            raise ValueError('SapiensID reference_bytes budget exceeded by lookup working set.')
        from .memory_size import Reservation
        self.reservation=Reservation(budget, label='hybrid-lookup')
        self.reservation.set(required)
        self.working_bytes=required
        self.faces=np.empty((len(identities),face_dim),dtype='float64')
        self.persons=np.empty((len(identities),person_dim),dtype='float64')
        for index,item in enumerate(identities):
            self.faces[index]=item.faces.mean(axis=0,dtype='float64') if len(item.faces) else 0
            self.persons[index]=item.persons.mean(axis=0,dtype='float64') if len(item.persons) else 0
        self.faces.flags.writeable=self.persons.flags.writeable=False

    def pairs(self, *, face_floor=None, person_floor=None, incident=None):
        count=len(self.identities)
        active=None if incident is None else np.array([i.person in incident for i in self.identities])
        if active is not None:
            if not active.any():return
            if active.all():active=None
        from .parallel_numeric import matrix_tiles
        active_blocks=(None if active is None else
                       [start for start in range(0,count,64) if active[start:start+64].any()])
        from bisect import bisect_left
        def tiles():
            for start in range(0,count,64):
                rows=np.arange(start,min(start+64,count))
                others=(range(start,count,64) if active_blocks is None or active[start:start+64].any()
                        else active_blocks[bisect_left(active_blocks,start):])
                for other in others:
                    cols=np.arange(other,min(other+64,count))
                    if active is None or active[rows].all() or active[cols].all():
                        yield (rows,cols), ((self.faces[start:start+64],self.faces[other:other+64]),
                                            (self.persons[start:start+64],self.persons[other:other+64]))
                        continue
                    # Contiguous runs preserve the dense kernel's output order.
                    # Inactive rows need only the active columns in this tile.
                    flags=active[rows]
                    for run in np.split(rows,np.flatnonzero(flags[1:]!=flags[:-1])+1):
                        columns=cols if active[run[0]] else cols[active[cols]]
                        columns=columns[columns>run[0]]
                        if not len(columns):continue
                        run=run[run<columns[-1]]
                        yield (run,columns), ((self.faces[run],self.faces[columns]),
                                              (self.persons[run],self.persons[columns]))
        has_face=np.array([bool(len(i.faces)) for i in self.identities])
        has_body=np.array([bool(len(i.persons)) for i in self.identities])
        for (rows,columns),(faces,persons) in matrix_tiles(tiles(),kind='hybrid_nomination'):
            mask=rows[:,None]<columns[None,:]
            if face_floor is not None or person_floor is not None:
                nominated=np.zeros_like(mask)
                if face_floor is not None:nominated |= (faces+1e-6>=face_floor) & has_face[rows,None] & has_face[None,columns]
                if person_floor is not None:nominated |= (persons+1e-6>=person_floor) & has_body[rows,None] & has_body[None,columns]
                mask &= nominated
            for i,j in zip(*np.nonzero(mask)):
                left,right=self.identities[rows[i]],self.identities[columns[j]]
                yield (left.person,right.person,
                    float(faces[i,j]) if len(left.faces) and len(right.faces) else None,
                    float(persons[i,j]) if len(left.persons) and len(right.persons) else None)
