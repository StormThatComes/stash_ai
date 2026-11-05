"""Optional measured CUDA nomination. Exact witness and rival scoring stays on CPU."""
from time import perf_counter
import numpy as np

from .profiling import count


class Resident:
    def __init__(self,torch,lookup,blocks):
        self.torch=torch
        self.active=True
        start=perf_counter()
        # torch.tensor owns a copy and does not alias the read-only NumPy bank.
        self.faces=torch.tensor(lookup.faces,device='cuda',dtype=torch.float64)
        self.bodies=torch.tensor(lookup.persons,device='cuda',dtype=torch.float64)
        torch.cuda.synchronize()
        self.upload=perf_counter()-start
        offset=0
        self.slices={}
        for key,faces,_,_ in blocks:
            self.slices[key]=slice(offset,offset+len(faces))
            offset+=len(faces)

    def scores(self,left,right):
        a,b=self.slices[left[0]],self.slices[right[0]]
        return ((self.faces[a]@self.faces[b].T).cpu().numpy(),
                (self.bodies[a]@self.bodies[b].T).cpu().numpy())

    def competitive(self,blocks,tiles):
        block=max(blocks,key=lambda item:len(item[1]))
        def elapsed(operation):
            samples=[]
            for _ in range(3):
                start=perf_counter()
                operation()
                samples.append(perf_counter()-start)
            return sorted(samples)[1]
        cpu=elapsed(lambda:(block[1]@block[1].T,block[2]@block[2].T))
        gpu=elapsed(lambda:self.scores(block,block))
        return gpu+self.upload/tiles < cpu*.9

    def measured_tile(self,left,right,face_floor,body_floor):
        if not self.active:return None
        try:
            faces,bodies=self.scores(left,right)
        except (RuntimeError,MemoryError):
            self.active=False
            count('nomination_cuda_fallback')
            return None
        # A device score near either nomination boundary is resolved with the
        # original NumPy kernel, including the original overselection tolerance.
        if (not np.isfinite(faces).all() or not np.isfinite(bodies).all() or
            np.any(abs(faces+1e-6-face_floor)<=1e-8) or np.any(abs(bodies+1e-6-body_floor)<=1e-8)):
            return None
        return faces,bodies


def select(config,lookup,blocks,tiles):
    if (config is None or config.data['execution'].get('rebuild_nomination_device')!='auto' or
        config.data['models']['device']=='cpu' or len(lookup.identities)<1024):
        return None
    try:
        import torch
        if not torch.cuda.is_available():return None
        free,_=torch.cuda.mem_get_info()
        required=lookup.faces.nbytes+lookup.persons.nbytes+4*1024**2
        if required>min(free//4,config.data['sapiensid']['execution']['reference_bytes']//2):
            return None
        resident=Resident(torch,lookup,blocks)
        if resident.competitive(blocks,tiles):
            count('nomination_cuda_selected')
            return resident
    except (ImportError,RuntimeError,MemoryError,OSError):
        count('nomination_cuda_fallback')
    return None
