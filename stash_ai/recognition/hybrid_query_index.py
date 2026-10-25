"""Lossless blocked query nomination; exact witnesses decide all finalists."""
import numpy as np


class QueryIndex:
    def __init__(self,banks,budget):
        self.banks=banks;self.means=None
        if not banks:return
        dimension=banks[0][3].shape[1]
        sources=sum(f.nbytes+b.nbytes for _,_,b,f in banks)
        # Leave room for the source owners, one exact block and centroid bank.
        if sources*4+len(banks)*dimension*8+64*dimension*8>budget:return
        self.means=np.asarray([f.mean(axis=0,dtype='float64') for _,_,_,f in banks])

    def candidates(self,face,floor):
        if self.means is None:
            yield from self.banks
            return
        query=face.astype('float64',copy=False)
        for start in range(0,len(self.banks),64):
            scores=self.means[start:start+64] @ query
            # Mean >= minimum. Overnomination protects rounding at the gate.
            for offset in np.flatnonzero(scores+1e-6>=floor):
                yield self.banks[start+int(offset)]
