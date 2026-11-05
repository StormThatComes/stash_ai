"""Lossless blocked query nomination; exact witnesses decide all finalists."""
import numpy as np


class QueryIndex:
    def __init__(self,banks,budget):
        self.banks=banks;self.means=None;self.batch_rows=1
        if not banks:return
        dimension=banks[0][3].shape[1]
        sources=sum(f.nbytes+b.nbytes for _,_,b,f in banks)
        # Leave room for the source owners, one exact block and centroid bank.
        if sources*4+len(banks)*dimension*8+64*dimension*8>budget:return
        available=budget-sources*4-len(banks)*dimension*8-64*dimension*8
        self.batch_rows=max(1,min(16,available//(dimension*8+(len(banks)+7)//8+2048)))
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

    def batch(self, rows, floor):
        """Bound temporary score masks and preserve observation and bank order."""
        from itertools import islice
        rows=iter(rows)
        while group:=list(islice(rows,self.batch_rows)):
            if self.means is None or len(group)==1:
                for row,face in group:
                    yield row,face,self.candidates(face,floor)
                continue
            # Use one bit per bank/query rather than an N-by-Q array of indices.
            queries=np.asarray([face for _,face in group],dtype='float64')
            bits=np.empty((len(group),(len(self.banks)+7)//8),dtype='uint8')
            for start in range(0,len(self.banks),64):
                mask=(queries@self.means[start:start+64].T)+1e-6>=floor
                packed=np.packbits(mask,axis=1,bitorder='little')
                bits[:,start//8:start//8+packed.shape[1]]=packed
            def selected(mask):
                for offset,byte in enumerate(mask):
                    for bit in range(8):
                        if int(byte)&(1<<bit):yield self.banks[offset*8+bit]
            for (row,face),mask in zip(group,bits):
                yield row,face,selected(mask)
