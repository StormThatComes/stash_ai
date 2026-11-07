"""Exact shortlist by mean upper bound, using bounded RAM or scratch pages."""
import numpy as np


class QueryIndex:
    def __init__(self,banks,budget):
        self.banks=banks;self.means=self.disk=None;self.batch_rows=1;self.tile=64
        if not banks:return
        dimension=banks[0][3].shape[1];self.dimension=dimension
        sources=sum(f.nbytes+b.nbytes for _,_,b,f in banks)
        available=budget-sources*4
        mean_bytes=len(banks)*dimension*8
        if available>=mean_bytes+64*dimension*8:
            self.means=np.empty((len(banks),dimension),dtype='float64')
            for i,(_,_,_,f) in enumerate(banks):self.means[i]=f.mean(axis=0,dtype='float64')
            available-=mean_bytes+64*dimension*8
        else:
            # At least one centroid is needed to score a query. Never allocate the
            # full bank or a full nomination mask on the fallback path.
            self.tile=max(1,min(64,max(0,available)//max(1,dimension*16)))
            from .scratch_space import connect
            self.disk=connect()
            try:
                self.disk.execute('PRAGMA cache_size=-128')
                self.disk.execute('CREATE TABLE centroids(start INTEGER PRIMARY KEY, data BLOB)')
                for start in range(0,len(banks),self.tile):
                    block=np.empty((min(self.tile,len(banks)-start),dimension),dtype='float64')
                    for i in range(len(block)):block[i]=banks[start+i][3].mean(axis=0,dtype='float64')
                    self.disk.execute('INSERT INTO centroids VALUES (?,?)',(start,block.tobytes()))
                self.disk.commit()
            except BaseException:
                self.close();raise
            return
        self.batch_rows=max(1,min(16,available//(dimension*8+(len(banks)+7)//8+2048)))

    def close(self):
        if self.disk is not None:
            self.disk.close();self.disk=None
        self.means=None

    def blocks(self):
        if self.disk is not None:
            for start,data in self.disk.execute('SELECT start,data FROM centroids ORDER BY start'):
                yield start,np.frombuffer(data,dtype='float64').reshape(-1,self.dimension)
        elif self.means is not None:
            for start in range(0,len(self.banks),64):yield start,self.means[start:start+64]

    def candidates(self,face,floor):
        query=face.astype('float64',copy=False)
        for start,centroids in self.blocks():
            scores=centroids @ query
            for offset in np.flatnonzero(scores+1e-6>=floor):
                yield self.banks[start+int(offset)]

    def batch(self, rows, floor):
        from itertools import islice
        rows=iter(rows)
        while group:=list(islice(rows,self.batch_rows)):
            if len(group)==1:
                for row,face in group:yield row,face,self.candidates(face,floor)
                continue
            queries=np.asarray([face for _,face in group],dtype='float64')
            bits=np.empty((len(group),(len(self.banks)+7)//8),dtype='uint8')
            for start,centroids in self.blocks():
                mask=(queries@centroids.T)+1e-6>=floor
                packed=np.packbits(mask,axis=1,bitorder='little')
                bits[:,start//8:start//8+packed.shape[1]]=packed
            def selected(mask):
                for offset,byte in enumerate(mask):
                    for bit in range(8):
                        if int(byte)&(1<<bit):yield self.banks[offset*8+bit]
            for (row,face),mask in zip(group,bits):yield row,face,selected(mask)
