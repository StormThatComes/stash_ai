"""Spatially ordered, conservative centroid bounds; no candidate truncation."""
import numpy as np
from .nomination_bounds import ball,upper


def reorder(search):
    """A disk reorder changes traversal only; every centroid remains searchable."""
    if len(search.entries)<256:return
    import sqlite3,tempfile
    from contextlib import closing
    from .scratch_space import connect,file
    with closing(connect()) as ordering:
        ordering.execute('PRAGMA cache_size=-512')
        ordering.execute('CREATE TABLE ordering(code BLOB,ordinal INTEGER)')
        for ordinal in range(len(search.entries)):
            search.index.seek(ordinal*32768)
            row=np.frombuffer(search.index.read(32768),dtype='<f8')
            # Morton order of a fixed projection is deterministic across runs.
            coordinates=np.clip(np.floor(row[:16]*64)+8,0,15).astype('uint8')
            code=np.packbits(np.array([(c>>bit)&1 for bit in range(3,-1,-1) for c in coordinates],dtype='uint8')).tobytes()
            ordering.execute('INSERT INTO ordering VALUES (?,?)',(code,ordinal))
        target=search.stack.enter_context(file(prefix='stash-body-spatial-'))
        entries=[]
        for (ordinal,) in ordering.execute('SELECT ordinal FROM ordering ORDER BY code,ordinal'):
            search.index.seek(ordinal*32768);target.write(search.index.read(32768));entries.append(search.entries[ordinal])
        target.flush();search.index=target;search.entries=entries
        search.mode_owners=np.asarray([owner for owner,_ in entries],dtype='int64')
        search.stats['spatial_centroids']=len(entries)


class Tree:
    def __init__(self,search,read,*,paged=False):
        self.search=search;self.read=read;self.nodes=PagedNodes(search) if paged else []
        leaves=list(range(0,len(search.entries),search.block))
        def build(starts):
            if len(starts)==1:
                start=starts[0];bound=ball(read(start));children=()
                begin,end=start,min(start+search.block,len(search.entries))
            else:
                mid=len(starts)//2;a=build(starts[:mid]);b=build(starts[mid:]);children=(a,b)
                x,y=self.nodes[a],self.nodes[b];begin,end=x[0],y[1]
                center=(x[2][0]+y[2][0])/2
                radius=max(float(np.linalg.norm(center-c[2][0]))+c[2][2] for c in (x,y))+1e-8
                bound=(center,float(np.linalg.norm(center)),radius)
            index=len(self.nodes);self.nodes.append((begin,end,bound,children));return index
        self.root=build(leaves) if leaves else None
    def pairs(self,active=None,floor=None):
        if self.root is None:return
        prefix=np.concatenate(([0],np.cumsum(active,dtype=np.int64))) if active is not None else None
        pending=[(self.root,self.root)]
        while pending:
            a,b=pending.pop();x,y=self.nodes[a],self.nodes[b]
            if prefix is not None and prefix[x[1]]==prefix[x[0]] and prefix[y[1]]==prefix[y[0]]:continue
            required=floor(x[0],x[1],y[0],y[1]) if callable(floor) else floor
            if required is not None and upper(x[2],y[2])+self.search.rounding<required:
                self.search.stats['centroid_subtrees_excluded']=self.search.stats.get('centroid_subtrees_excluded',0)+1
                continue
            if not x[3] and not y[3]:yield x[0],y[0];continue
            if a==b:
                l,r=x[3];pending.extend(((r,r),(l,r),(l,l)))
            elif x[3] and (not y[3] or x[1]-x[0]>=y[1]-y[0]):
                l,r=x[3];pending.extend(((r,b),(l,b)))
            else:
                l,r=y[3];pending.extend(((a,r),(a,l)))


class PagedNodes:
    """Fixed records preserve the same conservative tree under a small RAM cap."""
    def __init__(self,search):
        import struct
        from collections import OrderedDict
        from .scratch_space import file
        self.stream=search.stack.enter_context(file(prefix='stash-body-tree-'))
        self.header=struct.Struct('<qqddqq')
        self.record=self.header.size+4096*8
        self.count=0;self.cache=OrderedDict();self.stats=search.stats
    def __len__(self):return self.count
    def append(self,node):
        begin,end,(center,norm,radius),children=node
        self.stream.seek(self.count*self.record)
        self.stream.write(self.header.pack(begin,end,norm,radius,*(children or (-1,-1))))
        self.stream.write(np.asarray(center,dtype='<f8').tobytes())
        self.count+=1
        self.stats['centroid_tree_disk_bytes']=self.count*self.record
    def __getitem__(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        self.stream.seek(index*self.record)
        data=self.stream.read(self.record)
        if len(data)!=self.record:raise ValueError('Incomplete centroid tree node.')
        begin,end,norm,radius,left,right=self.header.unpack_from(data)
        value=(begin,end,(np.frombuffer(data,dtype='<f8',offset=self.header.size),norm,radius),
               () if left<0 else (left,right))
        self.cache[index]=value
        if len(self.cache)>4:self.cache.popitem(last=False)
        return value
