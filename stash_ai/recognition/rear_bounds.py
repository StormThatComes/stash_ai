"""Disk-backed conservative bounds for exact rear-query competitor searches."""
import numpy as np

from .scratch_space import file


class RivalBounds:
    def __init__(self, search):
        self.file = search.stack.enter_context(file(prefix='stash-rear-bounds-'))
        for start in range(0,len(search.entries),search.block):
            count = min(search.block,len(search.entries)-start)
            search.index.seek(start*4096*8)
            rows = np.frombuffer(search.index.read(count*4096*8),dtype='<f8').reshape(count,4096)
            center = rows.mean(axis=0)
            radius = np.linalg.norm(rows-center,axis=1).max()
            self.file.write(center.astype('<f8',copy=False).tobytes())
            self.file.write(np.asarray([radius],dtype='<f8').tobytes())
        self.file.flush()

    def possible(self, block, queries, norms, floors):
        self.file.seek(block*4097*8)
        record = np.frombuffer(self.file.read(4097*8),dtype='<f8')
        # q.x <= q.center + ||q|| * radius. Keep the exact scorer's tolerance.
        return record[:-1] @ queries + norms*record[-1] + 1e-6 + 1e-8 >= floors


def changed_modes(search, previous, budget):
    """A small changed set can invalidate just nearby scans. None means revisit all."""
    if previous is None:
        return None
    changed = {owner for owner,v in search.mean_versions.items() if previous.get(owner)!=v}
    if not changed:
        return ()
    addresses = []
    limit = min(32,max(0,budget//(16*4096*8)))
    for i,(ordinal,_) in enumerate(search.entries):
        if search.identities[ordinal] in changed:
            addresses.append(i)
            if len(addresses)>limit:
                return None
    result = []
    for i in addresses:
        search.index.seek(i*4096*8)
        vector = np.frombuffer(search.index.read(4096*8),dtype='<f8')
        result.append((vector,float(np.linalg.norm(vector))))
    return result
