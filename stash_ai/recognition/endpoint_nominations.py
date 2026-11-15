"""Index original strict face endpoints to nominate cross-perspective identity pairs."""
import math

from .resource_errors import ResourceLimit


def nominate(owners, representatives, face, floor, pair_limit, byte_limit):
    import numpy as np
    from scipy.spatial import cKDTree
    samples, keys, seen = [], [], set()
    charged = 0
    for owner in sorted(owners):
        for _,scan,_,_,roots in representatives(owner):
            # A verified pair must clear the floor on all four cross comparisons.
            # One real endpoint of each retained view suffices for lossless nomination.
            address = owner,scan,roots[0]
            if address in seen:
                continue
            item = face(scan,roots[0],owner)
            if item is None:
                continue
            vector = item[0]
            charged += 28*len(vector)+512
            if charged>byte_limit:
                raise ResourceLimit('Endpoint face index exceeds the recognition reference budget.')
            samples.append(vector)
            keys.append(owner)
            seen.add(address)
    if len(samples)<2:
        return set()
    matrix = np.asarray(samples,dtype='float64')
    # Retained unit vectors permit a small tolerance. Cover their actual norms
    # and float32 dot-product rounding before exact verification.
    norm2 = np.einsum('ij,ij->i',matrix,matrix)
    radius = math.sqrt(max(0.,2*(float(norm2.max())-floor+1.e-5)))
    index = cKDTree(matrix)
    pairs = set()
    for i, sample in enumerate(matrix):
        for j in index.query_ball_point(sample,radius):
            if j>i and keys[j]!=keys[i]:
                pairs.add((keys[i],keys[j]))
                if len(pairs)>pair_limit:
                    raise ResourceLimit('Too many distinct cross-perspective endpoint nominations.')
    return pairs
