"""Upper bounds on dot products between two complete vector blocks."""
import numpy as np


def ball(rows):
    center=rows.mean(axis=0,dtype='float64')
    radius=float(np.max(np.linalg.norm(rows-center,axis=1)))
    return center,float(np.linalg.norm(center)),radius


def upper(left,right):
    x,nx,rx=left
    y,ny,ry=right
    # All scoring vectors are normalized or means of normalized vectors.
    # This envelope exceeds the float64 reductions for the supported dimensions.
    return float(np.dot(x,y))+nx*ry+ny*rx+rx*ry+1e-8
