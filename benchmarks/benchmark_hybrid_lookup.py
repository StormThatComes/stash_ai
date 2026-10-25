"""Synthetic nomination parity and CPU timing; no models, media or Stash calls."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from stash_ai.recognition.hybrid_evidence import Identity,scores
from stash_ai.recognition.hybrid_lookup import Lookup


def run(count=128,samples=8):
    rng=np.random.default_rng(20261025);bank=[]
    centers=rng.normal(size=((count+1)//2,4096)).astype('float32')
    centers/=np.linalg.norm(centers,axis=1)[:,None]
    for index in range(count):
        body=centers[index//2]+rng.normal(scale=.002,size=(samples,4096)).astype('float32')
        body/=np.linalg.norm(body,axis=1)[:,None]
        face=body[:,:512].copy();face/=np.linalg.norm(face,axis=1)[:,None]
        bank.append(Identity(str(index),(),face,body,()))
    def baseline():
        return [(a.person,b.person) for a,b in combinations(bank,2)
                if scores(a.faces,b.faces)['mean']>=.4 or scores(a.persons,b.persons)['mean']>=.75]
    def blocked():
        return [(a,b) for a,b,face,body in Lookup(bank,2**30).pairs() if face+1e-6>=.4 or body+1e-6>=.75]
    timings={};outputs={}
    with threadpool_limits(limits=1):
        for name,method in (('exhaustive_witness_nomination',baseline),('blocked_centroid_nomination',blocked)):
            start=time.perf_counter();outputs[name]=sorted(method())
            timings[name]=time.perf_counter()-start
    same=len({json.dumps(value) for value in outputs.values()})==1
    if not same:raise AssertionError('Synthetic nominations changed.')
    return dict(identities=count,samples_per_identity=samples,person_dimensions=4096,
        face_dimensions=512,native_threads=1,compared_pairs=count*(count-1)//2,
        nominated_pairs=len(outputs['blocked_centroid_nomination']),nomination_parity=same,
        nominations_sha256=hashlib.sha256(json.dumps(outputs['blocked_centroid_nomination']).encode()).hexdigest(),
        seconds=timings,speedup=timings['exhaustive_witness_nomination']/timings['blocked_centroid_nomination'],
        limitations='Synthetic CPU nomination only; excludes exact candidate gates, DB I/O, inference and media. Not an accuracy or end-to-end speed claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identities',type=int,default=128)
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 2<=args.identities<=4096 or not 1<=args.samples<=32:parser.error('Use 2..4096 identities and 1..32 samples.')
    text=json.dumps(run(args.identities,args.samples),indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(text)
    else:print(text,end='')
