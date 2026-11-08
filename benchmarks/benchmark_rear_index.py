"""Compare selective nomination with exact replay on synthetic scene envelopes."""
import argparse
from contextlib import closing
import json
import sqlite3
import time
from types import SimpleNamespace

import numpy as np

from stash_ai.recognition.rear_gallery import Gallery
from stash_ai.recognition.rear_pending import _bank_versions
from stash_ai.recognition.rear_reverse import ReverseIndex
from stash_ai.recognition.config import digest


def run(scenes):
    rng = np.random.default_rng(749)
    # A population of clean front/side/back appearances in a 4096-D encoder.
    # Reuse vectors across scenes to represent repeated appearances.
    angles = rng.uniform(-np.pi,np.pi,128)
    vectors = np.zeros((len(angles),4096),dtype='<f4')
    vectors[:,0],vectors[:,1] = np.cos(angles),np.sin(angles)
    def catalogue(owners):
        banks = [(owner,'external',[dict(id=owner,roots=[owner+'a',owner+'b'],quality='{}',_vector=v)])
                 for owner,v in owners]
        gallery = Gallery(banks)
        return SimpleNamespace(gallery=gallery,banks=_bank_versions(gallery),radius=.56,floor=.8,
                               search=SimpleNamespace(mean_versions={}))
    old = catalogue([('A',vectors[0]),('B',vectors[51])])
    new = catalogue([('A',vectors[0]),('C',vectors[85])])
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value)')
        index=ReverseIndex(db,'benchmark','encoder','v',old)
        start=time.perf_counter()
        assignment=rng.integers(0,len(vectors),scenes)
        def nearby(q,v):
            return np.linalg.norm(q-v)<=old.radius
        def evidence(q,catalogue):
            return tuple(p for p,_,rows in catalogue.gallery.banks if nearby(q,rows[0]['_vector']))
        existing = [evidence(v,old) for v in vectors]
        for i,k in enumerate(assignment):
            index.post(str(i),index.bins(vectors[k],.003)|{'o'+digest(owner) for owner in existing[k]})
            index.state['scans'][str(i)]=[str(i)]
        index.finish()
        elapsed=time.perf_counter()-start
        start=time.perf_counter()
        index=ReverseIndex(db,'benchmark','encoder','v',new)
        chosen=index.nominate(new,[])
        expected=[evidence(v,new) for v in vectors]
        wrong=sum((expected[k] if str(i) in chosen else existing[k])!=expected[k] for i,k in enumerate(assignment))
        return dict(scene_headers=scenes,nominated=len(chosen),full_comparison=scenes,disagreements=wrong,
                    initial_posting_seconds=round(elapsed,3),select_and_verify_seconds=round(time.perf_counter()-start,3))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenes',type=int,default=10000)
    parser.add_argument('--profile',choices=('micro','library'),default='micro')
    parser.add_argument('--face-config',help='Place the library profile SQLite workspace under the configured recognition work root.')
    args = parser.parse_args()
    if args.profile=='library':
        from .benchmark_scene_library import run as library
        result = library(args.scenes,args.face_config)
    else:
        result=run(args.scenes)
    print(json.dumps(result))
    if result['disagreements']:
        raise SystemExit(1)
