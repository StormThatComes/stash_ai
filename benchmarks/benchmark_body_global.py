"""Synthetic full body planner: disk centroids, exact candidates, outsider guards.

Excludes model inference, media, source catalogue SQL and Stash publication.
Synthetic embeddings are generated on demand to avoid an N*references*4096 bank.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from stash_ai.recognition.body_global import plan
from stash_ai.recognition.config import defaults


class Source:
    def __init__(self,matches):self.matches=matches;self.vectors=OrderedDict()
    def execute(self,sql,args):
        person=int(args[0].split('-')[0]);center=person//2 if person<self.matches*2 else person
        if center not in self.vectors:
            value=np.random.default_rng(20261028+center).normal(size=4096).astype('<f4')
            value/=np.linalg.norm(value)
            self.vectors[center]=value.tobytes()
            if len(self.vectors)>64:self.vectors.popitem(last=False)
        self.vectors.move_to_end(center)
        return SimpleNamespace(fetchone=lambda:dict(embedding=self.vectors[center]))


def run(count=32,samples=16,matches=4,threads=1):
    config=SimpleNamespace(data=defaults())
    config.data['sapiensid']['global_matching']['enabled']=True
    bank={}
    for i in range(count):
        person=f'{i:05d}'
        refs=[dict(id=f'{i:05d}-{j:02d}',scan=person,cue=j) for j in range(samples)]
        bank[person]=dict(refs=refs,cores=[refs],faces=[],face_ids=[],support=samples,
            missing_original=False,cues={(person,j) for j in range(samples)})
    with threadpool_limits(limits=threads):
        start=time.perf_counter();result=plan(Source(matches),config,bank);elapsed=time.perf_counter()-start
    accepted=[d for d in result['decisions'] if d['outcome']=='merge']
    expected=[[f'{2*i:05d}',f'{2*i+1:05d}'] for i in range(matches)]
    if [d['people'] for d in accepted]!=expected:raise AssertionError('Unexpected synthetic partition.')
    return dict(samples_per_identity=samples,dimensions=4096,native_threads=threads,
        seconds=elapsed,expected_merges=matches,actual_merges=len(accepted),**result['search'],
        previous_body_cross_products=count*(count-1)*samples*samples,
        limits=config.data['sapiensid']['execution'],
        limitations=__doc__.strip())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identities',type=int,default=32)
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--matches',type=int,default=4)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 2<=args.identities<=10000 or not 3<=args.samples<=32 or not 0<=args.matches<=args.identities//2:
        parser.error('Use 2..10000 identities, 3..32 samples and at most identities/2 matches.')
    text=json.dumps(run(args.identities,args.samples,args.matches,args.threads),indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')
