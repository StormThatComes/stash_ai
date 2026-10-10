"""CPU overhead checks for equivalent preprocessing, floors and max aggregation."""
import argparse
import json
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from stash_ai.backends.onnx_preprocessing import prepare_wd
from stash_ai.inference.prediction import selected_scores


def measure(callback, count):
    callback()
    tracemalloc.start()
    start=time.perf_counter()
    for _ in range(count): callback()
    elapsed=time.perf_counter()-start
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(iterations=count,seconds=elapsed,python_peak_bytes=peak)


def reference_wd(image,size):
    rgba=image.convert('RGBA');background=Image.new('RGBA',rgba.size,(255,255,255,255))
    background.alpha_composite(rgba);rgb=background.convert('RGB');side=max(rgb.size)
    padded=Image.new('RGB',(side,side),(255,255,255));padded.paste(rgb,((side-rgb.width)//2,(side-rgb.height)//2))
    resized=padded.resize((size,size),Image.Resampling.BICUBIC)
    result=np.ascontiguousarray(np.asarray(resized,dtype=np.float32)[:,:,::-1])
    for value in (rgba,background,rgb,padded,resized):value.close()
    return result


def run(count):
    result={}
    with Image.new('RGB',(768,512),(21,42,84)) as image:
        result['wd_rgb']={name:measure(lambda:fn(image,448),count) for name,fn in [('reference',reference_wd),('optimized',prepare_wd)]}
    rng=np.random.default_rng(7)
    for kind,row in [('sparse',np.full(7000,.1,np.float32)),('dense',rng.random(7000,dtype=np.float32))]:
        def reference():
            return [(int(i),float(row[i])) for i in np.flatnonzero(row.astype(np.float64)>=.8)]
        result['scores_'+kind]={'reference':measure(reference,count*10),'optimized':measure(lambda:selected_scores(row,.8),count*10)}
    batches=rng.random((10,32,7000),dtype=np.float32)
    def reference_max():
        scores=np.empty((0,7000),np.float32)
        for batch in batches:
            combined=np.concatenate((scores,batch));scores=np.partition(combined,len(combined)-1,axis=0)[-1:]
        return scores
    def optimized_max():
        scores=np.empty((0,7000),np.float32)
        for batch in batches:
            maximum=np.max(batch,axis=0,keepdims=True);scores=np.maximum(scores,maximum) if len(scores) else maximum
        return scores
    np.testing.assert_array_equal(reference_max(),optimized_max())
    result['max']={'reference':measure(reference_max,count),'optimized':measure(optimized_max,count)}
    import torch
    from stash_ai.backends.loader import JoyTagger
    from stash_ai.models import JOYTAG
    from stash_ai.runtime.report import RunReport
    from stash_ai.runtime.profile import ProfileMetrics
    tagger=JoyTagger.__new__(JoyTagger);tagger.device=torch.device('cpu');tagger.settings=SimpleNamespace(source=JOYTAG)
    tagger.model=lambda inputs:dict(tags=inputs['image'].mean(dim=(2,3)))
    prepared=torch.zeros((2,3,8,8))
    result['optional_cpu_profiling']={}
    for mode in ('disabled','enabled'):
        tagger.report=RunReport(write=lambda _:None) if mode=='enabled' else None
        if tagger.report is not None:tagger.report.profile=ProfileMetrics()
        result['optional_cpu_profiling'][mode]=measure(lambda:tagger._predict_collated_scores(prepared,2),count*10)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--iterations',type=int,default=30)
    args=parser.parse_args();print(json.dumps(run(args.iterations),indent=2))


if __name__=='__main__':main()
