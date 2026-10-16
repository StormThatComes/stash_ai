"""Isolated random-weight FaceOcc scheduling probe, not a face-accuracy test."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time


def worker(threads,batch,repeats):
    import numpy as np
    import torch
    from PIL import Image
    from stash_ai.recognition.faceocc import build_model
    from stash_ai.recognition.models import checked_detection
    from stash_ai.recognition.visibility import prepare,measurement
    from stash_ai.runtime.memory import process_rss
    default=torch.get_num_threads();torch.set_num_threads(threads);torch.manual_seed(7)
    model=build_model().eval()
    face=checked_detection([64,64,128,128,96,108,160,108,128,136,105,160,151,160,.99])
    with Image.fromarray(np.random.default_rng(5).integers(0,256,(256,256,3),dtype='uint8')) as image:
        blob,region=prepare(image,face)
    data=np.concatenate([blob]*batch)
    samples=[];stopped=threading.Event()
    def observe():
        while not stopped.wait(.01):
            value=process_rss()
            if value is not None:samples.append(value)
    watcher=threading.Thread(target=observe);watcher.start()
    timings={};outputs={}
    try:
        with torch.inference_mode():
            for mode in ('scalar','batch'):
                times=[]
                for repeat in range(repeats+1):
                    started=time.perf_counter()
                    if mode=='scalar':
                        masks=np.concatenate([model(torch.from_numpy(value[None])).numpy() for value in data])
                    else:masks=model(torch.from_numpy(data)).numpy()
                    outputs[mode]=[measurement(masks[i:i+1],region,'cpu')['visibility']['visible_pixels'] for i in range(batch)]
                    if repeat:times.append(time.perf_counter()-started)
                timings[mode]=dict(median_seconds=statistics.median(times),trials_seconds=times)
    finally:
        stopped.set();watcher.join()
    return dict(native_threads=threads,default_native_threads=default,faces=batch,
        input_bytes=data.nbytes,full_mask_bytes=batch*256*256*4,device_count_bytes=batch*8+1,
        sampled_peak_rss_bytes=max(samples) if samples else None,rss_interval_seconds=.01,
        max_batch_scalar_pixel_delta=max(abs(a-b) for a,b in zip(outputs['scalar'],outputs['batch'])),
        measurements=timings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',nargs='+',type=int,default=[1,2,4])
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if not 1<=args.batch<=8 or args.repeats<1 or any(not 1<=n<=16 for n in args.threads):parser.error('invalid bounded probe size')
    if args.worker:
        print(json.dumps(worker(args.threads[0],args.batch,args.repeats)));return
    results=[]
    for threads in args.threads:
        process=subprocess.run([sys.executable,'-m','benchmarks.benchmark_portrait_quality','--worker',
            '--threads',str(threads),'--batch',str(args.batch),'--repeats',str(args.repeats)],
            capture_output=True,text=True,check=True,timeout=120)
        results.append(json.loads(process.stdout))
    print(json.dumps(dict(scope='random-weight-faceocc-scheduling-only',python=platform.python_version(),
        platform=platform.platform(),accuracy=None,trained_weights=False,results=results),indent=2))


if __name__=='__main__':main()
