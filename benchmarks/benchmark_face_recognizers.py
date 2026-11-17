"""Compare SFace/AuraFace/LVFace/AdaFace at a common calibrated false-match rate.

Manifest JSON: list of {image, face: [15 detection values], identity, scene,
split: calibration|validation, category: profile|occluded|small|ordinary}.
All models consume the same annotated source detections. Split identities AND
scenes must be disjoint. Model-specific cosine thresholds never cross namespaces.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

MODELS=('sface','auraface','lvface_b','adaface_ir50')


def validate_manifest(rows):
    if not rows:raise ValueError('Empty manifest.')
    for row in rows:
        if row.get('split') not in ('calibration','validation'):raise ValueError('Invalid split.')
        if not row.get('identity') or not row.get('scene'):raise ValueError('Identity and scene labels are required.')
    for field in ('identity','scene'):
        left={r[field] for r in rows if r['split']=='calibration'}
        right={r[field] for r in rows if r['split']=='validation'}
        if left & right:raise ValueError(f'{field} leakage between calibration and validation.')


def pairs(rows, embeddings, maximum=1000000):
    vectors=np.asarray(embeddings,dtype=float)
    if vectors.ndim!=2 or len(vectors)!=len(rows) or not np.isfinite(vectors).all():raise ValueError('Invalid embeddings.')
    norms=np.linalg.norm(vectors,axis=1)
    if (norms<=0).any():raise ValueError('Zero embedding.')
    vectors=vectors/norms[:,None]
    output={'calibration':[],'validation':[]};total=0
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j]
            if a['split']!=b['split'] or a['scene']==b['scene']:continue
            total+=1
            if total>maximum:raise ValueError('Pair budget exceeded; use a smaller held-out manifest or raise --max-pairs.')
            output[a['split']].append((float(np.clip(vectors[i]@vectors[j],-1,1)),a['identity']==b['identity'],
                                      sorted({a.get('category','ordinary'),b.get('category','ordinary')})))
    return output


def calibrate(samples,target):
    if not 0<=target<1:raise ValueError('False-match target must be in [0,1).')
    negative=sorted((score for score,same,_ in samples if not same),reverse=True)
    if not negative or not any(same for _,same,_ in samples):raise ValueError('Calibration needs genuine and impostor cross-scene pairs.')
    allowed=math.floor(target*len(negative))
    return float(np.nextafter(negative[allowed],math.inf))


def rates(samples,threshold):
    genuine=[score for score,same,_ in samples if same]
    impostor=[score for score,same,_ in samples if not same]
    return dict(genuine_pairs=len(genuine),impostor_pairs=len(impostor),
                true_match_rate=sum(s>=threshold for s in genuine)/len(genuine) if genuine else None,
                false_match_rate=sum(s>=threshold for s in impostor)/len(impostor) if impostor else None,
                false_match_resolution=1/len(impostor) if impostor else None)


def evaluate(rows,embeddings,target=.001,maximum=1000000):
    validate_manifest(rows)
    sample=pairs(rows,embeddings,maximum)
    threshold=calibrate(sample['calibration'],target)
    if not sample['validation']:raise ValueError('No held-out cross-scene pairs.')
    return dict(threshold=threshold,target_false_match_rate=target,
                calibration=rates(sample['calibration'],threshold),validation=rates(sample['validation'],threshold),
                strata={category:rates([r for r in sample['validation'] if category in r[2]],threshold)
                        for category in sorted({c for r in sample['validation'] for c in r[2]})})


def run(config,manifest,models,target,maximum):
    from PIL import Image
    from stash_ai.recognition.models import load_backend,ensure_assets,checked_detection
    from stash_ai.recognition.config import plain,_freeze
    raw=manifest.read_bytes(); rows=json.loads(raw);validate_manifest(rows)
    result=dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),models={})
    try:result['commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):result['commit']=None
    for name in models:
        if name not in MODELS:raise ValueError(f'Unsupported face recognizer: {name}')
        data=plain(config.data);data['models']['recognizer']=name
        selected=replace(config,data=_freeze(data));started=time.perf_counter()
        backend=load_backend(selected,ensure_assets(selected))
        try:
            vectors=[]
            for row in rows:
                with Image.open(manifest.parent/row['image']) as image:
                    vectors.append(backend.embed(image,checked_detection(row['face'])))
            report=evaluate(rows,vectors,target,maximum)
            report.update(seconds=time.perf_counter()-started,provenance=backend.provenance)
            result['models'][name]=report
        finally:backend.close()
    return result


def main():
    from stash_ai.recognition.config import load
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--face-config',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--false-match-rate',type=float,default=.001);p.add_argument('--max-pairs',type=int,default=1000000)
    args=p.parse_args()
    report=run(load(args.face_config),args.manifest,args.models,args.false_match_rate,args.max_pairs)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
