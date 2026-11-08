"""Optional paired KPR/SapiensID retrieval on identical labelled contact crops."""
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
from itertools import groupby
import subprocess
import sys
import time

import numpy as np

from stash_ai.recognition.crowd_labels import torso_owner


def prompts(image,people,index):
    target = people[index]
    left,top,right,bottom = target['bounds']
    def points(person):
        xy = np.asarray(person['keypoints'],dtype=np.float32)
        confidence = np.asarray(person.get('keypoint_visibility',person.get('keypoint_scores',[1.]*17)))
        visible = ((xy[:,0]>=left)&(xy[:,0]<right)&(xy[:,1]>=top)&(xy[:,1]<bottom)&(confidence>=.5))
        return np.where(visible[:,None],np.column_stack((xy-[left,top],confidence)),0).astype(np.float32)
    negative = [points(person) for i,person in enumerate(people) if i!=index]
    with image.crop(target['bounds']) as crop, crop.convert('RGB') as rgb:
        # Upstream demo uses cv2.imread (BGR).
        pixels = np.ascontiguousarray(np.asarray(rgb)[:,:,::-1])
    return dict(image=pixels,keypoints_xyc=points(target),
                negative_kps=np.asarray(negative,dtype=np.float32).reshape(-1,17,3))


class Extractor:
    """Run the optional upstream stack in its own Python environment."""
    def __init__(self,config,weights,directory,device,python=None):
        if not weights.is_file() or not config.is_file():
            raise ValueError('KPR comparison requires an existing upstream YAML and trained checkpoint.')
        self.config,self.weights = config.resolve(),weights.resolve()
        self.directory = directory/'kpr-inputs'
        self.directory.mkdir()
        self.python = python or sys.executable
        self.device = device
        self.batches = 0

    def stage(self,samples):
        batch = self.batches
        self.batches += 1
        fields = {f'{key}_{i}':value for i,sample in enumerate(samples) for key,value in sample.items()}
        np.savez(self.directory/f'input-{batch}.npz',**fields)
        return [(batch,i) for i in range(len(samples))]

    def extract(self,rows):
        if not rows:
            return
        worker = Path(__file__).with_name('kpr_worker.py')
        subprocess.run([str(self.python),str(worker),str(self.config),str(self.weights),str(self.directory),
                        str(self.batches),self.device],check=True)
        self.device = (self.directory/'device.txt').read_text()
        for batch,items in groupby(rows,key=lambda r:r['kpr'][0]):
            with np.load(self.directory/f'output-{batch}.npz',allow_pickle=False) as result:
                features,visible = result['features'],result['visible']
                if not np.isfinite(features).all() or not np.isfinite(visible).all() or np.any(visible<0):
                    raise ValueError('Invalid KPR part features or visibility.')
                for row in items:
                    i = row['kpr'][1]
                    row['kpr'] = features[i],visible[i]

    @staticmethod
    def distances(query,references):
        # Upstream mean of Euclidean part distances, weighted by shared visibility.
        features,visible = (np.stack(values) for values in zip(*references))
        weights = np.sqrt(visible*query[1])
        parts = np.linalg.norm(features-query[0],axis=-1)
        total = weights.sum(axis=-1)
        return np.divide((weights*parts).sum(axis=-1),total,out=np.full_like(total,np.inf,dtype=float),where=total>0)


def closest(distances, refs):
    by_owner = defaultdict(list)
    for distance,ref in zip(distances,refs):
        if np.isfinite(distance):
            by_owner[ref['truth']].append(float(distance))
    ranked = sorted((sum(v)/len(v),owner) for owner,v in by_owner.items())
    if not ranked or len(ranked)>1 and abs(ranked[0][0]-ranked[1][0])<1e-7:
        return None
    return ranked[0][1]


def compare(rows, counts, extractor):
    refs = [r for r in rows if r['reference']]
    totals = {model:Counter() for model in ('sapiensid','kpr')}
    for query in rows:
        if query['reference']:
            continue
        eligible = [r for r in refs if r['cue']!=query['cue'] and r['pixels']!=query['pixels']]
        if query['truth'] not in {r['truth'] for r in eligible} or len({r['truth'] for r in eligible})<2:
            counts['without_competing_references'] += 1
            continue
        sapiens = [1.-float(np.dot(query['body'],r['body'])) for r in eligible]
        kpr = extractor.distances(query['kpr'],[r['kpr'] for r in eligible])
        for model,values in [('sapiensid',sapiens),('kpr',kpr)]:
            owner = closest(values,eligible)
            result = ('abstained' if owner is None else 'own' if owner==query['truth']
                      else 'partner' if owner in query['others'] else 'foreign')
            totals[model][result] += 1
            if query['mixed_wrist']:
                totals[model]['mixed_wrist_'+result] += 1
    return dict(counts=counts,paired_top1=totals)


def run(db,selected,labels,source_for,extractor):
    from hashlib import sha256
    from stash_ai.recognition.body_identity import vector
    rows = []
    counts = Counter()
    started = time.monotonic()
    for item in selected:
        scene,scan = item['scene'],item['scan']
        with source_for(scene) as source, closing(source.frames_indices(sorted(map(int,labels[scene])))) as frames:
            for cue,_,image in frames:
                with image:
                    persons = labels[scene].get(str(cue))
                    if not persons:
                        continue
                    raw = list(db.execute('SELECT geometry,embedding,rejected FROM body_observations '
                                          'WHERE scan=? AND cue=? ORDER BY id',(scan,cue)))
                    poses = [json.loads(r[0]) for r in raw]
                    aligned = [torso_owner(p,persons) for p in poses]
                    selected_poses = []
                    for label in persons:
                        ref,truth = label.get('reference',False),label['truth']
                        if not ref and len(persons)<2:
                            continue
                        counts['labelled_references' if ref else 'labelled_contact_people'] += 1
                        if aligned.count(truth)!=1:
                            counts['missing_or_ambiguous_torso'] += 1
                            continue
                        i = aligned.index(truth)
                        if raw[i][1] is None or raw[i][2] is not None:
                            counts['unavailable_body_feature'] += 1
                            continue
                        sample = prompts(image,poses,i)
                        wrist_mix = False
                        for side,point in label.get('wrists',{}).items():
                            joint = poses[i]['keypoints'][9 if side=='left' else 10]
                            if min(joint)>=0:
                                wrist_mix |= any(np.linalg.norm(np.subtract(joint,p['wrists'][side])) <
                                    np.linalg.norm(np.subtract(joint,point)) for p in persons
                                    if p['truth']!=truth and side in p.get('wrists',{}))
                        selected_poses.append((sample,dict(reference=ref,truth=truth,cue=(scene,cue),
                            others={p['truth'] for p in persons if p['truth']!=truth},mixed_wrist=wrist_mix,
                            body=vector(dict(embedding=raw[i][1])),pixels=sha256(sample['image'].tobytes()).digest())))
                    if selected_poses:
                        samples,items = zip(*selected_poses)
                        for features,row in zip(extractor.stage(samples),items,strict=True):
                            rows.append(dict(row,kpr=features))
    extractor.extract(rows)
    result = compare(rows,counts,extractor)
    return dict(result,device=extractor.device,elapsed_seconds=time.monotonic()-started)
