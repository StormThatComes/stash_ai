"""Compare PMPose and its two SAM overlap modes on labelled VTT cues from read-only Stash."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing, nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import time


def score(db, selected, labels):
    from stash_ai.recognition.crowd_context import torso
    scanned = {s['scene']:s['scan'] for s in selected}
    observations = []
    anchors = defaultdict(set)
    counts = Counter()
    for scene,cues in labels.items():
        for cue,persons in cues.items():
            if not persons or len({p['truth'] for p in persons}) != len(persons):
                raise ValueError('Each labelled cue needs one torso point per distinct visible performer.')
            detections = defaultdict(list)
            for geometry, person, status in db.execute(
                    "SELECT b.geometry,m.person,m.status FROM body_observations b LEFT JOIN body_members m "
                    "ON m.observation=b.id WHERE b.scan=? AND b.cue=?", (scanned[scene],int(cue))):
                pose = json.loads(geometry)
                centre = torso(pose)
                if centre is None:
                    continue
                left,top,right,bottom = pose['bounds']
                size = max(1.,math.hypot(right-left,bottom-top))
                ranked = sorted((math.dist(centre,p['point'])/size,p['truth']) for p in persons)
                if ranked[0][0]>.25 or len(ranked)>1 and ranked[1][0]-ranked[0][0]<.05:
                    continue
                detections[ranked[0][1]].append(person if status=='accepted' else None)
            visible = {}
            for label in persons:
                truth = label['truth']
                matches = detections[truth]
                counts['duplicate_poses'] += max(0,len(matches)-1)
                counts['missing_poses'] += not matches
                visible[truth] = matches[0] if len(matches)==1 else None
                if label.get('reference') and visible[truth] is not None:
                    anchors[truth].add(visible[truth])
            if len(visible)==2 and not any(p.get('reference') for p in persons):
                observations.append(visible)
    for visible in observations:
        a,b = visible
        own,partner = visible[a],visible[b]
        counts['labelled_pairs'] += 1
        counts['partner_false_joins'] += bool(own is not None and (own==partner or own in anchors[b])
                                               or partner is not None and partner in anchors[a])
        counts['both_correct'] += bool(own is not None and partner is not None and own!=partner
                                       and anchors[a]=={own} and anchors[b]=={partner})
    return dict(counts)


def variant(config, name, database):
    from stash_ai.recognition.config import plain
    data = plain(config.data)
    pose = data['sapiensid']['pose']
    if pose['backend'] != 'pmpose' or not data['sapiensid']['identities']['enabled']:
        raise ValueError('Use a PMPose configuration with body identities enabled.')
    pose['refinement'].update(enabled=name!='unmasked',mode='overlap',masked_identity_crop=name=='masked_identity')
    data['execution'].update(max_cues=0,media_cache_bytes=0)
    return replace(config,data=data,database=database)


def pose_records(db, selected):
    records = {}
    for item in selected:
        rows = db.execute('SELECT b.cue,b.geometry,o.geometry FROM body_observations b LEFT JOIN observations o '
                          'ON o.id=b.face WHERE b.scan=? ORDER BY b.cue,b.id', (item['scan'],))
        for cue, geometry, face in rows:
            person = json.loads(geometry)
            key = (item['scene'],cue,tuple(person['bounds']))
            records.setdefault(key, []).append((person['keypoints'], face))
    return records


def compare_pose(reference, candidate):
    changed_boxes = association = visibility = 0
    max_error = 0.0
    for key in reference.keys() | candidate.keys():
        a,b = reference.get(key,()), candidate.get(key,())
        if len(a)!=len(b):
            changed_boxes += abs(len(a)-len(b))
        for (p,f),(q,g) in zip(sorted(a,key=lambda v:repr(v)),sorted(b,key=lambda v:repr(v))):
            association += f!=g
            for left,right in zip(p,q):
                if (min(left)>=0) != (min(right)>=0):
                    visibility += 1
                elif min(left)>=0:
                    max_error = max(max_error,math.dist(left,right))
    return dict(changed_box_counts=changed_boxes,changed_face_associations=association,
                changed_visible_keypoints=visibility,max_visible_displacement_pixels=max_error)


def run(config, labels, directory, *, profile_cuda=False, body_batches=None, compare_vitpose=False):
    from stash_ai.recognition.backend import LazyBackend
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.portrait_refresh import LazyAssets
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.run import Services, instance_scope
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.visibility import LazyVisibility
    from stash_ai.runtime.report import RunReport
    services = Services()
    url = os.getenv('STASH_URL','http://localhost:9999')
    results = {}
    with closing(services.client(url=url,api_key=os.getenv('STASH_API_KEY',''),
                                 timeout=config.data['execution']['timeout_seconds'])) as client:
        reference = None
        if compare_vitpose:
            from stash_ai.recognition.config import plain
            import torch
            if (config.data['sapiensid']['pose']['backend'] != 'vitpose_plus_large'
                    or config.data['models']['device'] == 'cpu' or not torch.cuda.is_available()):
                raise ValueError('ViTPose decoder comparison requires the ViTPose++ backend and CUDA.')
            variants = []
            for enabled in (False,True):
                data = plain(config.data)
                data['execution']['vitpose_device_decode'] = enabled
                name = 'vitpose-cuda' if enabled else 'vitpose-reference'
                variants.append((name,replace(config,data=data,database=directory/(name+'.sqlite3'))))
        elif body_batches:
            from stash_ai.recognition.config import plain
            variants = []
            for size in body_batches:
                data = plain(config.data)
                data['sapiensid']['execution']['batch_size'] = size
                name = f'body-batch-{size}'
                variants.append((name, replace(config, data=data, database=directory/(name+'.sqlite3'))))
        else:
            variants = [(name, variant(config, name, directory/(name+'.sqlite3')))
                        for name in ('unmasked','sam_overlap','masked_identity')]
        for name, cfg in variants:
            with ExitStack() as resources:
                state = resources.enter_context(FaceDatabase(cfg.database))
                paths = LazyAssets(cfg,services.assets,RunReport(False,print))
                paths.prepare_identity()
                backend = LazyBackend(cfg,paths,services.backend)
                resources.callback(backend.unload)
                visible = LazyVisibility(cfg,paths,services.visibility)
                resources.callback(visible.close)
                collector = PersonCollector(cfg,paths)
                resources.callback(collector.close)
                selected, cues, gpu_us = [], 0, 0.
                cuda = None
                if (body_batches or compare_vitpose) and cfg.data['models']['device'] != 'cpu':
                    import torch
                    if torch.cuda.is_available():
                        cuda = torch.cuda
                        cuda.synchronize()
                        cuda.reset_peak_memory_stats()
                start = time.monotonic()
                for scene in sorted(labels,key=int):
                    item = client.scene(scene)
                    with closing(services.source(client,item,cfg)) as source:
                        if any(not 0<=int(i)<len(source.cues) for i in labels[scene]):
                            raise ValueError('Label outside VTT cue range in scene '+scene)
                        monitor = nullcontext()
                        if profile_cuda:
                            import torch
                            if not torch.cuda.is_available():
                                raise ValueError('CUDA profiling requires a CUDA device.')
                            monitor = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                         torch.profiler.ProfilerActivity.CUDA])
                        with monitor as profiler:
                            scan,_ = scan_scene(state,instance_scope(url),item,cfg,namespace(cfg),source,
                                                backend.borrow,visibility=visible,on_cue=collector.collect_frame)
                            collector.flush()
                        if profile_cuda:
                            gpu_us += sum(e.time_range.elapsed_us() for e in profiler.events()
                                          if e.device_type==torch.autograd.DeviceType.CUDA)
                        selected.append(dict(scene=scene,scan=scan))
                        cues += len(source.cues)
                if cuda:
                    cuda.synchronize()
                elapsed = time.monotonic()-start
                peak = cuda.max_memory_allocated() if cuda else None
                bodies = state.connection.execute('SELECT COUNT(*) FROM body_observations WHERE embedding IS NOT NULL').fetchone()[0]
                rebuild_selected(state,selected,cfg)
                results[name] = dict(**score(state.connection,selected,labels), cues=cues,
                                     cues_per_second=cues/elapsed, bodies_per_second=bodies/elapsed,
                                     peak_torch_cuda_allocated_bytes=peak,
                                     scan_seconds_per_1000_cues=elapsed*1000/cues,
                                     cuda_activity_seconds_per_1000_cues=gpu_us/(1000*cues) if profile_cuda else None)
                if compare_vitpose:
                    records = pose_records(state.connection,selected)
                    if reference is None:
                        reference = records
                    else:
                        results[name]['reference_comparison'] = compare_pose(reference,records)
    return results


if __name__=='__main__':
    from stash_ai.recognition.config import load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--cuda-profile',action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--compare-vitpose-decode',action='store_true',
                       help='Compare reference/CUDA ViTPose coordinates, visibility and face associations.')
    modes.add_argument('--body-batches', type=int, nargs='+', choices=range(1, 17),
                        help='Compare SapiensID batch sizes using the configured pose backend.')
    args = parser.parse_args()
    config = load(args.config)
    configured = config.data['execution'].get('rebuild_scratch_directory')
    parent = (args.config.resolve().parent/Path(configured).expanduser()) if configured else config.database.parent/'recognition_work'
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='crowded-vtt-',dir=parent) as directory:
        print(json.dumps(run(config,json.loads(args.labels.read_text()),Path(directory),
                             profile_cuda=args.cuda_profile, body_batches=args.body_batches,compare_vitpose=args.compare_vitpose_decode),indent=2))
