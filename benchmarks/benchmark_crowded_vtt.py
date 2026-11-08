"""Compare production pose and identity routes on identical labelled VTT cues."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing, nullcontext
from dataclasses import replace
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time


def score(db, selected, labels, *, reference_config=None):
    from stash_ai.recognition.crowd_labels import torso_owner
    scanned = {s['scene']:s['scan'] for s in selected}
    observations = []
    identity_queries = []
    anchors = defaultdict(set)
    counts = Counter()
    face_owner = 'NULL'
    if reference_config is not None:
        from stash_ai.recognition.reference_roles import strict_sql
        from stash_ai.recognition.pose_policy import anchor_sql
        face_owner = ("CASE WHEN o.rejected IS NULL AND o.embedding IS NOT NULL AND " + strict_sql('o','fm')
                      + ' AND '+anchor_sql('o',reference_config.data['clustering'])+' THEN fm.person END')
    wrist_errors = []
    for scene,cues in labels.items():
        for cue,persons in cues.items():
            if not persons or len({p['truth'] for p in persons}) != len(persons):
                raise ValueError('Each labelled cue needs one torso point per distinct visible performer.')
            detections = defaultdict(list)
            references = defaultdict(list)
            for geometry, person, status, face in db.execute(
                    'SELECT b.geometry,m.person,m.status,'+face_owner+' FROM body_observations b '
                    'LEFT JOIN body_members m ON m.observation=b.id LEFT JOIN observations o ON o.id=b.face '
                    'LEFT JOIN members fm ON fm.observation=o.id WHERE b.scan=? AND b.cue=?',(scanned[scene],int(cue))):
                pose = json.loads(geometry)
                owner = torso_owner(pose,persons)
                if owner is None:
                    counts['unmatched_torsos'] += 1
                    continue
                detections[owner].append(person if status=='accepted' else None)
                references[owner].append(face)
                label = next(p for p in persons if p['truth']==owner)
                for side,point in label.get('wrists',{}).items():
                    joint = pose['keypoints'][9 if side=='left' else 10]
                    counts['wrist_predictions'] += 1
                    if min(joint)<0:
                        counts['missing_wrists'] += 1
                        continue
                    error = math.dist(joint,point)
                    wrist_errors.append(error)
                    foreign = [math.dist(joint,p['wrists'][side]) for p in persons
                               if p['truth']!=owner and side in p.get('wrists',{})]
                    counts['wrists_closer_to_partner'] += bool(foreign and min(foreign)<error)
            visible = {}
            for label in persons:
                truth = label['truth']
                matches = detections[truth]
                counts['duplicate_poses'] += max(0,len(matches)-1)
                counts['missing_poses'] += not matches
                visible[truth] = matches[0] if len(matches)==1 else None
                if not label.get('reference'):
                    identity_queries.append((truth,visible[truth]))
                if label.get('reference'):
                    owner = visible[truth]
                    if reference_config is not None:
                        owners = references[truth]
                        owner = owners[0] if len(owners)==1 else None
                    if owner is None:
                        counts['missing_reference_ids'] += 1
                    else:
                        anchors[truth].add(owner)
            if len(visible)==2 and not any(p.get('reference') for p in persons):
                observations.append(visible)
    for truth,owner in identity_queries:
        if len(anchors[truth])!=1:
            counts['unanchored_queries'] += 1
            continue
        counts['anchored_queries'] += 1
        foreign = any(owner in refs for label,refs in anchors.items() if label!=truth)
        counts['own_id_queries'] += owner in anchors[truth] and not foreign
        counts['foreign_id_queries'] += foreign
        counts['unassigned_queries'] += owner is None
    for visible in observations:
        a,b = visible
        own,partner = visible[a],visible[b]
        counts['labelled_pairs'] += 1
        counts['partner_false_joins'] += bool(own is not None and (own==partner or own in anchors[b])
                                               or partner is not None and partner in anchors[a])
        counts['both_correct'] += bool(own is not None and partner is not None and own!=partner
                                       and anchors[a]=={own} and anchors[b]=={partner})
    return dict(dict.fromkeys(('labelled_pairs','both_correct','partner_false_joins',
                 'missing_poses','duplicate_poses','unmatched_torsos'),0),**counts,
                 wrist_mean_error_px=sum(wrist_errors)/len(wrist_errors) if wrist_errors else None)


def variant(config, name, database):
    from stash_ai.recognition.config import plain
    data = plain(config.data)
    pose = data['sapiensid']['pose']
    if pose['backend'] != 'pmpose' or not data['sapiensid']['identities']['enabled']:
        raise ValueError('Use a PMPose configuration with body identities enabled.')
    pose['refinement'].update(enabled=name!='unmasked',mode='overlap',masked_identity_crop=name=='masked_identity')
    data['execution'].update(max_cues=0,media_cache_bytes=0)
    return replace(config,data=data,database=database)


def contact_variants(config,directory):
    from stash_ai.recognition.config import plain
    s = config.data['sapiensid']
    if not (s['enabled'] and s['collection']['enabled'] and s['identities']['enabled']):
        raise ValueError('Contact comparison requires body collection and local identities enabled.')
    for name,backend in [('rfdetr-vitpose','vitpose_plus_large'),('rfdetr-masked-pmpose','pmpose'),
                         ('rfdetr-vitpose-rtmo','vitpose_plus_large'),('rfdetr-vitpose-rtmo-torso','vitpose_plus_large'),
                         ('rfdetr-vitpose-rtmo-missing','vitpose_plus_large')]:
        data = plain(config.data)
        pose = data['sapiensid']['pose']
        pose.update(detector='rfdetr_xl',backend=backend)
        pose['refinement'].update(enabled=backend=='pmpose',mode='overlap',masked_identity_crop=backend=='pmpose')
        data['execution'].update(max_cues=0,media_cache_bytes=0)
        yield name,replace(config,data=data,database=directory/(name+'.sqlite3'))


class CheckedSource:
    """Fail the comparison if any variant sees a different VTT pixel or timing."""
    def __init__(self,source,scene,catalogue,first):
        self.source,self.scene,self.catalogue,self.first = source,scene,catalogue,first

    def __getattr__(self,name):
        return getattr(self.source,name)

    def frames(self,start=0):
        return self._checked(self.source.frames(start))

    def frames_indices(self,indices):
        if hasattr(self.source,'frames_indices'):
            return self._checked(self.source.frames_indices(indices))
        return self.frames(min(indices))

    def _checked(self,stream):
        with closing(stream) as frames:
            for index,cue,image in frames:
                try:
                    with image.convert('RGB') as rgb:
                        stamp = (rgb.size,cue.start,cue.end,hashlib.sha256(rgb.tobytes()).digest())
                    address = self.scene,index
                    if self.first:
                        self.catalogue.setdefault(address,stamp)
                    if self.catalogue.get(address)!=stamp:
                        raise ValueError(f'VTT pixels or timing changed at scene {self.scene}, cue {index}.')
                except BaseException:
                    image.close()
                    raise
                yield index,cue,image


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


def run(config, labels, directory, *, profile_cuda=False, body_batches=None, compare_vitpose=False, compare_contact=False, kpr_config=None, kpr_weights=None, kpr_python=None):
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
    catalogue = {}
    if (kpr_config is None)!=(kpr_weights is None) or kpr_config and not compare_contact or kpr_python and not kpr_config:
        raise ValueError('Use --kpr-config and --kpr-weights together with --compare-contact.')
    if compare_contact:
        from stash_ai.recognition.crowd_labels import validate
        validate(labels,labels)
        if not labels or not all(labels.values()):
            raise ValueError('Contact comparison requires labelled VTT cues.')
        for cues in labels.values():
            for persons in cues.values():
                for person in persons:
                    wrists = person.get('wrists',{})
                    if not isinstance(wrists,dict) or any(
                        side not in ('left','right') or not isinstance(point,list) or len(point)!=2 or
                        any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in point)
                        for side,point in wrists.items()):
                        raise ValueError('Optional wrists must map left/right to [x,y] VTT coordinates.')
    with closing(services.client(url=url,api_key=os.getenv('STASH_API_KEY',''),
                                 timeout=config.data['execution']['timeout_seconds'])) as client:
        reference = None
        if compare_contact:
            variants = contact_variants(config,directory)
        elif compare_vitpose:
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
                from stash_ai.recognition.work_locations import invocation
                resources.enter_context(invocation(cfg))
                state = resources.enter_context(FaceDatabase(cfg.database))
                paths = LazyAssets(cfg,services.assets,RunReport(False,print))
                paths.prepare_identity()
                backend = LazyBackend(cfg,paths,services.backend)
                resources.callback(backend.unload)
                visible = LazyVisibility(cfg,paths,services.visibility)
                resources.callback(visible.close)
                stats = Counter(retry_cues=0,replacement_cues=0)
                factory = None
                if name.startswith('rfdetr-vitpose-rtmo'):
                    from benchmarks.rtmo_contact import ContactModel
                    def factory(settings,assets):
                        return ContactModel(settings,assets,stats,torso_nms=name.endswith(('-torso','-missing')),
                                            missing_only=name.endswith('-missing'))
                collector = PersonCollector(cfg,paths,factory=factory)
                resources.callback(collector.close)
                selected, cues, gpu_us = [], 0, 0.
                cuda = None
                if (body_batches or compare_vitpose or compare_contact) and cfg.data['models']['device'] != 'cpu':
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
                        if compare_contact:
                            source = CheckedSource(source,scene,catalogue,name=='rfdetr-vitpose')
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
                            if compare_contact:
                                collector.enrich(state.connection,scan,source)
                        if profile_cuda:
                            gpu_us += sum(e.time_range.elapsed_us() for e in profiler.events()
                                          if e.device_type==torch.autograd.DeviceType.CUDA)
                        selected.append(dict(scene=scene,scan=scan))
                        if compare_contact and sum(s==scene for s,_ in catalogue)!=len(source.cues):
                            raise ValueError(f'Incomplete or changed VTT input for scene {scene}.')
                        cues += len(source.cues)
                if cuda:
                    cuda.synchronize()
                elapsed = time.monotonic()-start
                peak = cuda.max_memory_allocated() if cuda else None
                bodies = state.connection.execute('SELECT COUNT(*) FROM body_observations WHERE embedding IS NOT NULL').fetchone()[0]
                rebuild_selected(state,selected,cfg)
                results[name] = dict(**score(state.connection,selected,labels,reference_config=cfg if compare_contact else None), cues=cues,
                                     cues_per_second=cues/elapsed, bodies_per_second=bodies/elapsed,
                                     peak_torch_cuda_allocated_bytes=peak,
                                     scan_seconds_per_1000_cues=elapsed*1000/cues,
                                     cuda_activity_seconds_per_1000_cues=gpu_us/(1000*cues) if profile_cuda else None)
                if compare_contact:
                    retry_counts = dict(attempts=0,recovered=0)
                    for item in selected:
                        entry = state.connection.execute('SELECT value FROM metadata WHERE key=?',
                              ('future-crowd:'+item['scan'],)).fetchone()
                        if entry:
                            retry_counts['attempts'] += len(json.loads(entry[0]).get('attempted',{}))
                        retry_counts['recovered'] += state.connection.execute(
                            "SELECT COUNT(DISTINCT cue) FROM body_observations WHERE scan=? AND "
                            "json_extract(geometry,'$.future_torso_retry')=1",(item['scan'],)).fetchone()[0]
                    results[name]['temporal_retry'] = dict(retry_counts,attempts_per_1000_cues=1000*retry_counts['attempts']/cues)
                    results[name]['contact_retry'] = dict(stats)
                    results[name]['verified_vtt_cues'] = len(catalogue)
                    results[name]['scored_cues'] = sum(map(len,labels.values()))
                    base = results['rfdetr-vitpose']
                    results[name]['delta_from_vitpose'] = {
                        key:results[name][key]-base[key] if base[key] is not None else None
                        for key in ('both_correct','partner_false_joins','missing_poses','duplicate_poses',
                                    'scan_seconds_per_1000_cues','cuda_activity_seconds_per_1000_cues')}
                if name=='rfdetr-vitpose' and kpr_config:
                    from contextlib import contextmanager
                    from benchmarks import kpr_contact
                    # Release production models before loading the optional encoder.
                    collector.close()
                    backend.unload()
                    visible.close()
                    @contextmanager
                    def source_for(scene):
                        with closing(services.source(client,client.scene(scene),cfg)) as source:
                            yield CheckedSource(source,scene,catalogue,False)
                    extractor = kpr_contact.Extractor(kpr_config,kpr_weights,directory,cfg.data['models']['device'],kpr_python)
                    results[name]['paired_reid'] = kpr_contact.run(state.connection,selected,labels,source_for,extractor)
                    del extractor
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
    parser.add_argument('--kpr-python',type=Path,help='Python interpreter of the upstream KPR environment.')
    parser.add_argument('--kpr-config',type=Path,help='Upstream KPR evaluation YAML for paired contact retrieval.')
    parser.add_argument('--kpr-weights',type=Path,help='Trained KPR checkpoint for the selected YAML.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--compare-contact',action='store_true',
                       help='Compare RF-DETR+ViTPose++, masked PMPose and selective full-frame RTMO on the same VTT.')
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
                             profile_cuda=args.cuda_profile, body_batches=args.body_batches,compare_vitpose=args.compare_vitpose_decode,compare_contact=args.compare_contact,kpr_config=args.kpr_config,kpr_weights=args.kpr_weights,kpr_python=args.kpr_python),indent=2))
