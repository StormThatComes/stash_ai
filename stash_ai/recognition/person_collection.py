"""Resumable supplementary inference; one invocation owns all GPU models."""
from io import BytesIO
import json

from .models import checked_detection
from .person_signal import contract, save
from .progress import work
from stash_ai.runtime.cleanup import closing, cleanup


class PersonCollector:
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory
        self.model = None
        self.pipeline = None
        self.deferred = None
        self.deferred_bytes = 0
        self.pose_frames = []
        self.pose_bytes = 0
        self.resume_rows=None
        self.resume_bytes=self.resume_cues=0

    def _close_pose_frames(self):
        frames, self.pose_frames = self.pose_frames, []
        self.pose_bytes = 0
        cleanup([lease.close for *_, lease in frames])

    def close(self):
        self.deferred = None
        self.deferred_bytes = 0
        self._close_pose_frames()
        resources = [('stop person preparation', self.pipeline), ('close person model', self.model)]
        self.pipeline = self.model = None
        cleanup([(name, resource.close) for name, resource in resources if resource is not None])

    def missing_cues(self, db, scan, total):
        """Find sparse durable gaps for both face projections and independent body cues."""
        if not self.config.data['sapiensid']['enabled']:
            return set()
        expected=contract(self.config)
        # Include every face in association, even if its identity evidence was
        # rejected. Otherwise a second face could disappear from ambiguity checks.
        missing={r[0] for r in db.execute('SELECT DISTINCT o.cue FROM observations o '
            'LEFT JOIN person_evidence p ON p.observation=o.id AND p.contract=? '
            'WHERE o.scan=? AND o.rejected IS NULL AND p.observation IS NULL',(expected,scan))}
        from .body_observations import enabled,contract as body_contract
        if enabled(self.config):
            complete={r[0] for r in db.execute('SELECT cue FROM body_cues WHERE scan=? AND contract=?',(scan,body_contract(self.config)))}
            missing.update(set(range(total))-complete)
        return missing

    def enrich(self, db, scan, source):
        if not self.config.data['sapiensid']['enabled']:
            return 0
        from .body_observations import enabled,repair_projections
        if enabled(self.config):
            with db:repair_projections(db,self.config,scan)
        missing = self.missing_cues(db, scan, len(source.cues) if enabled(self.config) else 0)
        if not missing:
            from .person_crop_backfill import backfill
            backfill(db,scan,source,self.config)
            return 0
        self._load()
        frames=(source.frames_indices(sorted(missing)) if hasattr(source,'frames_indices') else source.frames(min(missing)))
        self.resume_rows=[]
        # Read only narrow geometry columns; cap each ordered lookup to 64 cues.
        blocks={cue:cue//64 for cue in missing}
        loaded=None
        try:
            with closing(frames,name='close person source frames') as stream:
                with work('Collect SapiensID evidence', len(missing), unit='cue') as progress:
                    for index,cue,image in stream:
                        with image:
                            if index in missing:
                                if loaded!=blocks[index]:
                                    loaded=blocks[index]
                                    cues=[i for i in range(loaded*64,(loaded+1)*64) if i in missing]
                                    lookup={i:[] for i in cues}
                                    for row in db.execute('SELECT cue,id,geometry,rejected FROM observations WHERE scan=? '
                                            'AND cue IN (SELECT value FROM json_each(?)) ORDER BY cue,id',(scan,json.dumps(cues))):
                                        lookup[row['cue']].append(row)
                                rows=lookup[index]
                                self.collect_records(db,scan,index,image,rows,seconds=cue.start,defer=False,fresh=False)
                                progress.advance()
            self.flush()
        finally:
            # Only sealed, completely inferred cues enter this buffer. Preserve
            # them even when a later inference fails; never publish partial cues.
            try:self._flush_resume(db)
            finally:self.resume_rows=None
        from .person_crop_backfill import backfill
        backfill(db,scan,source,self.config)
        return len(missing)

    def _load(self):
        if self.model is None:
            if self.factory is None:
                from .sapiensid_backend import SapiensIDFaces
                self.factory=lambda config,paths: SapiensIDFaces(config,paths,load_detector=False)
            self.model=self.factory(self.config,self.paths)

    def abort(self):
        self.deferred=None
        self.deferred_bytes=0
        # Failed inference may leave CUDA/session state unusable. A later scene
        # must acquire a fresh owner, even if closing this owner also fails.
        resources=[('abort person preparation',self.pipeline),('close failed person model',self.model)]
        self.pipeline=self.model=None
        cleanup([('release queued pose frames',self._close_pose_frames),
                 *[(name,resource.close) for name,resource in resources if resource is not None]])

    def flush(self):
        try:
            self._flush_pose()
            if self.pipeline is not None:self.pipeline.flush()
        except BaseException:
            self.abort();raise

    def collect_frame(self,db,scan,index,image,*,seconds=0.):
        try:return self._collect_frame(db,scan,index,image,seconds=seconds)
        except BaseException:
            self.abort();raise

    def collect_records(self,db,scan,index,image,rows,*,seconds=0.,frame_owner=None,defer=True,fresh=True):
        """Prepare fresh faces without rereading or committing them per cue."""
        if defer and self.deferred is None:self.deferred=[]
        try:
            if self.config.data['sapiensid']['enabled']:
                self._load()
                if callable(getattr(self.model, 'detect_many', None)):
                    size = image.width*image.height*3+4096
                    limit = self.config.data['execution']['scan_queue_bytes']//4
                    if self.pose_frames and self.pose_bytes+size > limit:self._flush_pose()
                    if size <= limit:
                        from .frame_ownership import Lease
                        lease = frame_owner.retain(image) if frame_owner is not None else Lease(image.copy())
                        self.pose_frames.append((db, scan, index, lease.image, rows, seconds, fresh, lease))
                        self.pose_bytes += size
                        if len(self.pose_frames) >= self.config.data['sapiensid']['pose']['batch_size']:
                            self._flush_pose()
                        return
            return self._collect_frame(db,scan,index,image,rows=rows,fresh=fresh,seconds=seconds)
        except BaseException:
            self.abort();raise

    def _flush_pose(self):
        if not self.pose_frames:return
        try:
            frames = [(image, [checked_detection(json.loads(r['geometry'])) for r in rows])
                      for _, _, _, image, rows, _, _, _ in self.pose_frames]
            self.model.detect_many(frames)
            for db, scan, index, image, rows, seconds, fresh, _ in self.pose_frames:
                self._collect_frame(db, scan, index, image, rows=rows,fresh=fresh, seconds=seconds)
        finally:
            self._close_pose_frames()

    def pending_bytes(self):
        # Completed inference waits for the same atomic face/body checkpoint.
        return self.pose_bytes+self.deferred_bytes

    def take_records(self):
        self.flush()
        records,self.deferred=self.deferred or [],None
        self.deferred_bytes=0
        return records

    def _flush_resume(self,db):
        if self.resume_rows:
            with db:self.write_records(db,self.resume_rows)
            self.resume_rows.clear()
        self.resume_bytes=self.resume_cues=0

    def _write(self,db,rows):
        if self.deferred is not None:
            for row in rows:
                _,metadata,vector,crop=row
                self.deferred_bytes+=(len(crop or b'')+getattr(vector,'nbytes',len(vector) if isinstance(vector,bytes) else 0)
                                      +len(str(metadata))+256)
                self.deferred.append(row)
        elif self.resume_rows is not None:
            self.resume_rows.extend(rows);self.resume_cues+=1
            self.resume_bytes+=sum(len(crop or b'')+getattr(vector,'nbytes',0)+len(str(metadata))+256
                                   for _,metadata,vector,crop in rows)
            if (self.resume_cues>=self.config.data['execution']['checkpoint_cues'] or
                    self.resume_bytes>=self.config.data['execution']['scan_queue_bytes']//4):
                self._flush_resume(db)
        else:
            with db:
                self.write_records(db,rows)

    def write_records(self,db,rows):
        from .body_observations import save_records
        save_records(db,self.config,rows)

    def _collect_frame(self,db,scan,index,image,*,rows=None,fresh=False,seconds=0.):
        if not self.config.data['sapiensid']['enabled']:return
        from .body_observations import enabled,contract as body_contract,repair_projections
        if enabled(self.config) and not fresh and db.execute(
                'SELECT 1 FROM body_cues WHERE scan=? AND contract=? AND cue=?',
                (scan,body_contract(self.config),index)).fetchone():
            with db:repair_projections(db,self.config,scan,cue=index)
            return
        self._load();expected=contract(self.config)
        if rows is None:rows=list(db.execute('SELECT id,geometry,rejected FROM observations WHERE scan=? AND cue=? ORDER BY id',(scan,index)))
        faces=[checked_detection(json.loads(r['geometry'])) for r in rows]
        from .body_observations import enabled,contract as body_contract
        if enabled(self.config):
            self.model.detect(image,faces=faces)
            from .body_collection import collect
            if self.pipeline is None:
                from .person_pipeline import PreparedCollector
                self.pipeline=PreparedCollector(self.config,self.model,writer=self._write)
            collect(self.pipeline,db,self.model,self.config,scan,index,seconds,image,rows,faces)
            return
        self.model.detect(image,faces=faces)
        if callable(getattr(self.model,'person_input',None)) and callable(getattr(self.model,'embed_inputs',None)):
            if self.pipeline is None:
                from .person_pipeline import PreparedCollector
                self.pipeline=PreparedCollector(self.config,self.model,writer=self._write)
            self.pipeline.collect(db,image,rows,faces,fresh=fresh)
            return
        pending=[];eligible=[]
        for row,face in zip(rows,faces):
            if row['rejected'] is not None:
                continue
            if not fresh and db.execute('SELECT 1 FROM person_evidence WHERE observation=? AND contract=?',
                          (row['id'],expected)).fetchone():
                continue
            metadata=self.model.observation_quality(image,face)
            vector=crop=None
            if metadata['identity_rejection'] is None:
                eligible.append((len(pending),face))
                if self.config.data['sapiensid']['storage']['retain_person_crops'] and self.config.data['storage']['crop_policy']=='all':
                    with image.crop(metadata['person']['bounds']) as person:
                        output=BytesIO();person.save(output,format='PNG')
                        if output.tell()<=self.config.data['storage']['max_crop_bytes']:
                            crop=output.getvalue()
                        else:
                            metadata=dict(metadata,crop_status='byte_budget_exceeded')
            pending.append([row['id'],metadata,vector,crop])
        size=self.config.data['sapiensid']['execution']['batch_size']
        for start in range(0,len(eligible),size):
            batch=eligible[start:start+size]
            batch_faces=[face for _,face in batch]
            embed_batch=getattr(self.model,'embed_faces',None)
            vectors=(embed_batch(image,batch_faces) if embed_batch else
                     [self.model.embed(image,face) for face in batch_faces])
            if len(vectors)!=len(batch):
                raise ValueError('SapiensID batch returned an invalid feature count.')
            for (index,_),vector in zip(batch,vectors):
                pending[index][2]=vector
        # Nothing from a partially failed cue becomes durable.
        self._write(db,pending)


def audit_input(detector,image,face,path):
    """Export exact pre-normalization RGB input and measured model geometry."""
    details=detector.observation_quality(image,face)
    if details['identity_rejection'] is not None:return details
    return _audit_prepared(image,face,details,detector.audit_input(image,face),path)


def audit_person(detector,image,person,path,*,face=None,details=None):
    from .sapiensid_v1 import prepare_person_input
    bounds,points=detector.person_geometry(image,person,face)
    from .person_geometry import encoder_crop
    mask = person.get('pose_refinement', {}).get('encoder_mask')
    with encoder_crop(image,bounds,mask) as crop:prepared=prepare_person_input(crop,points)
    return _audit_prepared(image,face,dict(details or {},person=person),prepared,path)


def _audit_prepared(image,face,details,prepared,path):
    from PIL import Image
    from .alignment_audit import _png
    from .person_audit import export as export_annotations
    batch,points,mask=prepared
    pixels=((batch[0].detach().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).round().astype('uint8')
    target=path.with_name(path.name+'-person-input.png')
    normalized_points=points[0].detach().cpu().tolist()
    with Image.fromarray(pixels) as png:
        _png(png,target)
        annotations=export_annotations(image,png,face,details['person'],normalized_points,path)
    mask_path=path.with_name(path.name+'-person-mask.png')
    with Image.fromarray(mask[0].detach().cpu().numpy().astype('uint8')*255) as png:_png(png,mask_path)
    return dict(details,input_file=target.name,mask_file=mask_path.name,
                normalized_keypoints=normalized_points,**annotations)


def retained_report(db,observation,config,directory,name,*,geometry,crops=None):
    from .person_signal import read
    from .identity_report import _image
    from .retained_annotations import export as annotate
    expected=contract(config)
    row=read(db,observation,expected)
    if row is None:
        return dict(status='missing_current_evidence',contract=expected)
    metadata=json.loads(row['metadata'])
    if row['crop'] is None and crops is not None:
        from .body_observations import contract as body_contract
        source=db.execute('SELECT id FROM body_observations WHERE face=? AND contract=? ORDER BY id LIMIT 1',(observation,body_contract(config))).fetchone()
        if source:
            row=dict(row);row['crop']=crops.body(source[0])
    from .person_anchor import pixel_gate
    def annotations(rgb,path):
        person=metadata.get('person')
        if not isinstance(person,dict):
            return dict(status='missing_person_geometry',file=None)
        return annotate(rgb,path,geometry,person.get('bounds'),pose=True,keypoints=person.get('keypoints'))
    return dict(status=row['status'],contract=expected,metadata=metadata,
                anchor_pixel_gate=pixel_gate(metadata,config.data['sapiensid']['identities']),
                image=_image(row['crop'],directory/('person-'+name),annotate=annotations))
