"""Resumable supplementary inference; one invocation owns all GPU models."""
from io import BytesIO
import json

from .models import checked_detection
from .person_signal import contract, save
from .progress import tracking
from stash_ai.runtime.cleanup import closing, cleanup


class PersonCollector:
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory
        self.model = None
        self.pipeline = None
        self.deferred = None

    def close(self):
        resources = [('stop person preparation', self.pipeline), ('close person model', self.model)]
        self.pipeline = self.model = None
        cleanup([(name, resource.close) for name, resource in resources if resource is not None])

    def enrich(self, db, scan, source):
        if not self.config.data['sapiensid']['enabled']:
            return
        expected=contract(self.config)
        # Include every face in association, even if its identity evidence was
        # rejected. Otherwise a second face could disappear from ambiguity checks.
        missing={r[0] for r in db.execute('SELECT DISTINCT o.cue FROM observations o '
            'LEFT JOIN person_evidence p ON p.observation=o.id AND p.contract=? '
            'WHERE o.scan=? AND o.rejected IS NULL AND p.observation IS NULL',(expected,scan))}
        if not missing:
            return
        self._load()
        frames=(source.frames_indices(sorted(missing)) if hasattr(source,'frames_indices') else source.frames(min(missing)))
        with closing(frames,name='close person source frames') as stream:
            with tracking(stream,'Collect SapiensID evidence',total=len(missing),unit='cue') as frames:
                for index,cue,image in frames:
                    with image:
                        if index in missing:self.collect_frame(db,scan,index,image)
        self.flush()

    def _load(self):
        if self.model is None:
            if self.factory is None:
                from .sapiensid_backend import SapiensIDFaces
                self.factory=lambda config,paths: SapiensIDFaces(config,paths,load_detector=False)
            self.model=self.factory(self.config,self.paths)

    def abort(self):
        self.deferred=None
        if self.pipeline is not None:
            pipeline,self.pipeline=self.pipeline,None
            cleanup([("abort person preparation", pipeline.close)])

    def flush(self):
        try:
            if self.pipeline is not None:self.pipeline.flush()
        except BaseException:
            self.abort();raise

    def collect_frame(self,db,scan,index,image):
        try:return self._collect_frame(db,scan,index,image)
        except BaseException:
            self.abort();raise

    def collect_records(self,db,scan,index,image,rows):
        """Prepare fresh faces without rereading or committing them per cue."""
        if self.deferred is None:self.deferred=[]
        try:return self._collect_frame(db,scan,index,image,rows=rows)
        except BaseException:
            self.abort();raise

    def take_records(self):
        self.flush()
        records,self.deferred=self.deferred or [],None
        return records

    def _write(self,db,rows):
        if self.deferred is not None:self.deferred.extend(rows)
        else:
            with db:
                for identifier,metadata,vector,crop in rows:
                    save(db,identifier,contract(self.config),metadata,vector,crop)

    def _collect_frame(self,db,scan,index,image,*,rows=None):
        if not self.config.data['sapiensid']['enabled']:return
        self._load();expected=contract(self.config)
        fresh=rows is not None
        if rows is None:rows=list(db.execute('SELECT * FROM observations WHERE scan=? AND cue=? ORDER BY id',(scan,index)))
        faces=[checked_detection(json.loads(r['geometry'])) for r in rows]
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
                if self.config.data['sapiensid']['storage']['retain_person_crops']:
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
    from PIL import Image
    from .alignment_audit import _png
    from .person_audit import export as export_annotations
    details=detector.observation_quality(image,face)
    if details['identity_rejection'] is not None:
        return details
    batch,points,mask=detector.audit_input(image,face)
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


def retained_report(db,observation,config,directory,name):
    from .person_signal import read
    from .identity_report import _image
    expected=contract(config)
    row=read(db,observation,expected)
    if row is None:
        return dict(status='missing_current_evidence',contract=expected)
    return dict(status=row['status'],contract=expected,metadata=json.loads(row['metadata']),
                image=_image(row['crop'],directory/('person-'+name)))
