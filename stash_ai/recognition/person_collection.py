"""Resumable supplementary inference; one invocation owns all GPU models."""
from io import BytesIO
import json

from .models import checked_detection
from .person_signal import contract, save
from .progress import tracking
from stash_ai.runtime.cleanup import closing


class PersonCollector:
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory
        self.model = None

    def close(self):
        if self.model is not None:
            model,self.model=self.model,None
            model.close()

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
        if self.model is None:
            if self.factory is None:
                from .sapiensid_backend import SapiensIDFaces
                self.factory=SapiensIDFaces
            self.model=self.factory(self.config,self.paths)
        with closing(source.frames(min(missing)),name='close person source frames') as stream:
            with tracking(stream,'Collect SapiensID evidence',total=len(source.cues)-min(missing),unit='cue') as frames:
                for index,cue,image in frames:
                    with image:
                        if index not in missing:
                            continue
                        rows=list(db.execute('SELECT * FROM observations WHERE scan=? AND cue=? ORDER BY id',(scan,index)))
                        faces=[checked_detection(json.loads(r['geometry'])) for r in rows]
                        self.model.detect(image,faces=faces)
                        pending=[]
                        for row,face in zip(rows,faces):
                            if row['rejected'] is not None:
                                continue
                            if db.execute('SELECT 1 FROM person_evidence WHERE observation=? AND contract=?',
                                          (row['id'],expected)).fetchone():
                                continue
                            metadata=self.model.observation_quality(image,face)
                            vector=crop=None
                            if metadata['identity_rejection'] is None:
                                vector=self.model.embed(image,face)
                                if self.config.data['sapiensid']['storage']['retain_person_crops']:
                                    with image.crop(metadata['person']['bounds']) as person:
                                        output=BytesIO();person.save(output,format='PNG');crop=output.getvalue()
                            pending.append((row['id'],metadata,vector,crop))
                        # Nothing from a partially failed cue becomes durable.
                        with db:
                            for identifier,metadata,vector,crop in pending:
                                save(db,identifier,expected,metadata,vector,crop)


def audit_input(detector,image,face,path):
    """Export exact pre-normalization RGB input and measured model geometry."""
    from PIL import Image
    details=detector.observation_quality(image,face)
    if details['identity_rejection'] is not None:
        return details
    batch,points,mask=detector.audit_input(image,face)
    pixels=((batch[0].detach().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).round().astype('uint8')
    target=path.with_name(path.name+'-person-input.png')
    with Image.fromarray(pixels) as png:png.save(target)
    mask_path=path.with_name(path.name+'-person-mask.png')
    with Image.fromarray(mask[0].detach().cpu().numpy().astype('uint8')*255) as png:png.save(mask_path)
    return dict(details,input_file=target.name,mask_file=mask_path.name,
                normalized_keypoints=points[0].detach().cpu().tolist())


def retained_report(db,observation,config,directory,name):
    from .person_signal import read
    from .identity_report import _image
    expected=contract(config)
    row=read(db,observation,expected)
    if row is None:
        return dict(status='missing_current_evidence',contract=expected)
    return dict(status=row['status'],contract=expected,metadata=json.loads(row['metadata']),
                image=_image(row['crop'],directory/('person-'+name)))
