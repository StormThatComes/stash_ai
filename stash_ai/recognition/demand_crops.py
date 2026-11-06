"""Reconstruct requested image artifacts from verified VTT, never from inference."""
from io import BytesIO
import hashlib
import json

from .cold_payloads import get
from .media import SpriteInput,verify_portrait_cue


class Crops:
    def __init__(self,db,config,client=None,factory=SpriteInput):
        self.db,self.config,self.client,self.factory=db,config,client,factory
        self.scene=self.source=None

    def close(self):
        if self.source is not None:self.source.close()
        self.source=self.scene=None

    def _image(self,row,*,face):
        if self.client is None:return None
        from .run import instance_scope
        if instance_scope(self.client.url)!=row['scope']:
            raise ValueError('Crop reconstruction Stash instance does not match retained evidence.')
        if self.scene!=row['scene']:
            self.close()
            self.source=self.factory(self.client,self.client.scene(row['scene']),self.config).__enter__()
            self.scene=row['scene']
        if face:verify_portrait_cue(self.source,row)
        elif (self.source.fingerprint!=row['input_sha'] or not 0<=row['cue']<len(self.source.cues)
              or self.source.cues[row['cue']].start!=row['seconds']):
            raise ValueError('VTT source changed after body recognition.')
        return self.source.thumbnail(row['cue'])

    def face(self,observation):
        row=self.db.execute('SELECT o.*,s.scene,s.scope,s.input_sha FROM observations o '
            'JOIN scans s ON s.id=o.scan WHERE o.id=?',(observation,)).fetchone()
        if row is None:return None
        blob=get(self.db,row['crop'])
        if blob is not None:return blob
        image=self._image(row,face=True)
        if image is None:return None
        from .quality import crop_bytes
        from .models import checked_detection
        quality=json.loads(row['quality']);details={}
        with image:
            blob,bounds=crop_bytes(image,checked_detection(json.loads(row['geometry'])),
                limit=self.config.data['storage']['max_crop_bytes'],details=details)
        if quality.get('evidence_pixel_sha256')!=details['evidence_pixel_sha256']:
            raise ValueError('Reconstructed face pixels differ from retained evidence.')
        if quality.get('crop_bounds',bounds)!=bounds:raise ValueError('Reconstructed face bounds changed.')
        return blob

    def body(self,observation):
        row=self.db.execute('SELECT b.*,s.scene,s.scope,s.input_sha FROM body_observations b '
            'JOIN scans s ON s.id=b.scan WHERE b.id=?',(observation,)).fetchone()
        if row is None:return None
        blob=get(self.db,row['crop'])
        if blob is not None:return blob
        image=self._image(row,face=False)
        if image is None:return None
        geometry=json.loads(row['geometry']);quality=json.loads(row['quality']).get('quality',{})
        with image,image.crop(geometry['bounds']) as crop,crop.convert('RGB') as rgb:
            digest=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
            if quality.get('pixel_digest')!=digest:
                raise ValueError('Reconstructed person pixels differ from retained evidence.')
            output=BytesIO();rgb.save(output,format='PNG')
            return output.getvalue() if output.tell()<=self.config.data['storage']['max_crop_bytes'] else None
