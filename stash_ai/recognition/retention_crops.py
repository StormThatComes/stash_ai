"""Rehydrate archived face pixels from verified VTT input, without inference."""
import json

from stash_ai.runtime.cleanup import closing
from .db_retention import FACE_CROPS
from .models import checked_detection
from .quality import crop_bytes


def restore(db, scan, source, config):
    if config.data['storage']['crop_policy']=='selected':return
    from .cold_payloads import put
    marker = FACE_CROPS + scan
    if (not db.execute('SELECT 1 FROM metadata WHERE key=?', (marker,)).fetchone() and
            not db.execute("SELECT 1 FROM observations WHERE scan=? AND crop IS NULL AND json_extract(quality,'$.crop_status')='on_demand' LIMIT 1",(scan,)).fetchone()):
        return
    cues = [r[0] for r in db.execute('SELECT DISTINCT cue FROM observations WHERE scan=? '
                                   'AND crop IS NULL AND rejected IS NULL ORDER BY cue', (scan,))]
    if cues:
        wanted = set(cues)
        stream = source.frames_indices(cues) if hasattr(source, 'frames_indices') else source.frames(min(cues))
        with closing(stream, name='close retained face crop input') as frames:
            for index, _, image in frames:
                with image:
                    if index not in wanted:
                        continue
                    with db:
                        for row in db.execute('SELECT id,geometry,quality FROM observations WHERE scan=? AND cue=? '
                                              'AND crop IS NULL AND rejected IS NULL', (scan, index)):
                            details={}
                            crop, bounds = crop_bytes(image, checked_detection(json.loads(row['geometry'])),
                                                     limit=config.data['storage']['max_crop_bytes'],details=details)
                            previous = json.loads(row['quality']).get('crop_bounds')
                            if previous is not None and list(previous) != bounds:
                                raise ValueError('Archived face crop geometry changed; refusing mismatched pixels.')
                            db.execute('UPDATE observations SET crop=? WHERE id=?', (put(db,crop), row['id']))
                            if json.loads(row['quality']).get('crop_status')=='on_demand':
                                from .database import encoded
                                encoding={k:v for k,v in details.items() if not k.startswith('evidence_')}
                                db.execute("UPDATE observations SET quality=json_set(quality,'$.crop_encoding',json(?),'$.crop_status','retained') WHERE id=?",(encoded(encoding),row['id']))
                    wanted.remove(index)
        if wanted:
            raise ValueError('VTT input ended before archived face crops were restored.')
    with db:
        db.execute('DELETE FROM metadata WHERE key=?', (marker,))
