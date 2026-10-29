"""Recover optional retained crops from measured geometry without model inference."""
from io import BytesIO
import json
import math

from .body_observations import contract as body_contract,enabled as bodies_enabled
from .person_signal import contract
from stash_ai.runtime.cleanup import closing


def rows(db,scan,config,cue=None):
    limit=config.data['storage']['max_crop_bytes']
    condition='' if cue is None else ' AND o.cue=?'
    args=(scan,contract(config),limit)+(() if cue is None else (cue,))
    for row in db.execute("SELECT o.cue,p.observation AS id,p.metadata AS metadata FROM person_evidence p "
            "JOIN observations o ON o.id=p.observation WHERE o.scan=? AND p.contract=? "
            "AND p.status='ready' AND p.crop IS NULL "
            "AND COALESCE(json_extract(p.metadata,'$.crop_attempt_limit'),0)<?"+condition,args):
        yield 'person_evidence',dict(row)
    if bodies_enabled(config):
        condition='' if cue is None else ' AND cue=?'
        args=(scan,body_contract(config),limit)+(() if cue is None else (cue,))
        for row in db.execute('SELECT cue,id,quality AS metadata FROM body_observations '
                'WHERE scan=? AND contract=? AND embedding IS NOT NULL AND crop IS NULL '
                "AND COALESCE(json_extract(quality,'$.crop_attempt_limit'),0)<?"+condition,args):
            yield 'body_observations',dict(row)


def backfill(db,scan,source,config):
    if not config.data['sapiensid']['storage']['retain_person_crops']:return
    cues=sorted({row['cue'] for _,row in rows(db,scan,config)})
    if not cues:return
    frames=source.frames_indices(cues) if hasattr(source,'frames_indices') else source.frames(min(cues))
    limit=config.data['storage']['max_crop_bytes'];wanted=set(cues)
    with closing(frames,name='close person crop backfill frames') as frames:
        for index,_,image in frames:
            with image:
                if index not in wanted:continue
                # The same measured bounds can serve a body and its face projection.
                encoded={}
                with db:
                    for table,row in rows(db,scan,config,index):
                        metadata=json.loads(row['metadata']);person=metadata.get('person') or {}
                        bounds=person.get('bounds',())
                        valid=(len(bounds)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in bounds)
                               and 0<=bounds[0]<bounds[2]<=image.width and 0<=bounds[1]<bounds[3]<=image.height)
                        crop=None;status='geometry_unavailable'
                        if valid:
                            key=tuple(bounds)
                            if key not in encoded:
                                encoded.clear()
                                with image.crop(bounds) as pixels:
                                    out=BytesIO();pixels.save(out,format='PNG')
                                encoded[key]=out.getvalue() if out.tell()<=limit else None
                            crop=encoded[key];status='retained' if crop is not None else 'byte_budget_exceeded'
                        field='metadata' if table=='person_evidence' else 'quality'
                        identity='observation' if table=='person_evidence' else 'id'
                        expected=contract(config) if table=='person_evidence' else body_contract(config)
                        db.execute(f"UPDATE {table} SET crop=?,{field}=json_set({field},'$.crop_status',?,"
                            "'$.crop_attempt_limit',?) WHERE "+identity+'=? AND contract=?',
                            (crop,status,limit,row['id'],expected))
