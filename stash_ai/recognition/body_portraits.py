"""Portraits from retained, currently accepted person crops; no model inference."""
from io import BytesIO
import hashlib
import json
import uuid

from .body_identity import available, enabled, policy as identity_policy
from .body_observations import contract
from .database import encoded


def eligible(db, config, person, observation):
    if not enabled(config) or not available(db):return False
    from .scan_heads import predicate as current_scan
    return db.execute("SELECT 1 FROM body_members m JOIN body_observations b ON b.id=m.observation "
        "JOIN scans s ON s.id=b.scan JOIN people p ON p.id=m.person WHERE m.person=? AND b.id=? "
        "AND m.status='accepted' AND m.policy=? AND b.contract=? AND b.rejected IS NULL "
        "AND s.status='complete' AND s.scope=p.scope AND s.namespace=p.namespace "
        "AND (?=0 OR m.role='anchor') AND "+current_scan(),(person,observation,identity_policy(config),contract(config),
        int(config.data['portraits']['anchor_only']))).fetchone() is not None


def score(quality):
    # Prefer a complete, unobstructed, sharp view. Keep the real crop's framing.
    return (1-quality.get('overlap',0),-quality.get('border_edges',0),
            min(quality.get('sharpness',0),1000),quality.get('keypoints',0),
            min(quality.get('width',0),quality.get('height',0)))


def choose(database, config, scope, target):
    from PIL import Image
    from .portraits import policy_identity
    from .quality import encode_portrait
    db=database.connection
    if not enabled(config) or not available(db):return None
    stamp=policy_identity(config);winner=None;best=None
    for item in sorted(target['scans'],key=lambda s:(s['scene'],s['scan'])):
        rows=db.execute("SELECT b.id,b.quality,b.seconds,m.role FROM body_observations b "
            "JOIN body_members m ON m.observation=b.id WHERE b.scan=? AND m.person=? "
            "AND b.crop IS NOT NULL ORDER BY b.seconds,b.id",(item['scan'],target['person']))
        for row in rows:
            if not eligible(db,config,target['person'],row['id']):continue
            quality=json.loads(row['quality']).get('quality',{})
            key=(*score(quality),-row['seconds'],row['id'])
            if best is not None and key<=best:continue
            blob=db.execute('SELECT crop FROM body_observations WHERE id=?',(row['id'],)).fetchone()[0]
            details={}
            try:
                with Image.open(BytesIO(blob)) as source,source.convert('RGB') as rgb:
                    data=encode_portrait(rgb,config.data['portraits'],limit=config.data['storage']['max_crop_bytes'],details=details)
            except (OSError,ValueError,Image.DecompressionBombError):continue
            identifier=str(uuid.uuid5(uuid.NAMESPACE_URL,encoded(['body-portrait',scope,target['person'],row['id'],stamp])))
            metadata=dict(portrait_kind='person',source_type='person_crop',observation=row['id'],
                person=target['person'],policy_sha=stamp,quality=quality,encoding=details,
                identity_policy=identity_policy(config),role=row['role'],score=list(score(quality)))
            with db:
                db.execute('INSERT OR REPLACE INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                    (identifier,scope,target['person'],item['scene'],encoded(metadata),data,
                     hashlib.sha256(data).hexdigest(),'selected'))
            winner=db.execute('SELECT * FROM portraits WHERE id=?',(identifier,)).fetchone();best=key
    return winner


def current_prior(database, prior, config, target):
    from .portrait_availability import fit_cached_bytes
    metadata=json.loads(prior['metadata'])
    if config.data['portraits']['kind']=='face' or not eligible(database.connection,config,target['person'],metadata.get('observation')):
        return None
    try:fit_cached_bytes(prior['image'],prior['digest'],config,dict(metadata))
    except (OSError,ValueError,KeyError,TypeError):return None
    return prior


def choose_kind(database,client,config,scope,target,paths,backend_factory,decoder_factory,source_factory,**kwargs):
    from .portraits import choose as face_choose
    from .quality import portrait_shortfalls
    kind=config.data['portraits']['kind']
    face=None if kind=='person' else face_choose(database,client,config,scope,target,paths,
                                                backend_factory,decoder_factory,source_factory,**kwargs)
    if kind=='face':return face
    if face is not None and not portrait_shortfalls(json.loads(face['metadata']).get('quality',{}),config.data['portraits']):
        return face
    return choose(database,config,scope,target) or face


def rank(metadata,config):
    from .quality import portrait_score,portrait_tiebreak,portrait_shortfalls
    quality=metadata.get('quality',{})
    if metadata.get('portrait_kind')=='person':return (1,*[-v for v in score(quality)])
    preference=0 if not enabled(config) or not portrait_shortfalls(quality,config.data['portraits']) else 2
    ranks={'video':0,'vtt':1,'recognition_crop':2}
    return (preference,-portrait_score(quality,config.data['portraits']),
            -portrait_tiebreak(quality,config.data['portraits']),ranks.get(metadata.get('source_type'),3))
