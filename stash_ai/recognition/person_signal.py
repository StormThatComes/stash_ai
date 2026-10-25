"""Model-independent contracts for optional retained person evidence.

This module must remain usable without torch, CUDA or model assets.
"""
from .config import digest

DIMENSIONS = 4096
CONTRACT = 'sapiensid-wb4m-person-v1'
SCHEMA = '''
CREATE TABLE person_evidence (
 observation TEXT NOT NULL REFERENCES observations(id),
 contract TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('ready','unavailable')),
 metadata TEXT NOT NULL,
 embedding BLOB,
 crop BLOB,
 PRIMARY KEY(observation,contract),
 CHECK((status='ready' AND embedding IS NOT NULL AND length(embedding)=16384)
    OR (status='unavailable' AND embedding IS NULL)));
'''


def defaults():
    return dict(enabled=False,
        model=dict(source_directory='', checkpoint_directory='',
            model_yaml_sha256='2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9'),
        pose=dict(path='',sha256='',score_threshold=.5,keypoint_score_threshold=.5,
                  nms_threshold=.7,max_people_per_frame=100),
        matching=dict(face_floor=.4,person_threshold=.85,ambiguity_margin=.1,min_support=3),
        execution=dict(batch_size=1,reference_bytes=134217728),
        storage=dict(retain_person_crops=True))


def validate(settings):
    from .config import _number
    for key in ('enabled',):
        if type(settings[key]) is not bool:
            raise ValueError('sapiensid.'+key+' must be boolean.')
    if type(settings['storage']['retain_person_crops']) is not bool:
        raise ValueError('sapiensid.storage.retain_person_crops must be boolean.')
    for section,keys in (('pose',('score_threshold','keypoint_score_threshold','nms_threshold')),
                         ('matching',('face_floor','person_threshold','ambiguity_margin'))):
        for key in keys:
            _number(settings[section][key],f'sapiensid.{section}.{key}',0,1)
    for section,key in (('pose','max_people_per_frame'),('matching','min_support'),
                        ('execution','batch_size'),('execution','reference_bytes')):
        value=settings[section][key]
        if type(value) is not int or value<1:
            raise ValueError(f'sapiensid.{section}.{key} must be a positive integer.')
    for section,key in (('model','source_directory'),('model','checkpoint_directory'),('pose','path')):
        value=settings[section][key]
        if not isinstance(value,str) or '\x00' in value or settings['enabled'] and not value.strip():
            raise ValueError(f'sapiensid.{section}.{key} requires a local path.')
    for section,key in (('model','model_yaml_sha256'),('pose','sha256')):
        value=settings[section][key]
        if not isinstance(value,str) or (value or settings['enabled']) and (
                len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError(f'sapiensid.{section}.{key} requires a lowercase SHA256.')


def contract(config):
    """Paths, matching thresholds and resource budgets do not change vectors."""
    s=config.data['sapiensid']
    return digest(dict(format=CONTRACT,model_yaml=s['model']['model_yaml_sha256'],
        pose={k:v for k,v in s['pose'].items() if k!='path'}))


def available(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_evidence'").fetchone() is not None


def read(db, observation, expected):
    if not available(db):
        return None
    return db.execute('SELECT * FROM person_evidence WHERE observation=? AND contract=?',
                      (observation,expected)).fetchone()


def save(db, observation, expected, metadata, vector=None, crop=None):
    """Caller owns the transaction; unavailable input is distinct from inference failure."""
    from .database import encoded
    if vector is not None:
        from .models import normalized_embedding
        vector=normalized_embedding(vector,DIMENSIONS).astype('<f4').tobytes()
    db.execute('INSERT INTO person_evidence VALUES (?,?,?,?,?,?) '
               'ON CONFLICT(observation,contract) DO UPDATE SET status=excluded.status,'
               'metadata=excluded.metadata,embedding=excluded.embedding,crop=excluded.crop',
               (observation,expected,'ready' if vector is not None else 'unavailable',encoded(metadata),vector,crop))
    # Reuse the existing audited observation dependency triggers. Persisted
    # rebuild certificates cannot survive changes to supplemental evidence.
    db.execute('UPDATE observations SET quality=quality WHERE id=?',(observation,))
