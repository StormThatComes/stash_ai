"""Model-independent contracts for optional retained person evidence.

This module must remain usable without torch, CUDA or model assets.
"""
from .config import digest

DIMENSIONS = 4096
CONTRACT = 'sapiensid-wb4m-person-v2'
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
        execution=dict(batch_size=1,reference_bytes=134217728,plan_bytes=16777216,max_candidate_pairs=100000),
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
                        ('execution','batch_size'),('execution','reference_bytes'),
                        ('execution','plan_bytes'),('execution','max_candidate_pairs')):
        value=settings[section][key]
        if type(value) is not int or value<1:
            raise ValueError(f'sapiensid.{section}.{key} must be a positive integer.')
    if settings['execution']['batch_size']>16:
        raise ValueError('sapiensid.execution.batch_size must be in [1,16].')
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
    # Older replay schemas need the observation trigger fallback. Current
    # schemas stamp person evidence directly without invalidating face vectors.
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='recognition_version_person_evidence_insert'").fetchone():
        db.execute('UPDATE observations SET quality=quality WHERE id=?',(observation,))


def copy_evidence(source,target,scan):
    """Copy only admitted observations into offline trials; never copy pixels."""
    if not available(source):return
    target.executemany('INSERT OR REPLACE INTO person_evidence VALUES (?,?,?,?,?,NULL)',source.execute(
        'SELECT e.observation,e.contract,e.status,e.metadata,e.embedding FROM person_evidence e '
        'JOIN observations o ON o.id=e.observation WHERE o.scan=?',(scan,)))


def coverage(db,config,scans):
    """Offline availability for selected observations; no inference or image reads."""
    from .database import encoded
    from .config import plain
    wanted=encoded(list(scans));expected=contract(config)
    eligible=db.execute('SELECT COUNT(*) FROM observations WHERE rejected IS NULL '
                        'AND scan IN (SELECT value FROM json_each(?))',(wanted,)).fetchone()[0]
    counts={}
    if available(db):
        counts=dict(db.execute('SELECT e.status,COUNT(*) FROM person_evidence e JOIN observations o '
            'ON o.id=e.observation WHERE o.rejected IS NULL AND e.contract=? '
            'AND o.scan IN (SELECT value FROM json_each(?)) GROUP BY e.status',(expected,wanted)))
    return dict(enabled=config.data['sapiensid']['enabled'],contract=expected,
        matching=plain(config.data['sapiensid']['matching']),eligible=eligible,
        ready=counts.get('ready',0),unavailable=counts.get('unavailable',0),
        missing=eligible-sum(counts.values()))


def same_contract(left,right):
    return (not (left.data['sapiensid']['enabled'] or right.data['sapiensid']['enabled'])
            or contract(left)==contract(right))
