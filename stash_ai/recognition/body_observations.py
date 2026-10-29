"""Independent, optional-face person observations; no model imports at read time."""
import uuid

from .config import digest, plain

SCHEMA = '''
CREATE TABLE body_observations (
 id TEXT PRIMARY KEY, scan TEXT NOT NULL REFERENCES scans(id), cue INTEGER NOT NULL,
 seconds REAL NOT NULL, contract TEXT NOT NULL, face TEXT REFERENCES observations(id),
 geometry TEXT NOT NULL, quality TEXT NOT NULL, rejected TEXT, embedding BLOB, crop BLOB,
 CHECK(embedding IS NULL OR length(embedding)=16384));
CREATE INDEX body_observations_scan ON body_observations(scan,contract,cue,id);
CREATE INDEX body_observations_face ON body_observations(face,contract);
CREATE TABLE body_cues (
 scan TEXT NOT NULL REFERENCES scans(id), contract TEXT NOT NULL, cue INTEGER NOT NULL,
 PRIMARY KEY(scan,contract,cue));
'''


def defaults():
    return dict(enabled=False, min_person_pixels=64, min_sharpness=10.,
                min_keypoints=4, max_overlap=.65)


def enabled(config):
    return config.data['sapiensid']['enabled'] and config.data['sapiensid']['collection']['enabled']


def available(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='body_observations' AND type='table'").fetchone() is not None


def _contract(config,version):
    from .person_signal import contract as person_contract
    settings=plain(config.data['sapiensid']['collection']);settings.pop('enabled')
    return digest([version,person_contract(config),settings])


def contract(config):
    return _contract(config,'independent-person-v2-rgb-pixels')


def compatible_contracts(config):
    # Old grayscale digests cannot train current cores, but their encoder and
    # collection policy are identical, so they remain conservative rivals.
    return [contract(config),_contract(config,'independent-person-v1')]


def identifier(scan, cue, person, expected):
    return str(uuid.uuid5(uuid.UUID(scan),digest([expected,cue,person['bounds'],person['keypoints']])))


def save_records(db, config, records):
    """Publish a complete cue in the caller's face/body checkpoint transaction."""
    from .database import encoded
    from .models import normalized_embedding
    from .person_signal import save, contract as person_contract
    for identity,metadata,vector,crop in records:
        marker=metadata.get('_body_cue')
        if marker is not None:
            db.execute('INSERT OR IGNORE INTO body_cues VALUES (?,?,?)',tuple(marker))
            continue
        body=metadata.get('_body')
        if body is None:
            save(db,identity,person_contract(config),metadata,vector,crop)
            continue
        feature=normalized_embedding(vector,4096).astype('<f4').tobytes() if vector is not None else None
        public={k:v for k,v in metadata.items() if not k.startswith('_')}
        db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?) '
            'ON CONFLICT(id) DO UPDATE SET face=excluded.face,geometry=excluded.geometry,'
            'quality=excluded.quality,rejected=excluded.rejected,embedding=excluded.embedding,crop=excluded.crop',
            (identity,body['scan'],body['cue'],body['seconds'],body['contract'],body['face'],
             encoded(metadata['person']),encoded(public),body['rejected'],feature,crop))
        if body['face_accepted']:
            # Compatibility projection uses the SAME inferred feature. Never
            # invoke the encoder again for a face-associated person.
            save(db,body['face'],person_contract(config),dict(public,identity_rejection=None),vector,crop)


def coverage(db,config,scans):
    from .database import encoded
    wanted=encoded(list(scans));expected=contract(config)
    if not available(db):return dict(enabled=enabled(config),contract=expected,complete_cues=0,observations=0,ready=0,rejected=0)
    complete=db.execute('SELECT COUNT(*) FROM body_cues WHERE contract=? AND scan IN (SELECT value FROM json_each(?))',
                        (expected,wanted)).fetchone()[0]
    rows=db.execute('SELECT COUNT(*),SUM(rejected IS NULL AND embedding IS NOT NULL),SUM(rejected IS NOT NULL) '
                    'FROM body_observations WHERE contract=? AND scan IN (SELECT value FROM json_each(?))',
                    (expected,wanted)).fetchone()
    return dict(enabled=enabled(config),contract=expected,complete_cues=complete,
                observations=rows[0],ready=rows[1] or 0,rejected=rows[2] or 0)


def copy_evidence(source,target,scan):
    if not available(source):return
    target.executemany('INSERT OR REPLACE INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        source.execute('SELECT id,scan,cue,seconds,contract,face,geometry,quality,rejected,embedding,NULL '
                       'FROM body_observations WHERE scan=?',(scan,)))
    target.executemany('INSERT OR REPLACE INTO body_cues VALUES (?,?,?)',
                      source.execute('SELECT * FROM body_cues WHERE scan=?',(scan,)))
