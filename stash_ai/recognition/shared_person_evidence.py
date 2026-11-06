"""Shared face/body payloads with copy-on-write projection semantics.

A projection stores a tagged body address, never a second 4096D feature or JPEG.
Legacy inline payloads remain readable. Body replacement/deletion materializes
references first, so retention and partial retries cannot leave dangling data.
"""
PREFIX = b"body:"
TAG = "X'626f64793a'"


def available(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='recognition_shared_body_delete'").fetchone())


def project(db, observation, contract, body, metadata, *, retain_crop=True, supported=None, validate_feature=False):
    """Publish the authoritative body address without writing duplicate BLOBs.

    Return False for legacy schemas; callers retain their inline fallback.
    The body and projection must be written in the same caller transaction.
    """
    if not (available(db) if supported is None else supported):
        return False
    from .database import encoded
    row = db.execute('SELECT face,rejected,length(embedding),crop IS NOT NULL '
                     'FROM body_observations WHERE id=?', (body,)).fetchone()
    if row is None or row[0] != observation:
        raise ValueError('Shared body projection requires an authoritative face association.')
    ready = row[1] is None and row[2] is not None
    if validate_feature and row[1] is None:
        if row[2] is None:
            raise ValueError('Completed body cue is missing its retained feature.')
        from .models import normalized_embedding
        import numpy as np
        raw=db.execute('SELECT embedding FROM body_observations WHERE id=?',(body,)).fetchone()[0]
        normalized_embedding(np.frombuffer(raw,dtype='<f4'),4096)
    if ready and row[2] != 16384:
        raise ValueError('Invalid shared body embedding length.')
    tag = PREFIX + body.encode()
    db.execute('INSERT INTO person_evidence VALUES (?,?,?,?,?,?) '
        'ON CONFLICT(observation,contract) DO UPDATE SET status=excluded.status,'
        'metadata=excluded.metadata,embedding=excluded.embedding,crop=excluded.crop',
        (observation,contract,'ready' if ready else 'unavailable',encoded(metadata),
         tag if ready else None,tag if retain_crop and row[3] else None))
    return True


def compact_cue(db, scan, cue, body_contract, person_contract):
    """Re-share only byte-identical restored payloads; retain old projections."""
    if not available(db):return
    for observation, body in db.execute('SELECT e.observation,b.id FROM person_evidence e '
        'JOIN body_observations b ON b.face=e.observation '
        'WHERE b.scan=? AND b.cue=? AND b.contract=? AND e.contract=? ORDER BY b.id',
        (scan,cue,body_contract,person_contract)):
        share(db,observation,person_contract,body)


def expression(column, alias='e'):
    value=f'{alias}.{column}'
    return (f"CASE WHEN substr({value},1,5)={TAG} THEN "
            f"(SELECT {column} FROM body_observations WHERE id=CAST(substr({value},6) AS TEXT)) "
            f"ELSE {value} END")


def resolve(db, row):
    if row is None:return None
    result=dict(row)
    for column in ('embedding','crop'):
        value=result[column]
        if isinstance(value,bytes) and value.startswith(PREFIX) and len(value)<256:
            source=db.execute(f'SELECT {column} FROM body_observations WHERE id=?',
                              (value[5:].decode('utf8'),)).fetchone()
            if source is None:raise ValueError('Missing shared body evidence.')
            result[column]=source[0]
    from .cold_payloads import get
    result['crop']=get(db,result['crop'])
    return result


def share(db, observation, contract, body):
    # Disposable historical schemas retain the inline representation.
    if not available(db):return
    tag=PREFIX+body.encode()
    db.execute('UPDATE person_evidence SET embedding=CASE WHEN person_evidence.embedding=b.embedding THEN ? ELSE person_evidence.embedding END,'
        'crop=CASE WHEN person_evidence.crop=b.crop AND person_evidence.crop IS NOT NULL THEN ? ELSE person_evidence.crop END FROM body_observations b '
        'WHERE observation=? AND person_evidence.contract=? AND b.id=? AND b.face=observation',
        (tag,tag,observation,contract,body))


def triggers():
    output={}
    for event in ('DELETE','UPDATE','INSERT'):
        address='old.id' if event!='INSERT' else 'new.id'
        condition=(" WHEN old.id IS NOT new.id OR old.embedding IS NOT new.embedding OR old.crop IS NOT new.crop"
                   if event=='UPDATE' else '')
        tag=f"CAST('body:'||{address} AS BLOB)"
        def value(column):
            return f'old.{column}' if event!='INSERT' else f'(SELECT {column} FROM body_observations WHERE id=new.id)'
        name='recognition_shared_body_'+event.lower()
        output[name]=(f'CREATE TRIGGER {name} BEFORE {event} ON body_observations{condition} BEGIN '
            f'UPDATE person_evidence SET embedding=CASE WHEN embedding={tag} THEN {value("embedding")} ELSE embedding END,'
            f'crop=CASE WHEN crop={tag} THEN {value("crop")} ELSE crop END '
            f'WHERE (length(embedding)<256 AND embedding={tag}) OR (length(crop)<256 AND crop={tag}); END')
    return output


TRIGGERS=triggers()
INDEXES=('CREATE INDEX IF NOT EXISTS person_shared_embedding ON person_evidence(embedding) WHERE length(embedding)<256',
         'CREATE INDEX IF NOT EXISTS person_shared_crop ON person_evidence(crop) WHERE length(crop)<256')
SCHEMA=';\n'.join([*INDEXES,*TRIGGERS.values()])+';\n'


def migrate(db, person_schema):
    triggers=list(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name='person_evidence'"))
    for name,_ in triggers:db.execute('DROP TRIGGER '+name)
    db.execute(person_schema.replace('person_evidence (','person_evidence_new ('))
    db.execute('INSERT INTO person_evidence_new SELECT * FROM person_evidence')
    db.execute('DROP TABLE person_evidence')
    db.execute('ALTER TABLE person_evidence_new RENAME TO person_evidence')
    for _,sql in triggers:db.execute(sql)
    for sql in INDEXES:db.execute(sql)
    for sql in TRIGGERS.values():db.execute(sql)
    # Indexed face lookup; equality includes payload bytes and retains old contracts.
    for observation,contract,body in db.execute('SELECT e.observation,e.contract,b.id FROM person_evidence e '
        'JOIN body_observations b ON b.face=e.observation WHERE e.embedding=b.embedding '
        'AND e.crop IS b.crop AND b.id=(SELECT MIN(x.id) FROM body_observations x '
        'WHERE x.face=e.observation AND x.embedding=e.embedding AND x.crop IS e.crop)'):
        share(db,observation,contract,body)
