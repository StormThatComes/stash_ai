"""Content-addressed images outside frequently scanned evidence rows.

Inline historical crops remain readable. Addresses are immutable, so shared
body projections retain copy-on-write semantics without copying image bytes.
"""
import hashlib

PREFIX = b'crop:'
SCHEMA = '''
CREATE TABLE crop_payloads(id BLOB PRIMARY KEY, data BLOB NOT NULL);
CREATE INDEX observations_cold_crop ON observations(crop) WHERE length(crop)=69;
CREATE INDEX bodies_cold_crop ON body_observations(crop) WHERE length(crop)=69;
'''


def available(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='crop_payloads'").fetchone() is not None


def put(db, value, *, supported=None):
    if value is None or len(value)<=256 or not (available(db) if supported is None else supported):
        return value
    key=PREFIX+hashlib.sha256(value).hexdigest().encode('ascii')
    db.execute('INSERT OR IGNORE INTO crop_payloads VALUES (?,?)',(key,value))
    return key


def get(db, value):
    if isinstance(value,bytes) and len(value)==69 and value.startswith(PREFIX):
        row=db.execute('SELECT data FROM crop_payloads WHERE id=?',(value,)).fetchone()
        if row is None:raise ValueError('Missing retained crop payload.')
        if PREFIX+hashlib.sha256(row[0]).hexdigest().encode('ascii')!=value:
            raise ValueError('Retained crop payload checksum mismatch.')
        return row[0]
    return value


def length_sql(value):
    return (f"CASE WHEN length({value})=69 AND substr({value},1,5)=X'63726f703a' "
            f"THEN (SELECT length(data) FROM crop_payloads WHERE id={value}) ELSE length({value}) END")


def pack_scene(db,scan):
    """Incrementally relocate historical crops, within the caller transaction."""
    if not available(db):return
    for table,key,where in (('observations','id','scan=?'),('body_observations','id','scan=?'),
            ('person_evidence','observation', 'observation IN (SELECT id FROM observations WHERE scan=?)')):
        extra=',contract' if table=='person_evidence' else ''
        for row in db.execute(f'SELECT {key},crop{extra} FROM {table} WHERE {where} AND length(crop)>256',(scan,)):
            args=(put(db,row[1],supported=True),row[0])+((row[2],) if extra else ())
            db.execute(f'UPDATE {table} SET crop=? WHERE {key}=?'+(' AND contract=?' if extra else ''),args)


def collect(db):
    if not available(db):return
    db.execute('DELETE FROM crop_payloads WHERE NOT EXISTS '
        '(SELECT 1 FROM observations WHERE length(crop)=69 AND crop=crop_payloads.id) '
        'AND NOT EXISTS (SELECT 1 FROM body_observations WHERE length(crop)=69 AND crop=crop_payloads.id) '
        'AND NOT EXISTS (SELECT 1 FROM person_evidence WHERE length(crop)<256 AND crop=crop_payloads.id)')
