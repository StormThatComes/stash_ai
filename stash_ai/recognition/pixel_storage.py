"""Content-addressed thumbnail BLOBs; legacy inline quality remains readable."""
import hashlib
import json
def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

SCHEMA = '''CREATE TABLE evidence_thumbnails (
 id TEXT PRIMARY KEY, pixels BLOB NOT NULL CHECK(length(pixels)=3072));'''


def compact(db, raw):
    quality=json.loads(raw)
    thumbnail=quality.pop('evidence_thumbnail',None)
    if thumbnail is not None:
        pixels=bytes.fromhex(thumbnail)
        if len(pixels)!=3072:raise ValueError('Invalid evidence thumbnail.')
        key=hashlib.sha256(pixels).hexdigest()
        db.execute('INSERT OR IGNORE INTO evidence_thumbnails VALUES (?,?)',(key,pixels))
        quality['evidence_thumbnail_id']=key
    return encoded(quality)


def read(db, quality):
    if 'evidence_thumbnail' in quality:return bytes.fromhex(quality['evidence_thumbnail'])
    key=quality.get('evidence_thumbnail_id')
    if key is None:return None
    row=db.execute('SELECT pixels FROM evidence_thumbnails WHERE id=?',(key,)).fetchone()
    if row is None or hashlib.sha256(row[0]).hexdigest()!=key:
        raise ValueError('Missing or changed evidence thumbnail.')
    return row[0]


def copy_rows(source,target,rows):
    keys={json.loads(r['quality']).get('evidence_thumbnail_id') for r in rows}-{None}
    if not keys:return
    target.executemany('INSERT OR IGNORE INTO evidence_thumbnails VALUES (?,?)',source.execute(
        'SELECT id,pixels FROM evidence_thumbnails WHERE id IN (SELECT value FROM json_each(?))',(encoded(sorted(keys)),)))
