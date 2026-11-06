"""Incremental disk index for exact, non-transitive duplicate representatives."""
import hashlib
import json
from collections import OrderedDict
import numpy as np
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
from .config import digest
from .dependencies import version
from .profiling import count

TABLES=('pixel_catalogue_scans','pixel_catalogue_rows','pixel_catalogue_aliases')
SCHEMA="""
CREATE TABLE pixel_catalogue_scans(scope TEXT,namespace TEXT,scan TEXT,generation TEXT,PRIMARY KEY(scope,namespace,scan));
CREATE TABLE pixel_catalogue_rows(scope TEXT,namespace TEXT,scan TEXT,observation TEXT,signature BLOB,
 fingerprint TEXT,proof TEXT,PRIMARY KEY(scope,namespace,observation));
CREATE INDEX pixel_catalogue_bucket ON pixel_catalogue_rows(scope,namespace,signature,fingerprint,observation);
CREATE INDEX pixel_catalogue_scan ON pixel_catalogue_rows(scope,namespace,scan);
CREATE TABLE pixel_catalogue_aliases(scope TEXT,namespace TEXT,signature BLOB,fingerprint TEXT,representative TEXT,
 PRIMARY KEY(scope,namespace,fingerprint));
CREATE INDEX pixel_catalogue_alias_bucket ON pixel_catalogue_aliases(scope,namespace,signature);
"""


def ready(db,scope,namespace):
    return (scope is not None and namespace is not None and not db.execute('PRAGMA query_only').fetchone()[0]
        and db.execute("SELECT 1 FROM sqlite_master WHERE name='pixel_catalogue_rows'").fetchone() is not None)


class Aliases:
    def __init__(self,db,scope,namespace):
        self.db,self.scope,self.namespace=db,scope,namespace
        self.cache=OrderedDict()
    def get(self,key,default=None):
        if key not in self.cache:
            row=self.db.execute('SELECT representative FROM pixel_catalogue_aliases '
                'WHERE scope=? AND namespace=? AND fingerprint=?',(self.scope,self.namespace,key)).fetchone()
            self.cache[key]=row[0] if row else key
            if len(self.cache)>1024:self.cache.popitem(last=False)
        return self.cache[key]
    def __bool__(self):return True
    def items(self):
        return self.db.execute('SELECT fingerprint,representative FROM pixel_catalogue_aliases '
                               'WHERE scope=? AND namespace=?',(self.scope,self.namespace))


def prepare(db,scope,namespace,*,disk_bytes):
    from .evidence_pixels import signature,verified
    from .pixel_storage import read
    from .rebuild_cache import _hash_row
    from .resource_errors import ResourceLimit
    from .dependency_schema import validate_triggers
    validate_triggers(db,required=True)
    args=(scope,namespace)
    db.execute('CREATE TEMP TABLE IF NOT EXISTS dirty_pixel_buckets(signature BLOB PRIMARY KEY)')
    db.execute('DELETE FROM dirty_pixel_buckets')
    # Retractions dirty the old bucket before removing its addresses.
    db.execute('INSERT OR IGNORE INTO dirty_pixel_buckets SELECT signature FROM pixel_catalogue_rows '
        'WHERE scope=? AND namespace=? AND scan NOT IN (SELECT id FROM scans WHERE scope=? AND namespace=?)',args+args)
    db.execute('DELETE FROM pixel_catalogue_rows WHERE scope=? AND namespace=? '
        'AND scan NOT IN (SELECT id FROM scans WHERE scope=? AND namespace=?)',args+args)
    db.execute('DELETE FROM pixel_catalogue_scans WHERE scope=? AND namespace=? '
        'AND scan NOT IN (SELECT id FROM scans WHERE scope=? AND namespace=?)',args+args)
    scans=db.execute('SELECT s.id,COALESCE(v.version,0),COALESCE(p.version,0),c.generation FROM scans s '
        "LEFT JOIN evidence_versions v ON v.kind='raw-observations' AND v.key=s.id "
        "LEFT JOIN evidence_versions p ON p.kind='pixel-scan' AND p.key=s.id "
        'LEFT JOIN pixel_catalogue_scans c ON c.scan=s.id AND c.scope=s.scope AND c.namespace=s.namespace '
        'WHERE s.scope=? AND s.namespace=? ORDER BY s.id',args)
    for scan,raw,thumb,previous in scans:
        generation=encoded([raw,thumb])
        if generation==previous:continue
        count('pixel_index_dirty_scans')
        db.execute('INSERT OR IGNORE INTO dirty_pixel_buckets SELECT signature FROM pixel_catalogue_rows '
                   'WHERE scope=? AND namespace=? AND scan=?',args+(scan,))
        db.execute('DELETE FROM pixel_catalogue_rows WHERE scope=? AND namespace=? AND scan=?',args+(scan,))
        for row in db.execute('SELECT id,quality,embedding FROM observations WHERE scan=? AND rejected IS NULL',(scan,)):
            q=json.loads(row['quality']);fingerprint=q.get('evidence_pixel_sha256')
            if not fingerprint or not (q.get('evidence_thumbnail_id') or q.get('evidence_thumbnail')):continue
            pixels=read(db,q)
            bucket=signature(np.frombuffer(pixels,dtype='uint8'))
            h=hashlib.sha256();_hash_row(h,[row['embedding'],pixels,q.get('evidence_aspect')])
            db.execute('INSERT INTO pixel_catalogue_rows VALUES (?,?,?,?,?,?,?)',args+(scan,row['id'],bucket,fingerprint,h.hexdigest()))
            db.execute('INSERT OR IGNORE INTO dirty_pixel_buckets VALUES (?)',(bucket,))
        db.execute('INSERT OR REPLACE INTO pixel_catalogue_scans VALUES (?,?,?,?)',args+(scan,generation))
    # Disk storage has its own admission bound; it is not charged as RAM.
    size=db.execute('SELECT COALESCE(SUM(length(observation)+length(fingerprint)+length(proof)+256),0) '
                    'FROM pixel_catalogue_rows').fetchone()[0]
    if size>disk_bytes:raise ResourceLimit('Persistent pixel index exceeds execution.pixel_index_bytes.')
    db.execute('CREATE TEMP TABLE IF NOT EXISTS current_pixel_witnesses(fingerprint TEXT PRIMARY KEY,observation TEXT)')
    for (bucket,) in db.execute('SELECT signature FROM dirty_pixel_buckets ORDER BY signature'):
        count('pixel_index_dirty_buckets')
        db.execute('DELETE FROM current_pixel_witnesses')
        db.execute('DELETE FROM pixel_catalogue_aliases WHERE scope=? AND namespace=? AND signature=?',args+(bucket,))
        for oid,fingerprint in db.execute('SELECT observation,fingerprint FROM pixel_catalogue_rows '
                'WHERE scope=? AND namespace=? AND signature=? ORDER BY fingerprint,observation',args+(bucket,)):
            row=db.execute('SELECT quality,embedding FROM observations WHERE id=?',(oid,)).fetchone()
            q=json.loads(row['quality']);pixels=np.frombuffer(read(db,q),dtype='uint8')
            vector=np.frombuffer(row['embedding'],dtype='<f4');alias=None
            for other,observation in db.execute('SELECT fingerprint,observation FROM current_pixel_witnesses ORDER BY fingerprint'):
                if other==fingerprint:alias=other;break
                witness=db.execute('SELECT quality,embedding FROM observations WHERE id=?',(observation,)).fetchone()
                w=json.loads(witness['quality'])
                if verified(pixels,np.frombuffer(read(db,w),dtype='uint8'),vector,np.frombuffer(witness['embedding'],dtype='<f4'),q.get('evidence_aspect'),w.get('evidence_aspect')):
                    alias=other;break
            if alias is None:db.execute('INSERT OR IGNORE INTO current_pixel_witnesses VALUES (?,?)',(fingerprint,oid))
            elif alias!=fingerprint:db.execute('INSERT OR REPLACE INTO pixel_catalogue_aliases VALUES (?,?,?,?,?)',args+(bucket,fingerprint,alias))
    # Other consumers may have an active evidence cursor on this connection.
    # Reuse temp tables instead of issuing DDL underneath that cursor.
    db.execute('DELETE FROM current_pixel_witnesses');db.execute('DELETE FROM dirty_pixel_buckets')
    return Aliases(db,scope,namespace)
