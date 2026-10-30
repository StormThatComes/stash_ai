"""Verify compression duplicates before independent evidence is counted.

Perceptual signatures only retrieve candidates. A copy needs matching aspect,
textured RGB pixels (strict residual bounds), and an almost identical embedding.
Aliases always refer to the original representative, never a transitive chain.
Only the private replay's compact metadata is changed.
"""
from .resource_errors import ResourceLimit
import json
from contextvars import ContextVar
from collections import OrderedDict
_pixel_proofs=ContextVar('recognition_pixel_proofs',default=None)
import numpy as np
from .run_cache import metadata
from .progress import tracking


def signature(pixels):
    gray = pixels.reshape(32, 32, 3).astype('float64').mean(axis=2)
    coarse = gray.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    return np.packbits(coarse[:, 1:] > coarse[:, :-1]).tobytes()


def _verified(left, right, a, b, aspect_a, aspect_b):
    if not (isinstance(aspect_a, (int, float)) and isinstance(aspect_b, (int, float))):
        return False
    if not (0 < aspect_a and 0 < aspect_b) or abs(aspect_a / aspect_b - 1) > .005:
        return False
    residual = left.astype('float64') - right.astype('float64')
    return (min(float(left.std()), float(right.std())) >= 12 and
            float(np.max(np.abs(residual))) <= 8 and float(np.mean(residual * residual)) <= 4 and
            a.shape == b.shape and a.size > 0 and float(np.dot(a, b)) >= .999)


def _cached_verified(left, right, a, b, aspect_a, aspect_b):
    from .run_cache import current, MISSING
    cache=current()
    if cache is None:return _verified(left,right,a,b,aspect_a,aspect_b)
    key=('pixel-proof',left.tobytes(),right.tobytes(),a.tobytes(),b.tobytes(),aspect_a,aspect_b)
    value=cache.get(key)
    if value is MISSING:
        value=_verified(left,right,a,b,aspect_a,aspect_b)
        cache.put(key,value,sum(len(v) for v in key[1:5])+512)
    return value


def verified(left,right,a,b,aspect_a,aspect_b):
    pool=_pixel_proofs.get()
    if pool is None:return _cached_verified(left,right,a,b,aspect_a,aspect_b)
    from .rebuild_cache import _hash_row
    import hashlib
    h=hashlib.sha256()
    _hash_row(h,[ALIAS_FORMAT,left.tobytes(),right.tobytes(),a.dtype.str,a.shape,a.tobytes(),
                 b.dtype.str,b.shape,b.tobytes(),aspect_a,aspect_b])
    key=h.hexdigest();entries,limit=pool
    if key in entries:
        entries.move_to_end(key)
        return entries[key]
    result=_cached_verified(left,right,a,b,aspect_a,aspect_b)
    if limit>=512:
        entries[key]=bool(result)
        while len(entries)*512>limit:entries.popitem(last=False)
    return result


def _aliases(db,scope=None,namespace=None,*,budget=67108864):
    import sqlite3
    scratch = None
    rows = candidates = None
    used=0
    try:
        join=' JOIN scans s ON s.id=o.scan' if scope is not None or namespace is not None else ''
        where=' AND (? IS NULL OR s.scope=?) AND (? IS NULL OR s.namespace=?)' if join else ''
        args=(scope,scope,namespace,namespace) if join else ()
        rows = db.execute("SELECT o.id,o.quality,o.embedding FROM observations o"+join+" WHERE o.rejected IS NULL "
            "AND (json_extract(quality,'$.evidence_thumbnail') IS NOT NULL OR json_extract(quality,'$.evidence_thumbnail_id') IS NOT NULL) "+where+
            " ORDER BY json_extract(quality,'$.evidence_pixel_sha256'),o.id",args)
        with tracking(rows, 'Verify duplicate VTT crops', unit='observation') as tracked:
            for row in tracked:
                record = dict(metadata(row['quality']))
                from .pixel_storage import read
                raw = read(db,record)
                fingerprint = record.get('evidence_pixel_sha256')
                if len(raw) != 3072 or not fingerprint:
                    continue
                pixels = np.frombuffer(raw, dtype='uint8')
                sample = np.frombuffer(row['embedding'], dtype='<f4')
                if scratch is None:
                    scratch = sqlite3.connect('')
                    scratch.row_factory = sqlite3.Row
                    scratch.execute('CREATE TABLE pixel_witnesses(signature BLOB, fingerprint TEXT PRIMARY KEY, pixels BLOB, embedding BLOB, aspect REAL)')
                    scratch.execute('CREATE INDEX pixel_witness_signature ON pixel_witnesses(signature)')
                bucket = signature(pixels)
                alias = None
                candidates = scratch.execute('SELECT * FROM pixel_witnesses WHERE signature=? ORDER BY fingerprint', (bucket,))
                for candidate in candidates:
                    if fingerprint == candidate['fingerprint'] or verified(pixels,
                            np.frombuffer(candidate['pixels'], dtype='uint8'), sample,
                            np.frombuffer(candidate['embedding'], dtype='<f4'),
                            record.get('evidence_aspect'), candidate['aspect']):
                        alias = candidate['fingerprint']
                        break
                candidates.close()
                if alias is None:
                    used+=len(raw)+len(row['embedding'])+1024
                    if used>budget:raise ResourceLimit('Pixel witness index exceeds recognition evidence budget.')
                    scratch.execute('INSERT OR IGNORE INTO pixel_witnesses VALUES (?,?,?,?,?)',
                        (bucket, fingerprint, raw, row['embedding'], record.get('evidence_aspect')))
                elif alias != fingerprint:
                    yield fingerprint, alias
    finally:
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('close pixel alias reader', resource.close)
                 for resource in (rows, candidates, scratch) if resource is not None])


def _generation(db,scope,namespace):
    from .dependencies import version
    from .rebuild_cache import _hash_row
    import hashlib
    # Raw scan generations survive membership/portrait publication. Thumbnail
    # bytes are checked too: their content-addressed IDs are not trusted alone.
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='recognition_raw_observations_insert'").fetchone():
        return str(version(db))
    h=hashlib.sha256()
    args=(scope,scope,namespace,namespace)
    for row in db.execute("SELECT s.id,COALESCE(v.version,0) FROM scans s LEFT JOIN evidence_versions v "
        "ON v.kind='raw-observations' AND v.key=s.id WHERE (? IS NULL OR s.scope=?) "
        "AND (? IS NULL OR s.namespace=?) ORDER BY s.id",args):_hash_row(h,row)
    for row in db.execute("SELECT t.id,t.pixels FROM evidence_thumbnails t WHERE t.id IN ("
        "SELECT json_extract(o.quality,'$.evidence_thumbnail_id') FROM observations o JOIN scans s ON s.id=o.scan "
        "WHERE (? IS NULL OR s.scope=?) AND (? IS NULL OR s.namespace=?)) ORDER BY t.id",args):_hash_row(h,row)
    return h.hexdigest()


def aliases(db,scope=None,namespace=None,*,budget=67108864,persist=False):
    from .dependencies import snapshot
    with snapshot(db):
        yield from _aliases_cached(db,scope,namespace,budget=budget,persist=persist)


def _aliases_cached(db,scope=None,namespace=None,*,budget=67108864,persist=False):
    """Persistent scoped map guarded by audited transactional dependencies."""
    from .dependencies import supported,version,_writable
    from .database import encoded
    from .config import digest
    from .profiling import count
    cached=supported(db)
    key='pixel-alias-cache:'+digest([ALIAS_FORMAT,scope,namespace])
    generation=_generation(db,scope,namespace) if cached else None
    if cached:
        row=db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
        if row is not None and len(row[0])<=budget:
            try:
                record=json.loads(row[0]);checksum=record.pop('checksum')
                mapping=record['aliases']
                if (record['version']==generation and checksum==digest(record)
                    and isinstance(mapping,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in mapping.items())
                    and sum(len(k)+len(v)+256 for k,v in mapping.items())<=budget*3//4):
                    count('pixel_alias_index_hits')
                    yield from mapping.items()
                    return
            except (ValueError,KeyError,TypeError):pass
    # Reuse unchanged verification edges even after an inserted/changed row
    # invalidates the complete map. Keys include exact pixels and embeddings.
    proof_key='pixel-alias-cache:proofs:'+digest([ALIAS_FORMAT,scope,namespace])
    proofs=OrderedDict()
    row=db.execute('SELECT value FROM metadata WHERE key=?',(proof_key,)).fetchone() if cached else None
    if row is not None and len(row[0])<=budget//4:
        try:
            record=json.loads(row[0]);checksum=record.pop('checksum')
            entries=record['proofs']
            if (checksum==digest(record) and isinstance(entries,dict) and len(entries)*512<=budget//4
                and all(isinstance(k,str) and len(k)==64 and type(v) is bool for k,v in entries.items())):
                proofs.update(entries)
        except (ValueError,KeyError,TypeError):pass
    mapping={};used=0
    token=_pixel_proofs.set((proofs,budget//4))
    try:
        for a,b in _aliases(db,scope,namespace,budget=budget*3//4):
            if a not in mapping:used+=len(a)+len(b)+256
            if used>budget*3//4:raise ResourceLimit('Pixel alias map exceeds recognition evidence budget.')
            mapping[a]=b
    finally:_pixel_proofs.reset(token)
    if cached and persist and _writable.get() and not db.execute('PRAGMA query_only').fetchone()[0]:
        record=dict(version=generation,aliases=mapping)
        record['checksum']=digest(record)
        payload=encoded(record)
        if len(payload.encode())<=budget:
            try:
                proof_record=dict(proofs=proofs)
                proof_record['checksum']=digest(proof_record)
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(proof_key,encoded(proof_record)))
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,payload))
                total=0
                for old,size in db.execute("SELECT key,length(CAST(value AS BLOB)) FROM metadata WHERE key LIKE 'pixel-alias-cache:%' ORDER BY rowid DESC"):
                    total+=size+len(old)+256
                    if total>budget:db.execute('DELETE FROM metadata WHERE key=?',(old,))
            except __import__('sqlite3').DatabaseError as error:
                if not any(reason in str(error).lower() for reason in ('readonly','not authorized')):raise
    yield from mapping.items()


# Increment when the pixel verification contract changes.
ALIAS_FORMAT = 1


def canonicalize(db):
    """Materialize the same verified aliases in a disposable replay."""
    mapping = dict(aliases(db))
    if not mapping:
        return
    for row in db.execute('SELECT id,quality FROM observations WHERE rejected IS NULL'):
        record = dict(metadata(row['quality']))
        alias = mapping.get(record.get('evidence_pixel_sha256'))
        if alias is not None:
            record['evidence_pixel_sha256'] = alias
            db.execute('UPDATE observations SET quality=? WHERE id=?',
                       (json.dumps(record, separators=(',', ':')), row['id']))


from contextlib import contextmanager
from contextvars import ContextVar
_active_aliases = ContextVar('recognition_pixel_aliases', default=None)


class _LazyAliases:
    def __init__(self,db,scope,namespace,budget,persist):
        self.db,self.mapping=db,None
        self.scope,self.namespace,self.budget,self.persist=scope,namespace,budget,persist

    def get(self,value,default):
        if value is None:return default
        if self.mapping is None:self.mapping=dict(aliases(self.db,self.scope,self.namespace,budget=self.budget,persist=self.persist))
        return self.mapping.get(value,default)


def canonical_pixel(value):
    mapping = _active_aliases.get()
    return mapping.get(value, value) if mapping is not None else value


@contextmanager
def alias_scope(db,scope=None,namespace=None,*,budget=67108864,persist=False):
    """Read-only input snapshot shared by face, body, and report votes.

    Persistent entries require the same transactional evidence generation;
    changed vectors, thumbnails or ownership invalidate the map.
    """
    if _active_aliases.get() is not None:
        yield
        return
    # Warm certificate validation may never count a vote. Do not read every
    # retained thumbnail until the first physical evidence key is requested.
    token = _active_aliases.set(_LazyAliases(db,scope,namespace,budget,persist))
    try:
        yield
    finally:
        _active_aliases.reset(token)
