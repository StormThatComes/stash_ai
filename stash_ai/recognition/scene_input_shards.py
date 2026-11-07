"""Independent pre-canonicalization replay inputs in one bounded SQLite cache."""
from contextlib import contextmanager,closing
import hashlib
import json
import os
import sqlite3
import time
from .database import encoded
from .profiling import count

SCHEMA=(
    'CREATE TABLE shards(scan TEXT PRIMARY KEY, stamp TEXT, checksum TEXT, size INTEGER, touched INTEGER)',
    'CREATE TABLE records(scan TEXT, pos INTEGER, doc TEXT, vector BLOB, pixels BLOB, PRIMARY KEY(scan,pos)) WITHOUT ROWID',
    'CREATE INDEX shard_age ON shards(touched)',
)


class Shards:
    def __init__(self,db,cache,shared,limit):
        self.db,self.cache,self.shared,self.limit=db,cache,shared,limit

    def stamp(self,scan):
        from .config import digest
        return digest(['raw-shard-v1',self.cache.schema_stamp,self.cache.owner,scan,self.cache.stamps[scan],self.shared])

    def remove(self,scan):
        self.db.execute('DELETE FROM records WHERE scan=?',(scan,))
        self.db.execute('DELETE FROM shards WHERE scan=?',(scan,))

    @staticmethod
    def update(h,doc,vector,pixels):
        for value in (doc.encode(),vector,pixels):
            if value is None:h.update(b'N');continue
            h.update(b'V'+len(value).to_bytes(8,'big'));h.update(value)

    def restore(self,replay,scan):
        try:meta=self.db.execute('SELECT stamp,checksum FROM shards WHERE scan=?',(scan,)).fetchone()
        except sqlite3.DatabaseError:
            count('input_shard_rejections');return False
        if not meta or meta[0]!=self.stamp(scan):return False
        h=hashlib.sha256()
        replay.execute('SAVEPOINT shard_import')
        try:
            for doc,vector,pixels in self.db.execute('SELECT doc,vector,pixels FROM records WHERE scan=? ORDER BY pos',(scan,)):
                self.update(h,doc,vector,pixels)
                fields,key,thumb=json.loads(doc)
                if fields[1]!=scan or len(fields)!=10:raise ValueError('Foreign shard row.')
                replay.execute('INSERT INTO main.observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(*fields,vector,None))
                replay.execute('INSERT INTO sample_keys VALUES (?,?)',(fields[0],key))
                if pixels is not None:
                    if hashlib.sha256(pixels).hexdigest()!=thumb:raise ValueError('Changed shard thumbnail.')
                    replay.execute('INSERT OR IGNORE INTO evidence_thumbnails VALUES (?,?)',(thumb,pixels))
            if h.hexdigest()!=meta[1]:raise ValueError('Changed replay shard.')
        except (ValueError,TypeError,sqlite3.DatabaseError):
            replay.execute('ROLLBACK TO shard_import');replay.execute('RELEASE shard_import')
            count('input_shard_rejections');return False
        replay.execute('RELEASE shard_import');count('input_shard_restores');return True

    def save(self,replay,scan):
        try:
            rows=replay.execute("SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,o.rejected,"
                "o.dimensions,o.embedding,k.sample_key,t.id,t.pixels FROM main.observations o "
                "JOIN sample_keys k ON k.observation=o.id LEFT JOIN evidence_thumbnails t "
                "ON t.id=json_extract(o.quality,'$.evidence_thumbnail_id') WHERE o.scan=? ORDER BY o.cue,o.id",(scan,))
            h=hashlib.sha256();size=0
            with self.db:
                from .disk_budget import check
                check(0,self.cache.directory,throttled=True)
                self.remove(scan)
                # max_page_count bounds physical space. Batch eviction leaves spare pages for later scenes.
                if self.db.execute('PRAGMA freelist_count').fetchone()[0]<16 and self.db.execute('PRAGMA page_count').fetchone()[0]*4096>self.limit*.85:
                    for (old,) in self.db.execute('SELECT scan FROM shards ORDER BY touched LIMIT 32').fetchall():
                        self.remove(old)
                for i,row in enumerate(rows):
                    doc=encoded([tuple(row[:10]),row[11],row[12]])
                    self.update(h,doc,row[10],row[13]);size+=len(doc)+len(row[10] or b'')+len(row[13] or b'')
                    self.db.execute('INSERT INTO records VALUES (?,?,?,?,?)',(scan,i,doc,row[10],row[13]))
                self.db.execute('INSERT INTO shards VALUES (?,?,?,?,?)',(scan,self.stamp(scan),h.hexdigest(),size,time.time_ns()))
            count('input_shard_saves')
        except (sqlite3.Error,OSError):count('input_shard_save_failures')


@contextmanager
def open_shards(cache,shared):
    if cache is None or not cache.enabled:
        yield None;return
    db=None
    try:
        cache.owned()
        # Leave room for an existing whole-selection image during rollout.
        available=cache.limit-sum(x.stat().st_size for x in cache.directory.glob('*.sqlite3') if x.name!='shards.sqlite3')
        if available<131072:raise ValueError('Shard cache has no available capacity.')
        path=cache.directory/'shards.sqlite3'
        if path.is_symlink():raise ValueError('Unowned shard cache.')
        db=sqlite3.connect(path)
        db.execute('PRAGMA trusted_schema=OFF')
        schema=tuple(r[0] for r in db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid"))
        if not schema:
            with db:
                db.execute('BEGIN IMMEDIATE')
                for sql in SCHEMA:db.execute(sql)
        elif schema!=SCHEMA:raise ValueError('Unrecognized shard schema.')
        db.execute(f'PRAGMA max_page_count={available//4096}')
        value=Shards(db,cache,shared,available)
    except (ValueError,sqlite3.Error,OSError):
        count('input_shard_unavailable');value=None
    try:yield value
    finally:
        if db is not None:db.close()
