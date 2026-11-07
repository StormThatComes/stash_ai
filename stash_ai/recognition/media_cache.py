"""Disposable bounded HTTP cache; only strong origin validators permit reuse."""
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import time


from dataclasses import dataclass


@dataclass(frozen=True)
class Validator:
    sha256: str
    size: int


class VerifiedBytes(bytes):
    """Immutable bytes carrying the digest verified at this exact read boundary."""
    def __new__(cls,body,sha=None):
        value=super().__new__(cls,body)
        value.sha256=sha if sha is not None else hashlib.sha256(body).hexdigest()
        return value


def strong_etag(value):
    return (isinstance(value,str) and len(value)<=4096 and len(value)>=2
            and value[0]==value[-1]=='"' and all(32<=ord(c)<127 for c in value))


class MediaCache:
    def __init__(self,path,limit,client):
        self.path=Path(str(path)+'.media.sqlite3');self.limit=limit
        # Auth changes must revalidate independently; credentials are never stored.
        self.owner=hashlib.sha256((client.url+'\0'+getattr(client,'api_key','')).encode()).hexdigest()

    def key(self,url):
        return hashlib.sha256((self.owner+'\0'+url).encode()).hexdigest()

    def peek(self,url):
        """Read validators only; defer BLOB I/O and hashing until origin says 304."""
        if not self.limit or not self.path.exists():return None
        try:
            with closing(sqlite3.connect(self.path,timeout=5)) as db:
                manifest=db.execute("SELECT 1 FROM sqlite_master WHERE name='manifests'").fetchone()
                row=(db.execute('SELECT etag,sha,size FROM manifests WHERE key=?',(self.key(url),)).fetchone() if manifest else None)
                if row is None:row=db.execute('SELECT etag,sha,length(body) FROM media WHERE key=?',(self.key(url),)).fetchone()
                if row and strong_etag(row[0]) and isinstance(row[1],str) and len(row[1])==64 and all(c in '0123456789abcdef' for c in row[1]) and type(row[2]) is int and 0<=row[2]<=self.limit:
                    return row
        except (sqlite3.Error,OSError):pass
        return None

    def get(self,url):
        if not self.limit or not self.path.exists():return None
        try:
            with closing(sqlite3.connect(self.path,timeout=5)) as db,db:
                row=db.execute('SELECT etag,sha,body FROM media WHERE key=?',(self.key(url),)).fetchone()
                if row is None:return None
                etag,sha,body=row
                if (not strong_etag(etag) or not isinstance(body,bytes) or len(body)>self.limit
                        or hashlib.sha256(body).hexdigest()!=sha):
                    db.execute('DELETE FROM media WHERE key=?',(self.key(url),));return None
                db.execute('UPDATE media SET touched=? WHERE key=?',(time.time_ns(),self.key(url)))
                return etag,VerifiedBytes(body,sha)
        except (sqlite3.Error,OSError):return None

    def put(self,url,etag,body):
        if not self.limit:return
        if not strong_etag(etag) or len(body)>self.limit:
            self.discard(url);return
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            descriptor=os.open(self.path,os.O_CREAT|os.O_WRONLY,0o600);os.close(descriptor)
            with closing(sqlite3.connect(self.path,timeout=5)) as db,db:
                db.execute('CREATE TABLE IF NOT EXISTS media(key TEXT PRIMARY KEY,etag TEXT,sha TEXT,body BLOB,touched INTEGER)')
                db.execute('CREATE INDEX IF NOT EXISTS media_touched ON media(touched,key)')
                db.execute('CREATE TABLE IF NOT EXISTS manifests(key TEXT PRIMARY KEY,etag TEXT,sha TEXT,size INTEGER,touched INTEGER)')
                db.execute('CREATE INDEX IF NOT EXISTS manifests_touched ON manifests(touched,key)')
                db.execute('BEGIN IMMEDIATE')
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='usage'").fetchone():
                    db.execute('CREATE TABLE usage(bytes INTEGER, manifests INTEGER)')
                    db.execute('INSERT INTO usage SELECT COALESCE(SUM(length(body)),0),(SELECT COUNT(*) FROM manifests) FROM media')
                    for name,table,event,delta in (
                            ('mi','media','INSERT','bytes=bytes+length(new.body)'),
                            ('md','media','DELETE','bytes=bytes-length(old.body)'),
                            ('mu','media','UPDATE OF body','bytes=bytes+length(new.body)-length(old.body)'),
                            ('vi','manifests','INSERT','manifests=manifests+1'),
                            ('vd','manifests','DELETE','manifests=manifests-1')):
                        db.execute(f'CREATE TRIGGER usage_{name} AFTER {event} ON {table} BEGIN UPDATE usage SET {delta}; END')
                key=self.key(url)
                db.execute('DELETE FROM media WHERE key=?',(key,))
                size=db.execute('SELECT bytes FROM usage').fetchone()[0]
                for old,length in db.execute('SELECT key,length(body) FROM media ORDER BY touched,key'):
                    if size+len(body)<=self.limit:break
                    db.execute('DELETE FROM media WHERE key=?',(old,));size-=length
                sha=body.sha256 if isinstance(body,VerifiedBytes) else hashlib.sha256(body).hexdigest()
                db.execute('INSERT INTO manifests VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET etag=excluded.etag,sha=excluded.sha,size=excluded.size,touched=excluded.touched',(key,etag,sha,len(body),time.time_ns()))
                maximum=min(250000,max(1024,self.limit//512))
                if db.execute('SELECT manifests FROM usage').fetchone()[0]>maximum:
                    db.execute('DELETE FROM manifests WHERE key IN (SELECT key FROM manifests ORDER BY touched,key LIMIT ?)',(max(1,maximum//16),))
                db.execute('INSERT INTO media VALUES (?,?,?,?,?)',
                    (key,etag,body.sha256 if isinstance(body,VerifiedBytes) else hashlib.sha256(body).hexdigest(),body,time.time_ns()))
        except (sqlite3.Error,OSError):pass # Recomputable transport optimization only.

    def discard(self,url):
        if not self.path.exists():return
        try:
            with closing(sqlite3.connect(self.path,timeout=5)) as db,db:
                db.execute('DELETE FROM media WHERE key=?',(self.key(url),))
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='manifests'").fetchone():
                    db.execute('DELETE FROM manifests WHERE key=?',(self.key(url),))
        except (sqlite3.Error,OSError):pass
