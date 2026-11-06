"""Disposable bounded HTTP cache; only strong origin validators permit reuse."""
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import time


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
                row=db.execute('SELECT etag,sha,length(body) FROM media WHERE key=?',(self.key(url),)).fetchone()
                if row and strong_etag(row[0]) and isinstance(row[1],str) and type(row[2]) is int and 0<=row[2]<=self.limit:
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
                db.execute('BEGIN IMMEDIATE')
                key=self.key(url)
                db.execute('DELETE FROM media WHERE key=?',(key,))
                size=db.execute('SELECT COALESCE(SUM(length(body)),0) FROM media').fetchone()[0]
                for old,length in db.execute('SELECT key,length(body) FROM media ORDER BY touched,key'):
                    if size+len(body)<=self.limit:break
                    db.execute('DELETE FROM media WHERE key=?',(old,));size-=length
                db.execute('INSERT INTO media VALUES (?,?,?,?,?)',
                    (key,etag,body.sha256 if isinstance(body,VerifiedBytes) else hashlib.sha256(body).hexdigest(),body,time.time_ns()))
        except (sqlite3.Error,OSError):pass # Recomputable transport optimization only.

    def discard(self,url):
        if not self.path.exists():return
        try:
            with closing(sqlite3.connect(self.path,timeout=5)) as db,db:
                db.execute('DELETE FROM media WHERE key=?',(self.key(url),))
        except (sqlite3.Error,OSError):pass
