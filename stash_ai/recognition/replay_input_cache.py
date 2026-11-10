"""Optional reviewed raw replay image, reused before importing grouping checkpoints."""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .config import digest
from .profiling import count
from .replay_checkpoint import _lease,_json,_exclusive_json,_sync_directory


def schema(db):
    return [tuple(r) for r in db.execute('SELECT type,name,tbl_name,sql FROM main.sqlite_master ORDER BY type,name')]


@lru_cache(maxsize=1)
def expected_schema():
    from .rebuild import REPLAY_SCHEMA
    with closing(sqlite3.connect(':memory:')) as db:
        db.executescript(REPLAY_SCHEMA)
        return schema(db)


def checksum(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):value.update(block)
    return value.hexdigest()


def raw(db):
    while not isinstance(db,sqlite3.Connection):db=db.raw
    return db


class InputCache:
    def __init__(self,database,selected,config,scope,namespace):
        self.database,self.config=database,config
        self.enabled=False;self.reused=0;self.replace_invalid=False
        if not config.data['execution']['rebuild_checkpoint_scenes']:return
        try:
            from .database import FaceDatabase
            from .dependencies import version
            from .replay_checkpoint import ENGINE
            from .dependency_schema import validate_triggers
            if not isinstance(database,FaceDatabase) or database.lock is None:return
            validate_triggers(database.connection,required=True)
            self.limit=config.data['execution']['replay_snapshot_bytes']
            self.owner=dict(format=1,installation=database.installation,path=str(database.path))
            stamps=[(item['scan'],version(database.connection,'raw-observations',item['scan']),
                     version(database.connection,'pixel-scan',item['scan'])) for item in selected]
            self.schema_stamp=digest([ENGINE,expected_schema()])
            self.stamps={scan:(raw,pixel) for scan,raw,pixel in stamps}
            journal = database.connection.execute('PRAGMA main.journal_mode').fetchone()[0]
            self.key=digest(['raw-input-v3-live-vector-locks',journal,ENGINE,self.owner,selected,scope,namespace,stamps,
                digest(expected_schema()),config.data['clustering']['max_anchor_pose_deviation'] is not None])
            self.directory=database.path.with_name(database.path.name+'.rebuild-input')
            self.directory.mkdir(mode=0o700,exist_ok=True)
            if self.directory.is_symlink():raise ValueError('Unowned raw replay cache directory.')
            marker=self.directory/'owner.json'
            if not marker.exists():
                if any(self.directory.iterdir()):raise ValueError('Unowned raw replay cache files.')
                _exclusive_json(marker,self.owner)
            self.marker,self.marker_lease=_json(marker)
            if self.marker!=self.owner:raise ValueError('Foreign raw replay cache.')
            self.directory_lease=(self.directory.stat().st_dev,self.directory.stat().st_ino)
            self.enabled=True
        except Exception:count('input_cache_failures')
    def owned(self):
        if (self.directory.is_symlink() or (self.directory.stat().st_dev,self.directory.stat().st_ino)!=self.directory_lease
                or _json(self.directory/'owner.json')!=(self.owner,self.marker_lease)):
            raise ValueError('Raw replay cache ownership changed.')
    def restore(self,replay):
        if not self.enabled:return False
        try:
            self.owned()
            manifest,_=_json(self.directory/'current.json')
            if manifest.get('key')!=self.key:return False
            self.replace_invalid = True
            name = manifest['file']
            if name!=str(uuid.UUID(name.removesuffix('.sqlite3')))+'.sqlite3':raise ValueError('Invalid cache filename.')
            path=self.directory/name;lease=_lease(path)
            if lease[2]>self.limit or checksum(path)!=manifest['checksum']:raise ValueError('Invalid raw replay checksum.')
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as saved:
                if schema(saved)!=expected_schema():raise ValueError('Unreviewed raw replay schema.')
                from .replay_vector_inputs import KEY
                source=saved.execute('SELECT value FROM metadata WHERE key=?',(KEY,)).fetchone()
                if source and Path(source[0])!=self.database.path:raise ValueError('Changed vector source.')
                if lease!=_lease(path):raise ValueError('Raw replay image changed.')
                target=raw(replay)
                # SQLite backup bypasses Connection.execute; admit its complete size.
                if getattr(target,'budget',None):
                    if hasattr(target,'account_pages'):target.budget.reserve(target.owner,lease[2])
                    else:
                        target.budget.set(target,lease[2]*2);target.capacity=target.estimate=lease[2]
                saved.backup(target)
            from .replay_vector_inputs import install
            install(replay)
            count('input_snapshot_restores');return True
        except Exception:
            count('input_cache_rejections')
            # Validation precedes backup. A failed backup must not be used as a
            # partially copied cold replay; the caller gets a fresh empty schema.
            target=raw(replay)
            if target.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
                raise
            return False
    def save(self,replay):
        if not self.enabled:return
        path=None
        try:
            self.owned();source=raw(replay)
            if source.in_transaction:raise ValueError('Raw replay cache requires a committed image.')
            size=source.execute('PRAGMA page_count').fetchone()[0]*source.execute('PRAGMA page_size').fetchone()[0]
            shards=self.directory/'shards.sqlite3'
            if size+(shards.stat().st_size if shards.exists() else 0)>self.limit:return
            from .disk_budget import check
            check(size,self.directory)
            path=self.directory/(str(uuid.uuid4())+'.sqlite3')
            descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(descriptor)
            with closing(sqlite3.connect(path)) as target:source.backup(target)
            with path.open('r+b') as stream:os.fsync(stream.fileno())
            manifest=dict(key=self.key,file=path.name,checksum=checksum(path))
            stage=self.directory/(str(uuid.uuid4())+'.json')
            _exclusive_json(stage,manifest);self.owned()
            previous=None
            current=self.directory/'current.json'
            if current.exists():previous,_=_json(current)
            os.replace(stage,current);_sync_directory(self.directory)
            if previous:
                old=previous.get('file','')
                try:
                    if old==str(uuid.UUID(old.removesuffix('.sqlite3')))+'.sqlite3':
                        old_path=self.directory/old
                        if old_path!=path:_lease(old_path);old_path.unlink()
                except (ValueError,OSError):pass
            count('input_snapshot_saves')
        except Exception:
            count('input_cache_failures')
            if path is not None:path.unlink(missing_ok=True)
