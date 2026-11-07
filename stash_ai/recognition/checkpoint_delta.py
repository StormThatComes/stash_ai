"""Immutable strict-prefix deltas with atomic pointer publication.

The first generation is a base. Later generations contain changed rows and
deletions captured by transaction-owned TEMP triggers. At most 16 generations
are replayed; a new base compacts the chain without modifying its predecessor.
"""
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import uuid

from . import replay_checkpoint as base
from .database import encoded
from .profiling import count

KEYS=dict(zip(base.SNAPSHOT_TABLES,('id','observation','id','source',
    'source,target,observation,policy','observation','observation','source')))
SCHEMA=base.SCHEMA+'CREATE TABLE deletions(name TEXT NOT NULL,address TEXT NOT NULL,PRIMARY KEY(name,address));'


def review(db,*,integrity=True):
    if db.execute('PRAGMA application_id').fetchone()[0]!=base.APPLICATION_ID or db.execute('PRAGMA user_version').fetchone()[0]!=2:
        raise ValueError('Unsupported incremental checkpoint.')
    with closing(sqlite3.connect(':memory:')) as expected:
        expected.executescript(SCHEMA)
        if base._schema(db)!=base._schema(expected):raise ValueError('Invalid incremental checkpoint schema.')
    if integrity and db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Damaged incremental checkpoint.')


def checksum(db,header):
    h=hashlib.sha256(base._checksum(db,header).encode())
    for row in db.execute('SELECT * FROM deletions ORDER BY name,address'):base.rebuild_cache._hash_row(h,row)
    return h.hexdigest()


def exists(db,table):
    schema='sqlite_temp_master' if table in base.EXTRA_SCHEMA else 'sqlite_master'
    return db.execute(f'SELECT 1 FROM {schema} WHERE name=?',(table,)).fetchone() is not None


def bounded(rows):
    batch=[];size=0
    for row in rows:
        length=len(encoded(tuple(row)).encode())
        if length>base._BATCH_BYTES:raise ValueError('Oversized checkpoint row.')
        if batch and (len(batch)>=256 or size+length>base._BATCH_BYTES):
            yield batch;batch=[];size=0
        batch.append(tuple(row));size+=length
    if batch:yield batch


class DeltaCheckpoints(base.Checkpoints):
    suffix='-delta'

    def __init__(self,*args,**kwargs):
        self.chain=[];self.journal=set();self.valid=False
        super().__init__(*args,**kwargs)

    def _walk(self,*,verify=False):
        pointer=self.manifest;chain=[];seen=set();size=0
        while pointer:
            name=pointer['generation']
            if not base._NAME.fullmatch(name) or name in seen or len(chain)>=16:
                raise ValueError('Invalid incremental checkpoint ancestry.')
            seen.add(name);path=self.directory/name;lease=base._lease(path);size+=lease[2]
            if size>self.limit:raise ValueError('Incremental checkpoint chain exceeds its disk budget.')
            with closing(self._open(path)) as db:
                review(db,integrity=verify);header=base._header(db)
                if header.get('owner')!=self.owner or header.get('generation')!=name[7:-8]:
                    raise ValueError('Foreign incremental checkpoint generation.')
                if verify and checksum(db,header)!=pointer['checksum']:
                    raise ValueError('Incremental checkpoint checksum mismatch.')
                if lease!=base._lease(path):raise ValueError('Incremental checkpoint changed while reading.')
                chain.append((path,header,lease))
                pointer=header.get('parent')
        return list(reversed(chain))

    def _prune(self):
        try:keep={p.name for p,_,_ in self._walk()}
        except (ValueError,OSError,KeyError,sqlite3.Error):return
        for path in self.directory.glob('replay-*.sqlite3'):
            if path.name in keep or not base._NAME.fullmatch(path.name):continue
            try:
                lease=base._lease(path)
                with closing(self._open(path)) as db:
                    review(db,integrity=False);header=base._header(db)
                if header.get('owner')==self.owner and header.get('generation')==path.name[7:-8] and lease==base._lease(path):path.unlink()
            except (ValueError,OSError,KeyError,sqlite3.Error):pass

    def _journal(self,replay):
        replay.execute('CREATE TEMP TABLE IF NOT EXISTS checkpoint_dirty(name TEXT,address TEXT,PRIMARY KEY(name,address))')
        for table,key in KEYS.items():
            if not exists(replay,table):continue
            if table in self.journal and replay.execute("SELECT 1 FROM sqlite_temp_master WHERE name=?",('checkpoint_dirty_'+table+'_insert',)).fetchone():continue
            columns=key.split(',');schema='temp' if table in base.EXTRA_SCHEMA else 'main'
            # A table introduced after the base contributes every existing row.
            replay.execute(f"INSERT OR IGNORE INTO checkpoint_dirty SELECT ?,json_array({key}) FROM {table}",(table,))
            for event,sides in (('INSERT',('new',)),('DELETE',('old',)),('UPDATE',('old','new'))):
                commands=''.join(f"INSERT OR IGNORE INTO checkpoint_dirty VALUES ('{table}',json_array({','.join(side+'.'+c for c in columns)}));" for side in sides)
                replay.execute(f'CREATE TEMP TRIGGER checkpoint_dirty_{table}_{event.lower()} AFTER {event} ON {schema}.{table} BEGIN {commands} END')
            if table=='people':
                replay.execute("CREATE TEMP TRIGGER checkpoint_dirty_people_replace BEFORE INSERT ON main.people BEGIN INSERT OR IGNORE INTO checkpoint_dirty SELECT 'people',json_array(id) FROM people WHERE token=new.token; END")
            self.journal.add(table)
        replay.commit()

    def _restore(self,replay):
        self._owned();chain=self._walk(verify=True)
        header=chain[-1][1];cursor,clock=self._compatible(header)
        try:
            replay.execute('BEGIN')
            replay.execute('PRAGMA defer_foreign_keys=ON')
            for path,part,lease in chain:
                if part['cursor']>cursor or part['selected']!=header['selected'][:len(part['selected'])]:
                    raise ValueError('Inconsistent incremental checkpoint prefix.')
                with closing(self._open(path)) as saved:
                    for table in reversed(base.SNAPSHOT_TABLES):
                        if not exists(replay,table):continue
                        key=KEYS[table].split(',')
                        for name,address in saved.execute('SELECT * FROM deletions WHERE name=?',(table,)):
                            values=json.loads(address)
                            if not isinstance(values,list) or len(values)!=len(key):raise ValueError('Invalid checkpoint deletion.')
                            replay.execute(f'DELETE FROM {table} WHERE '+ ' AND '.join(c+'=?' for c in key),values)
                    for table,order in zip(base.SNAPSHOT_TABLES,base.ORDER):
                        if table in base.EXTRA_SCHEMA and not exists(replay,table):
                            if not saved.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone():continue
                            replay.execute(f'CREATE TEMP TABLE {table}({base.EXTRA_SCHEMA[table]})')
                        columns=[r[1] for r in saved.execute(f'PRAGMA table_info({table})')]
                        update=','.join(c+'=excluded.'+c for c in columns if c not in KEYS[table].split(','))
                        suffix=' DO UPDATE SET '+update if update else ' DO NOTHING'
                        for batch in base._rows(saved,table,order):
                            replay.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in columns)}) ON CONFLICT({KEYS[table]})'+suffix,batch)
                if lease!=base._lease(path):raise ValueError('Incremental checkpoint changed during restore.')
            self._validate(replay,cursor,clock)
            replay.commit()
        except BaseException:
            replay.rollback();raise
        self.chain=chain;self.valid=True;self.saved_cursor=cursor
        self._journal(replay);replay.execute('DELETE FROM checkpoint_dirty');replay.commit()
        count('checkpoint_resumed_scenes',cursor);count('checkpoint_delta_restores',len(chain))
        return cursor,clock

    def _save(self,replay,cursor,next_clock):
        previous=self.manifest
        self._staging_delta=False
        try:return self._save_once(replay,cursor,next_clock)
        except sqlite3.OperationalError as error:
            # A larger-than-expected delta may overflow its remaining chain
            # share although a new base fits. SQL_FULL rolls back the private
            # stage; its immutable parent remains usable until pointer swap.
            if (getattr(error,'sqlite_errorcode',None)!=sqlite3.SQLITE_FULL or
                    not self._staging_delta or self.manifest is not previous):raise
            return self._save_once(replay,cursor,next_clock,force_base=True)

    def _save_once(self,replay,cursor,next_clock,*,force_base=False):
        if replay.in_transaction:raise ValueError('Only committed cues can be checkpointed.')
        self._owned()
        if self.valid:self._journal(replay)
        chain_bytes=sum(lease[2] for _,_,lease in self.chain)
        # Compact before the next delta can exhaust the retained chain quota.
        # The first generation is a full base, not a prediction of the next delta.
        last_bytes=self.chain[-1][2][2] if len(self.chain)>1 else 0
        compact=(not self.valid or len(self.chain)>=16 or
                 chain_bytes+max(last_bytes,32*4096)>=self.limit)
        compact=compact or force_base
        self._staging_delta=not compact
        parent=None if compact else dict(generation=self.manifest['generation'],checksum=self.manifest['checksum'])
        if self.input_hash is None:
            self.input_hash=base.rebuild_cache.input_context(self.database.connection,self.selected,self.scope,self.namespace,self.config)
        generation=uuid.uuid4().hex
        header=dict(format=base.FORMAT,engine=base.ENGINE,owner=self.owner,generation=generation,
            scope=self.scope,namespace=self.namespace,selected=self.selected,policy=base.matching_policy(self.config),
            inputs=self.input_hash,limits=base.rebuild_cache.resource_limits(self.config),cursor=cursor,next_clock=next_clock,parent=parent)
        if len(encoded(header).encode())>base._MAX_META:raise ValueError('Oversized checkpoint header.')
        path=self.directory/f'replay-{generation}.sqlite3';stage=self.directory/f'.manifest-{generation}.json'
        previous_bytes=sum(lease[2] for _,_,lease in self.chain) if parent else 0
        from .disk_budget import reserve_file,check
        published=False
        try:
            with path.open('xb'):pass
            with reserve_file(path,self.limit-previous_bytes) as grant, closing(sqlite3.connect(path)) as target:
                target.execute('PRAGMA journal_mode=OFF');target.execute('PRAGMA synchronous=OFF')
                page=target.execute('PRAGMA page_size').fetchone()[0]
                if grant<page*32:raise ValueError('Incremental checkpoint chain budget exhausted.')
                target.execute(f'PRAGMA max_page_count={grant//page}')
                target.executescript(SCHEMA)
                target.execute(f'PRAGMA application_id={base.APPLICATION_ID}');target.execute('PRAGMA user_version=2')
                target.execute('INSERT INTO metadata VALUES (?,?)',('state',encoded(header)))
                written=0
                for table,order in zip(base.SNAPSHOT_TABLES,base.ORDER):
                    if not exists(replay,table):continue
                    if compact:
                        batches=base._rows(replay,table,order)
                    else:
                        join=' AND '.join(f"t.{c}=json_extract(d.address,'$[{i}]')" for i,c in enumerate(KEYS[table].split(',')))
                        target.executemany('INSERT INTO deletions VALUES (?,?)',replay.execute(
                            f'SELECT name,address FROM checkpoint_dirty d WHERE name=? AND NOT EXISTS (SELECT 1 FROM {table} t WHERE {join})',(table,)))
                        rows=replay.execute(f'SELECT t.* FROM checkpoint_dirty d JOIN {table} t ON {join} WHERE d.name=? ORDER BY d.address',(table,))
                        batches=bounded(rows)
                    for batch in batches:
                        base._check_cancel();written+=len(batch)
                        target.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in batch[0])})',batch)
                target.commit();stamp=checksum(target,header)
            check(0,self.directory)
            with path.open('r+b') as stream:os.fsync(stream.fileno())
            base._sync_directory(self.directory)
            manifest=dict(owner=self.owner,generation=path.name,checksum=stamp)
            base._exclusive_json(stage,manifest);base._check_cancel();self._owned()
            if self.manifest is None:os.link(stage,self.current);stage.unlink()
            else:os.replace(stage,self.current)
            published=True;self.manifest=manifest;self.manifest_lease=base._lease(self.current)
            base._sync_directory(self.directory)
            self.chain=([] if compact else self.chain)+[(path,header,base._lease(path))]
            self.saved_cursor=cursor;self.valid=True
            self._journal(replay);replay.execute('DELETE FROM checkpoint_dirty');replay.commit()
            count('checkpoint_delta_rows_written',written);count('checkpoint_saved_scenes',cursor)
            count('checkpoint_delta_generations',0 if compact else 1)
            if compact:self._prune()
        finally:
            stage.unlink(missing_ok=True)
            if not published:path.unlink(missing_ok=True)
