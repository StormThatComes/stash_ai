"""Whole independent face scenes on bounded CPU workers, committed in order."""
from contextlib import closing
from functools import lru_cache
import sqlite3
from types import SimpleNamespace

from .compute import Work
from .config import plain
from .resource_errors import ResourceLimit


@lru_cache(maxsize=1)
def schema():
    from .database import SCHEMA
    with closing(sqlite3.connect(':memory:')) as db:
        db.executescript(SCHEMA)
        # The private input is immutable. Durable invalidation triggers would
        # add row-write overhead without any consumers in this isolated job.
        return ';\n'.join(r[0] for r in db.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','index') AND sql IS NOT NULL ORDER BY type DESC,rowid"))+';'


def solve(payload):
    if payload is None:return None
    if 'cached' in payload:return payload['cached']
    from .clustering import cluster_scene
    from .replay_checkpoint import ReplayClock
    from .run_cache import scope
    data=payload['settings'];size=payload['bytes']
    data['execution'].update(rebuild_memory_bytes=size,rebuild_cache_bytes=size//4,
        rebuild_matrix_cache_bytes=size//8,rebuild_queue_bytes=0)
    config=SimpleNamespace(data=data)
    with closing(sqlite3.connect(':memory:')) as db,scope(config):
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA cache_size=-1024');db.execute('PRAGMA temp_store=MEMORY')
        db.execute(f'PRAGMA max_page_count={max(128,size//3//4096)}')
        try:
            db.executescript(schema())
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',payload['scan'])
            db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',payload['rows'])
            db.execute('CREATE TEMP TABLE local_core_order(observation TEXT PRIMARY KEY,position INTEGER)')
            db.executemany('INSERT INTO local_core_order VALUES (?,?)',payload['order'])
            db.execute('CREATE TEMP TABLE prior_attachment_queries(id TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO prior_attachment_queries VALUES (?)',payload['prior'])
            db.commit()
            clock=ReplayClock()
            owner=SimpleNamespace(connection=db,installation=payload['installation'],identity_clock=clock)
            cluster_scene(owner,payload['scan'][0],config,_cold_replay=True)
            people=[list(r) for r in db.execute('SELECT * FROM people ORDER BY created,id')]
            if clock.value-1!=len(people):return None
            return [people,[list(r) for r in db.execute('SELECT * FROM candidates ORDER BY rowid')],
                    [list(r) for r in db.execute('SELECT * FROM members ORDER BY rowid')]]
        except ResourceLimit:return None
        except sqlite3.OperationalError as error:
            if getattr(error,'sqlite_errorcode',None)==sqlite3.SQLITE_FULL:return None
            raise


def matches(database,selected,cursor,config,keys,pool,build):
    from . import scene_artifacts,derived_cache
    db=database.connection;prepared={}
    prior=db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='prior_attachment_queries'").fetchone()
    def jobs():
        for ordinal in range(cursor,len(selected)):
            item=selected[ordinal];scan=item['scan']
            key,entry=scene_artifacts.lookup(database,scan,config,keys.get(scan))
            prepared[ordinal]=(key,entry is not None)
            rows,raw=db.execute('SELECT COUNT(*),COALESCE(SUM(length(embedding)+length(geometry)+length(quality)+length(id)+length(scan)),0) '
                                'FROM observations WHERE scan=?',(scan,)).fetchone()
            size=8*1024**2+12*(raw+rows*2048)
            payload=None
            if size<=pool.limit:
                payload=dict(cached=entry) if entry is not None else dict(
                    scan=tuple(db.execute('SELECT * FROM scans WHERE id=?',(scan,)).fetchone()),
                    rows=[tuple(r) for r in db.execute('SELECT id,scan,cue,seconds,source,rect,geometry,quality,rejected,dimensions,embedding,NULL FROM observations WHERE scan=? ORDER BY cue,id',(scan,))],
                    order=[tuple(r) for r in db.execute('SELECT observation,position FROM local_core_order JOIN observations o ON o.id=observation WHERE o.scan=?',(scan,))],
                    prior=[tuple(r) for r in db.execute('SELECT q.id FROM prior_attachment_queries q JOIN observations o ON o.id=q.id WHERE o.scan=?',(scan,))] if prior else [],
                    installation=database.installation,settings=plain(config.data),bytes=size)
            else:pool.fallback('local_scene_admission');size=1024
            yield Work(ordinal,scan,payload,size)
    with closing(pool.ordered(solve,jobs(),backend=config.data['execution']['rebuild_cohort_backend'])) as results:
        for result in results:
            item=selected[result.ordinal];key,reused=prepared.pop(result.ordinal)
            if result.generation!=item['scan']:raise RuntimeError('Stale local-scene result.')
            entry=result.value
            if entry is None:
                pool.fallback('local_scene_working_budget')
                scene_artifacts.match(database,item['scan'],config,keys.get(item['scan']),lambda:build(item))
                reused=False
            else:
                if key is not None and not reused:derived_cache.current().put('strict-scene',key,entry)
                people,candidates,members=entry
                for person in people:
                    person[3]=database.identity_clock()
                    db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',person)
                db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)',candidates)
                db.executemany('INSERT INTO members VALUES (?,?)',members)
                db.commit()
                pool.record('local_face_scenes',computed=not reused,consumed=1)
            yield result.ordinal,item,reused
