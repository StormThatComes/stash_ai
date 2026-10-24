"""Completed-result descriptors and bounded, dependency-checked warm reads."""
import hashlib
import itertools
import json
import sqlite3
import uuid
from contextlib import contextmanager, ExitStack

from stash_ai.runtime.cleanup import cleanup
from .config import digest
from .database import encoded
from .profiling import count, measured
from .progress import work

_RESOURCE_KEYS=('max_reference_bytes','max_people_per_lookup')


def resource_limits(config):
    return {key:config.data['clustering'][key] for key in _RESOURCE_KEYS}


def admissible(state,config):
    limits=state.get('limits')
    return (state.get('format')==2 and isinstance(limits,dict)
            and not state.get('corroboration_pending',False)
            and all(type(limits.get(k)) is int and config.data['clustering'][k]>=limits[k]
                    for k in _RESOURCE_KEYS)
            and (limits==resource_limits(config) or state.get('resource_reuse') is True))


def _hash_row(hasher,row):
    values=list(row)
    blobs=[]
    for index,value in enumerate(values):
        if isinstance(value,bytes):
            blobs.append(value)
            values[index]=dict(bytes=len(value))
    header=encoded(values).encode()
    hasher.update(len(header).to_bytes(8,'little'));hasher.update(header)
    for blob in blobs:
        hasher.update(blob)


@contextmanager
def _assignment_index(db, selected):
    """Read candidate JSON once, retaining only selected observation addresses.

    Rowid order preserves the latest candidate across all policies, including
    candidates whose scan field disagrees with their observation's scan. The
    temporary lookup and its writes belong to a private savepoint, so neither
    success nor interruption commits or rolls back the caller's transaction.
    """
    name = 'rebuild_assignments_' + uuid.uuid4().hex
    selected_scans = {item['scan'] for item in selected}
    db.execute(f'SAVEPOINT {name}')
    try:
        db.execute(f'CREATE TEMP TABLE {name}(observation TEXT PRIMARY KEY, candidate INTEGER NOT NULL)')
        scans=encoded(sorted(selected_scans))
        with work('Index rebuild assignment candidates',
                  lambda: db.execute('SELECT COUNT(*) FROM observations WHERE scan IN '
                      '(SELECT value FROM json_each(?))',(scans,)).fetchone()[0],unit='observation') as indexed:
            # The durable expression index finds the latest decision by actual
            # observation address, regardless of its policy or claimed scan.
            from .reference_roles import observation_sql
            rows=db.execute('SELECT o.id,(SELECT c.rowid FROM candidates c WHERE '+observation_sql('c')+
                '=o.id ORDER BY c.rowid DESC LIMIT 1) FROM observations o WHERE o.scan IN '
                '(SELECT value FROM json_each(?)) ORDER BY o.id',(scans,))
            while batch := list(itertools.islice(rows,256)):
                db.executemany(f'INSERT INTO {name} VALUES (?,?)',((r[0],r[1]) for r in batch if r[1] is not None))
                indexed.advance(len(batch))
        yield name
    finally:
        cleanup([('discard assignment lookup', lambda: db.execute(f'ROLLBACK TO {name}')),
                 ('release assignment lookup', lambda: db.execute(f'RELEASE {name}'))])


@measured('input_scene_dependencies')
def _input_context(db,selected,scope,namespace,config=None,*,scene_progress=True):
    hasher=hashlib.sha256()
    from .pose_policy import anchor_visibility
    visibility_fields=",'$.visible_face_fraction','$.visibility.contract'" if config is not None and anchor_visibility(config.data['clustering']) else ''
    with ExitStack() as stack:
        progress=stack.enter_context(work('Validate rebuild input catalogue',len(selected),unit='scene',position=0)) if scene_progress else None
        assignments_index = None
        for start in range(0,len(selected),256):
            batch=selected[start:start+256]
            scans=encoded([item['scan'] for item in batch])
            records={r['id']:r for r in db.execute('SELECT * FROM scans WHERE id IN '
                '(SELECT value FROM json_each(?))',(scans,))}
            for item in batch:
                row=records.get(item['scan'])
                if (row is None or row['scope']!=scope or row['namespace']!=namespace
                        or row['status']!='complete' or row['scene']!=item['scene']):
                    raise ValueError('Identity rebuild requires complete compatible selected scans.')
                _hash_row(hasher,row)
            # Count only indexed rows for telemetry; keep embeddings streamed in
            # the original digest order, and skip the count when UI is disabled.
            with work('Validate rebuild face observations',lambda: db.execute(
                    'SELECT COUNT(*) FROM observations WHERE scan IN '
                    '(SELECT value FROM json_each(?))',(scans,)).fetchone()[0],
                    unit='observation') as observations:
                for row in db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'
                    "o.rejected,o.dimensions,o.embedding,CASE WHEN json_valid(o.quality) THEN "
                    "json_extract(o.quality,'$.pixels','$.sharpness','$.pose_deviation','$.roll_deviation','$.alignment_valid','$.alignment_error','$.alignment_face_coverage'"+visibility_fields+") END "
                    'FROM observations o JOIN scans s ON s.id=o.scan '
                    'WHERE o.scan IN (SELECT value FROM json_each(?)) '
                    'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',(scans,)):
                    _hash_row(hasher,row)
                    observations.advance()
            if config is not None:
                from .corroborated_views import enabled
                if enabled(config):
                    if not scene_progress:
                        from .reference_roles import observation_sql
                        join='LEFT JOIN candidates c ON c.rowid=(SELECT z.rowid FROM candidates z WHERE '+observation_sql('z')+'=o.id ORDER BY z.rowid DESC LIMIT 1) '
                    else:
                        if assignments_index is None:
                            assignments_index = stack.enter_context(_assignment_index(db, selected))
                        join=f'LEFT JOIN {assignments_index} a ON a.observation=o.id LEFT JOIN candidates c ON c.rowid=a.candidate '
                    with work('Validate rebuild assignment evidence',observations.total,
                            unit='observation') as assignments:
                        for row in db.execute('SELECT o.id,c.person,c.status,CASE WHEN json_valid(c.payload) THEN '
                                "json_extract(c.payload,'$.reason','$.strict_confirmation') END FROM observations o "
                                + join +
                                'WHERE o.scan IN (SELECT value FROM json_each(?)) ORDER BY o.id',
                                (scans,)):
                            _hash_row(hasher,row)
                            assignments.advance()
            if progress is not None:progress.advance(len(batch))
    return hasher.hexdigest()


@measured('input_dependencies')
def input_context(db,selected,scope,namespace,config=None):
    from .dependencies import cached,supported,snapshot
    from .dependency_schema import validate_triggers
    if not supported(db):return _input_context(db,selected,scope,namespace,config)
    from .clustering import matching_policy
    validate_triggers(db,required=True)
    policy=matching_policy(config) if config is not None else 'no-assignment-evidence'
    hasher=hashlib.sha256(b'per-scene-rebuild-input-v1')
    with snapshot(db), work('Validate rebuild input catalogue',len(selected),unit='scene',position=0) as progress:
        for item in selected:
            key=item['scan']
            # Scope/scene are part of the key even for a cached input digest.
            fingerprint=cached(db,'scene-input',key,digest([scope,namespace,item['scene'],policy]),
                lambda:_input_context(db,[item],scope,namespace,config,scene_progress=False),
                dependency='scan',dependency_key=key,checked=True)
            _hash_row(hasher,(key,item['scene'],fingerprint))
            progress.advance()
    return hasher.hexdigest()


def binding_context(db,scope,namespace):
    from .dependencies import cached
    return cached(db,'bindings',digest([scope,namespace]),'binding-v1',
                  lambda:_binding_context(db,scope,namespace),dependency='bindings')


def _binding_context(db,scope,namespace):
    hasher=hashlib.sha256()
    for row in db.execute('SELECT id,created,token FROM people WHERE scope=? AND namespace=? ORDER BY id',
                          (scope,namespace)):
        _hash_row(hasher,row)
    for row in db.execute('SELECT m.source,m.target FROM identity_merges m JOIN people p ON p.id=m.source '
                          'WHERE p.scope=? AND p.namespace=? ORDER BY m.source',(scope,namespace)):
        _hash_row(hasher,row)
    # Origins have no scope column. Any origin repair conservatively invalidates
    # cached label binding, including changes outside the selected subset.
    for row in db.execute('SELECT * FROM identity_origins ORDER BY sample_key'):
        _hash_row(hasher,row)
    return hasher.hexdigest()


def lookup(db,scope,namespace,fingerprint):
    prefix='identity-rebuild:'+digest([scope,namespace])
    latest=db.execute('SELECT value FROM metadata WHERE key=?',(prefix,)).fetchone()
    if latest is None:
        return None
    try:
        state=json.loads(latest[0])
        if isinstance(state,dict) and state.get('fingerprint')==fingerprint:
            return state
        row=db.execute('SELECT value FROM metadata WHERE key=?',
            ('identity-rebuild-cache:'+digest([scope,namespace])+':'+fingerprint,)).fetchone()
        state=json.loads(row[0]) if row else None
        return state if isinstance(state,dict) and state.get('fingerprint')==fingerprint else None
    except (ValueError,TypeError):
        return None


def save(db,scope,namespace,state):
    payload=encoded(state)
    prefix=digest([scope,namespace])
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('identity-rebuild:'+prefix,payload))
    cache_prefix='identity-rebuild-cache:'+prefix+':'
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(cache_prefix+state['fingerprint'],payload))
    db.execute('DELETE FROM metadata WHERE key IN (SELECT key FROM metadata '
        'WHERE key LIKE ? ORDER BY rowid DESC LIMIT -1 OFFSET 8)',(cache_prefix+'%',))
    from .dependencies import certify
    certify(db,scope,namespace,state)


def validate_vectors(rows):
    """Native acceptance only well inside scalar finite/unit-norm boundaries."""
    import numpy as np
    from .clustering import vector
    if not rows:
        return
    dimension=rows[0]['dimensions']
    shape_ok=(type(dimension) is int and 1<=dimension<=1024 and all(
        type(r['dimensions']) is int and r['dimensions']==dimension
        and isinstance(r['embedding'],bytes) and len(r['embedding'])==4*dimension for r in rows))
    if shape_ok:
        matrix=np.stack([np.frombuffer(r['embedding'],dtype='<f4') for r in rows])
        norms=np.linalg.norm(matrix,axis=1)
        margin=4*dimension*np.finfo(np.float32).eps
        if np.isfinite(matrix).all() and np.isfinite(norms).all() and (np.abs(norms-1)<=.001-margin).all():
            count('validated_vectors',len(rows));count('batch_validated_vectors',len(rows))
            return
    # Same original scalar error and first bad record for ambiguous boundaries.
    for row in rows:
        vector(row)


@measured('cached_candidate_integrity')
def complete(db,selected,policy,context):
    performers={}
    try:
        for start in range(0,len(selected),256):
            ids=encoded([item['scan'] for item in selected[start:start+256]])
            remaining=dict(db.execute('SELECT scan,COUNT(*) FROM observations WHERE rejected IS NULL '
                'AND scan IN (SELECT value FROM json_each(?)) GROUP BY scan',(ids,)))
            rows=db.execute("SELECT c.id,c.person,c.status,c.scan,json_extract(c.payload,'$.observation') AS observation,"
                "json_extract(c.payload,'$.reference_context') AS context,o.scan AS actual_scan,o.rejected,"
                'o.dimensions,o.embedding,m.person AS member,p.performer,p.scope AS member_scope,'
                'p.namespace AS member_namespace,s.scope AS scan_scope,s.namespace AS scan_namespace FROM candidates c '
                "LEFT JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
                'LEFT JOIN scans s ON s.id=o.scan LEFT JOIN members m ON m.observation=o.id '
                'LEFT JOIN people p ON p.id=m.person '
                'WHERE c.policy=? AND c.scan IN (SELECT value FROM json_each(?)) '
                'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',(policy,ids))
            with work('Validate cached face observations',sum(remaining.values()),unit='observation') as checked:
                block=[];size=0
                for row in rows:
                    try:
                        expected=str(uuid.uuid5(uuid.UUID(row['observation']),policy))
                    except (ValueError,TypeError,AttributeError):
                        return False
                    if (row['id']!=expected or row['scan']!=row['actual_scan'] or row['rejected'] is not None
                            or row['context']!=context or row['status'] not in ('accepted','withheld')):
                        return False
                    if row['status']=='accepted':
                        if (row['person'] is None or row['person']!=row['member']
                                or row['member_scope']!=row['scan_scope']
                                or row['member_namespace']!=row['scan_namespace']):
                            return False
                        if row['performer']:
                            old=performers.setdefault(row['performer'],row['person'])
                            if old!=row['person']:
                                return False
                    elif row['member'] is not None:
                        return False
                    remaining[row['scan']]=remaining.get(row['scan'],0)-1
                    row_bytes=len(row['embedding']) if isinstance(row['embedding'],bytes) else 0
                    if block and (len(block)>=128 or size+row_bytes>1048576):
                        validate_vectors(block);checked.advance(len(block));block=[];size=0
                    block.append(row);size+=row_bytes
                validate_vectors(block);checked.advance(len(block))
            if any(remaining.values()):
                return False
    except sqlite3.Error:
        # Candidate payloads are disposable derived state, never trusted input.
        return False
    return True
