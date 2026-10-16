"""Completed-result descriptors and bounded, dependency-checked warm reads."""
import hashlib
import json
import sqlite3
import uuid

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


@measured('input_dependencies')
def input_context(db,selected,scope,namespace):
    hasher=hashlib.sha256()
    with work('Validate rebuild input catalogue',len(selected),unit='scene',position=0) as progress:
        for start in range(0,len(selected),256):
            batch=selected[start:start+256]
            records={r['id']:r for r in db.execute('SELECT * FROM scans WHERE id IN '
                '(SELECT value FROM json_each(?))',(encoded([item['scan'] for item in batch]),))}
            for item in batch:
                row=records.get(item['scan'])
                if (row is None or row['scope']!=scope or row['namespace']!=namespace
                        or row['status']!='complete' or row['scene']!=item['scene']):
                    raise ValueError('Identity rebuild requires complete compatible selected scans.')
                _hash_row(hasher,row)
            for row in db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'
                'o.rejected,o.dimensions,o.embedding FROM observations o JOIN scans s ON s.id=o.scan '
                'WHERE o.scan IN (SELECT value FROM json_each(?)) '
                'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',
                (encoded([item['scan'] for item in batch]),)):
                _hash_row(hasher,row)
            progress.advance(len(batch))
    return hasher.hexdigest()


def binding_context(db,scope,namespace):
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
