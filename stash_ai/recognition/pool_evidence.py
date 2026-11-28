"""Bounded transfer between the retained source and the private claim tables."""
from itertools import islice

from .database import encoded
from .query_support import cohorts


def claim_cohorts(source, replay, table, historical, config, stronger):
    from .decision_resolution import conflicting_claims
    owner = 'owner' if historical else 'canonical'
    replay.execute('DROP TABLE IF EXISTS temp.frozen_pool_rows')
    replay.execute('CREATE TEMP TABLE frozen_pool_rows(id TEXT PRIMARY KEY,scan,seconds,cue,geometry,quality,dimensions,embedding,owner,mode,prototype,person)')
    rows = replay.execute(f'SELECT * FROM {table} ORDER BY observation')
    while batch := list(islice(rows, 256)):
        claims = {r['observation']:r for r in batch}
        raw = source.execute('SELECT id,scan,seconds,cue,geometry,quality,dimensions,embedding FROM observations WHERE id IN (SELECT value FROM json_each(?))',(encoded(list(claims)),))
        for r in raw:
            claim = claims[r['id']]
            replay.execute('INSERT INTO frozen_pool_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (*r, claim['owner'], claim['mode'] if historical else '', claim['medoids' if historical else 'medoid'], claim[owner]))
    replay.execute('CREATE INDEX temp.frozen_family ON frozen_pool_rows(owner,mode,embedding,geometry,id)')
    conflicts = replay.execute('SELECT * FROM frozen_pool_rows ORDER BY person,scan,cue,id')
    for r in conflicting_claims(conflicts,stronger,deduplicate=historical and config.data['clustering'].get('deduplicate_detections',False)):
        replay.execute(f'DELETE FROM {table} WHERE observation=?',(r['id'],))
    replay.execute(f'DELETE FROM frozen_pool_rows WHERE NOT EXISTS (SELECT 1 FROM {table} WHERE observation=frozen_pool_rows.id)')
    for family,mode in replay.execute('SELECT DISTINCT owner,mode FROM frozen_pool_rows ORDER BY owner,mode'):
        key = (family,mode) if historical else family
        def stream():
            return replay.execute('SELECT * FROM frozen_pool_rows WHERE owner=? AND mode=? ORDER BY embedding,geometry,id',(family,mode))
        minimum = 1 if historical and mode!='view' else config.data['clustering']['view_attach_min_support']
        yield from cohorts(stream,config,key=lambda row:key,prototype=lambda row:row['prototype'],minimum=lambda _:minimum)
    replay.execute('DROP TABLE frozen_pool_rows')


def compact(replay, proof):
    from hashlib import sha256
    lists = {key:proof[key] for key in ('query_ids','query_support_ids') if key in proof}
    if not lists:
        return proof
    raw = encoded(lists)
    key = sha256(raw.encode()).hexdigest()
    replay.execute('CREATE TEMP TABLE IF NOT EXISTS pool_witness_context(id TEXT PRIMARY KEY,payload TEXT)')
    replay.execute('INSERT OR IGNORE INTO pool_witness_context VALUES (?,?)',(key, raw))
    return dict(((k,v) for k,v in proof.items() if k not in lists), witness_context=key)
