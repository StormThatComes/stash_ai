"""Scene-local appearance priorities on isolated CPU jobs, applied in order.

A worker reads immutable shared arrays directly for the same streaming
cohort algorithm. It receives only selected observation values, never a durable
connection or person labels. Oversize scenes retain the bounded serial path.
"""

from .compute import Work
from .config import plain
from .pose_policy import query_sql
from .progress import tracking
from stash_ai.runtime.cleanup import closing

FIELDS = 'id,scan,cue,seconds,geometry,quality,dimensions,embedding'


def _scene(payload):
    if payload is None:
        return None
    item, rows, settings = payload
    from .shared_payload import borrowed
    from .cohort_arrays import priorities
    with borrowed(rows) as values:
        try:
            return priorities(values,settings)
        except BaseException as error:
            # Drop NumPy buffer views held by completed error frames before
            # closing the shared segment, preserving the original exception.
            import traceback
            traceback.clear_frames(error.__traceback__)
            raise


def prepare_parallel(db, selected, config, pool):
    from .local_cohorts import _prepare_serial
    db.execute('CREATE TEMP TABLE local_core_order(observation TEXT PRIMARY KEY,position INTEGER)')
    condition = 'WHERE scan=? AND rejected IS NULL AND '+query_sql('observations', config.data['clustering'])
    settings = {k: plain(config.data[k]) for k in ('clustering', 'recognition')}
    def jobs():
        for ordinal, item in enumerate(selected):
            # Includes Python rows, queue serialization, private SQLite, witness
            # arrays, local assignments and returned priorities. No crop bytes.
            count, raw = db.execute('SELECT COUNT(*),COALESCE(SUM(length(embedding)+length(geometry)+'
                'length(quality)+length(id)+length(scan)),0) FROM observations '+condition, (item['scan'],)).fetchone()
            size = 4096 + 12*(raw + count*2048)
            payload = None
            if size <= pool.limit:
                rows = tuple(tuple(r) for r in db.execute('SELECT '+FIELDS+' FROM observations '+condition+' ORDER BY embedding,geometry,cue,id',
                                                         (item['scan'],)))
                if backend == 'processes' and pool.workers > 1:
                    from .shared_payload import pack
                    rows, segment = pack(rows)
                    if segment is not None:
                        pool.shared_inputs[ordinal] = segment
                        pool.record('shared_input_bytes', computed=segment.size)
                payload = (dict(item), rows, settings)
            else:
                pool.fallback('cohort_scene_admission')
                size = 1024
            yield Work(ordinal, item['scan'], payload, size)
    backend = config.data['execution']['rebuild_cohort_backend']
    with closing(pool.ordered(_scene, jobs(), backend=backend), name='close local cohort jobs') as stream:
        with tracking(stream, 'Freeze local appearance cohorts', total=len(selected), unit='scene', position=0) as completed:
            for result in completed:
                item = selected[result.ordinal]
                if result.generation != item['scan']:
                    raise RuntimeError('Stale local appearance cohort result.')
                if result.value is None:
                    _prepare_serial(db, [item], config)
                else:
                    db.executemany('INSERT INTO local_core_order VALUES (?,?)', result.value)
                    pool.record('local_cohort_scenes', computed=1, consumed=1)
