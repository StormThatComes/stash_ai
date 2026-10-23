"""Bounded, deterministic quality/diversity sampling of strict members.

At most twice the reference cap is materialized. Repeated reductions retain
appearance and scene coverage rather than permanently freezing the first N cues.
The returned order is canonical so scalar score reductions remain reproducible.
"""
from collections import Counter
import json
import math


def quality(raw):
    try:
        record = json.loads(raw)
        def number(key):
            value = record.get(key, 0.)
            return max(0., float(value)) if type(value) in (int, float) and math.isfinite(value) else 0.
        return (min(1., math.log1p(number('pixels')) / math.log1p(512)) +
                min(1., math.log1p(number('sharpness')) / math.log1p(1000)) +
                1. / (1. + number('pose_deviation'))) / 3
    except (ValueError, TypeError, AttributeError):
        return 0.


def select(rows, maximum):
    import numpy as np
    from .clustering import vector
    from .view_prototypes import pose_view
    pool, scene = [], None

    def reduce():
        nonlocal pool
        if len(pool) <= maximum:
            return
        if len({sample.shape for _, _, sample, _, _ in pool}) != 1:
            raise ValueError('Face namespace contains mixed embedding dimensions.')
        matrix = np.stack([item[2] for item in pool])
        chosen, scenes, poses = [], Counter(), Counter()
        nearest = np.full(len(pool), -1., dtype=np.float64)
        for _ in range(maximum):
            def priority(i):
                order, row, _, score, pose = pool[i]
                novelty = 0. if not chosen else max(0., 1. - float(nearest[i]))
                return (novelty + .05 / (1 + scenes[row['scene']]) +
                        .02 / (1 + poses[pose]) + .03 * score, -order)
            winner = max((i for i in range(len(pool)) if i not in chosen), key=priority)
            chosen.append(winner)
            scenes[pool[winner][1]['scene']] += 1
            poses[pool[winner][4]] += 1
            nearest = np.maximum(nearest, matrix @ matrix[winner])
        pool = [pool[i] for i in sorted(chosen, key=lambda i: pool[i][0])]

    for order, row in enumerate(rows):
        if scene is not None and scene != row['scene']:
            reduce()
        scene = row['scene']
        sample = vector(row)
        entry = (order, row, sample, quality(row['quality']), pose_view(row['geometry']))
        # A duplicated cue time cannot fill the bounded reference bank.
        duplicate = next((i for i, item in enumerate(pool)
                          if (item[1]['scene'], item[1]['seconds']) == (row['scene'], row['seconds'])), None)
        if duplicate is not None:
            if entry[3] > pool[duplicate][3]:
                pool[duplicate] = entry
        else:
            pool.append(entry)
        if len(pool) >= 2 * maximum:
            reduce()
    reduce()
    return [(row['id'], sample) for _, row, sample, _, _ in sorted(pool, key=lambda item: item[0])]


def refresh_local(db, group, observation, maximum, usage):
    """Refresh provisional representatives without forgetting the original seed.

    Only already admitted strict observations are eligible. Keeping the seed
    prevents a sequence of replacements from walking away from its origin.
    Full banks refresh in batches of at most eight accepted timestamps. SQLite
    supplies at most cap+8 rows; the sampler never loads a whole scene.
    """
    from .database import encoded
    if maximum == 1:
        return
    pending=getattr(group,'_appearance_pending',None)
    if pending is None:
        pending={};group._appearance_pending=pending
    if observation is not None:
        pending[(observation['scan'],observation['seconds'])]=observation['id']
        if len(group.samples)>=maximum and len(pending)<min(8,maximum):
            return
    if not pending:
        return
    ids = tuple(dict.fromkeys(group.sample_ids + tuple(pending.values())))
    pending.clear()
    seed = group.sample_ids[0]
    rows = list(db.execute('SELECT o.id,o.scan,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,s.scene '
        'FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id IN '
        '(SELECT value FROM json_each(?)) ORDER BY o.cue,o.seconds,o.geometry,o.embedding,o.id', (encoded(ids),)))
    seed_row = next(r for r in rows if r['id'] == seed)
    chosen = select((r for r in rows if r['id'] != seed and
                     (r['scan'],r['seconds']) != (seed_row['scan'],seed_row['seconds'])), maximum-1)
    samples = [group.samples[0]] + [v for _,v in chosen]
    usage.add_counts(sum(v.nbytes for v in samples)-sum(v.nbytes for v in group.samples),
                     len(samples)-len(group.samples),kind='Selected appearance representatives')
    group.sample_ids = (seed,) + tuple(i for i,_ in chosen)
    group.samples = samples
