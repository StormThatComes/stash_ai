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
