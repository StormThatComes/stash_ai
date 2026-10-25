"""Bounded, deterministic quality/diversity sampling of strict members.

At most twice the reference cap is materialized. Repeated reductions retain
appearance and scene coverage rather than permanently freezing the first N cues.
The returned order is canonical so scalar score reductions remain reproducible.
"""
from collections import Counter
import json
from .run_cache import metadata
import math


def quality(raw):
    try:
        record = metadata(raw)
        def number(key):
            value = record.get(key, 0.)
            return max(0., float(value)) if type(value) in (int, float) and math.isfinite(value) else 0.
        score = (min(1., math.log1p(number('pixels')) / math.log1p(512)) +
                min(1., math.log1p(number('sharpness')) / math.log1p(1000)) +
                1. / (1. + number('pose_deviation'))) / 3
        if record.get('alignment_valid') is False:
            return 0.
        from .visibility import cached
        if cached(record):
            score *= min(1., number('visible_face_fraction'))
        if 'alignment_error' in record:
            score /= 1. + 8. * number('alignment_error')
        if 'alignment_face_coverage' in record:
            score *= min(1., number('alignment_face_coverage'))
        return score
    except (ValueError, TypeError, AttributeError):
        return 0.


def select(rows, maximum, *, vector_reader=None):
    import numpy as np
    from .clustering import vector
    from .view_prototypes import pose_view
    if maximum <= 0:
        return []
    vector = vector_reader or vector
    pool, scene = [], None
    # Saturated support keeps only enough metadata to certify a rare look.
    # Frozen pivots prevent gradually changing observations from joining a mode.
    from .independent_evidence import Votes, unit
    def evidence(row):
        return unit(dict(row, scan=row['scan'] if 'scan' in row.keys() else row['scene']))
    reserves = []

    def reserve(entry):
        order, row, sample, score, pose = entry
        matching = [slot for slot in reserves if slot['pose'] == pose and
                    float(np.dot(slot['pivot'], sample)) >= .95 and
                    float(np.dot(slot['best'][2], sample)) >= .95]
        slot = max(matching, key=lambda v: float(np.dot(v['pivot'], sample))) if matching else None
        if slot is None:
            # Confirmed modes keep their protection; probationary singletons
            # cannot monopolize it. A bounded second bank discovers later modes.
            if len(reserves) >= 2 * maximum:
                probation = [v for v in reserves if len(v['votes']) < 3]
                if not probation:
                    return
                reserves.remove(min(probation, key=lambda v: (len(v['votes']), v['best'][0])))
            slot = dict(pivot=sample, pose=pose, best=entry, votes=Votes())
            reserves.append(slot)
        if len(slot['votes']) < 3:
            slot['votes'].add(dict(row, scan=row['scan'] if 'scan' in row.keys() else row['scene']))
        if score > slot['best'][3]:
            slot['best'] = entry


    def reduce():
        nonlocal pool
        supported = [slot['best'] for slot in reserves if len(slot['votes']) >= 3]
        known = {item[1]['id'] for item in pool}
        pool += [item for item in supported if item[1]['id'] not in known]
        unique = {}
        for entry in pool:
            key = evidence(entry[1])
            previous = unique.get(key)
            if previous is None or entry[3] > previous[3]:
                unique[key] = entry
        pool = sorted(unique.values(), key=lambda entry: entry[0])
        if len(pool) <= maximum:
            return
        if len({sample.shape for _, _, sample, _, _ in pool}) != 1:
            raise ValueError('Face namespace contains mixed embedding dimensions.')
        matrix = np.stack([item[2] for item in pool])
        chosen, scenes, poses = [], Counter(), Counter()
        nearest = np.full(len(pool), -1., dtype=np.float64)
        reserved = {item[1]['id'] for item in supported}
        for _ in range(maximum):
            def priority(i):
                order, row, _, score, pose = pool[i]
                novelty = 0. if not chosen else max(0., 1. - float(nearest[i]))
                return (row['id'] in reserved, novelty * (.25 + .75 * score) + .05 / (1 + scenes[row['scene']]) +
                        .02 / (1 + poses[pose]) + .20 * score, -order)
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
        reserve(entry)
        # A duplicated cue time cannot fill the bounded reference bank.
        duplicate = next((i for i, item in enumerate(pool)
                          if evidence(item[1]) == evidence(row) or
                          (item[1]['scene'], item[1]['seconds']) == (row['scene'], row['seconds'])), None)
        if duplicate is not None:
            if entry[3] > pool[duplicate][3]:
                pool[duplicate] = entry
        else:
            pool.append(entry)
        if len(pool) >= 2 * maximum:
            reduce()
    reduce()
    return [(row['id'], sample) for _, row, sample, _, _ in sorted(pool, key=lambda item: item[0])]


def refresh_local(db, group, observation, maximum, usage, *, policy=None, config=None):
    """Refresh provisional representatives while keeping a validated fixed core.

    Only already admitted strict observations are eligible. The original seed
    may be replaced once by a strictly validated core; its medoid stays fixed.
    Full banks refresh after at most eight accepted timestamps. SQLite streams
    all admitted rows in canonical order; the sampler keeps at most twice the cap.
    """
    from .database import encoded
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
    if not hasattr(group, '_identity_origin'):
        group._identity_origin = group.sample_ids[0]
    seed = group.sample_ids[0]
    from .replay_addresses import address
    def rows():
        return db.execute('SELECT o.id,o.scan,o.seconds,o.geometry,o.quality,o.dimensions,o.embedding,s.scene '
            'FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id IN ('
            'SELECT value FROM json_each(?) UNION SELECT '+address(db,'c')+
            " FROM candidates c WHERE c.person=? AND c.policy=? AND c.status IN ('pending','accepted') "
            "AND json_extract(c.payload,'$.reason')='candidate_group') "
            'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
            (encoded(ids),group.identifier,policy))
    # The stable label origin is separate from the first scoring vector. Replace
    # a weak seed once, using an independently supported complete-link core.
    # Its validated medoid then stays protected for the rest of this local run.
    if config is not None and not getattr(group, '_seed_validated', False):
        from .independent_evidence import Votes
        from .clustering import vector
        import numpy as np
        support = Votes()
        core = []
        minimum = max(3, config.data['recognition']['min_identity_support'])
        cohesion = config.data['clustering']['cohesion_threshold']
        seed_quality = quality(db.execute('SELECT quality FROM observations WHERE id=?', (seed,)).fetchone()[0])
        for row in rows():
            sample = vector(row)
            if quality(row['quality']) <= seed_quality or any(float(np.dot(sample,v)) < cohesion for _,v in core):
                continue
            before = len(support)
            support.add(row)
            if len(support) > before and len(core) < minimum:
                core.append((row, sample))
            if len(support) >= minimum:
                winner = max(core, key=lambda pair: (sum(float(np.dot(pair[1],v)) for _,v in core),
                                                      quality(pair[0]['quality']), pair[0]['id']))
                seed = winner[0]['id']
                group.samples[0] = winner[1]
                group.sample_ids = (seed,) + group.sample_ids[1:]
                group._seed_validated = True
                break
    seed_row = db.execute('SELECT scan,seconds FROM observations WHERE id=?',(seed,)).fetchone()
    chosen = (select((r for r in rows() if r['id'] != seed and
                     (r['scan'],r['seconds']) != (seed_row['scan'],seed_row['seconds'])), maximum-1)
              if maximum > 1 else [])
    samples = [group.samples[0]] + [v for _,v in chosen]
    usage.add_counts(sum(v.nbytes for v in samples)-sum(v.nbytes for v in group.samples),
                     len(samples)-len(group.samples),kind='Selected appearance representatives')
    group.sample_ids = (seed,) + tuple(i for i,_ in chosen)
    group.samples = samples
