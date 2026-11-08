"""Physical body origins survive collection and scan-policy changes."""
from collections import defaultdict
from .config import digest


def physical_key(scan, row):
    return digest([scan[k] for k in ('scope', 'namespace', 'scene', 'input_sha')] +
                  [row[k] for k in ('cue', 'seconds', 'geometry')])


def origins_for_scene(db, scan, current=()):
    result = defaultdict(set)
    import json
    from .track_geometry import torso_points
    from .body_identity import vector
    import numpy as np
    by_time = defaultdict(list)
    for row in current:
        by_time[row['cue'],row['seconds']].append(row)
    candidates = defaultdict(set)
    for row in db.execute('SELECT b.id,b.scan,b.cue,b.seconds,b.geometry,i.person FROM body_origins i '
            'JOIN body_observations b ON b.id=i.observation JOIN scans s ON s.id=b.scan '
            'WHERE s.scope=? AND s.namespace=? AND s.scene=? AND s.input_sha=?',
            tuple(scan[k] for k in ('scope','namespace','scene','input_sha'))):
        key = physical_key(scan,row)
        result[key].add(row['person'])
        if row['scan']==scan['id']:
            continue
        previous = json.loads(row['geometry'])
        old_points = torso_points(previous)
        bounds = previous.get('bounds')
        if not bounds or len(old_points)<3:
            continue
        for new in by_time[row['cue'],row['seconds']]:
            geometry = json.loads(new['geometry'])
            points = torso_points(geometry)
            common = points.keys() & old_points.keys()
            if (len(common)<3 or not geometry.get('bounds') or
                    max(abs(a-b) for a,b in zip(bounds,geometry['bounds']))>2 or
                    any(np.linalg.norm(points[i]-old_points[i])>2.5 for i in common)):
                continue
            raw = db.execute('SELECT embedding FROM body_observations WHERE id=?',(row['id'],)).fetchone()
            if raw[0] and float(vector(raw) @ vector(new))>=.95:
                candidates[new['id']].add(row['person'])
    for current_rows in by_time.values():
        used = defaultdict(list)
        for row in current_rows:
            owners = result.get(physical_key(scan,row)) or candidates.get(row['id'],())
            for owner in owners:
                used[owner].append(row)
        for owner,rows in used.items():
            if len(rows)==1 and candidates[rows[0]['id']]=={owner}:
                result.setdefault(physical_key(scan,rows[0]),{owner})
    return result


def ordered_owners(db, origins):
    from .database import encoded
    owners=sorted({owner for values in origins.values() for owner in values})
    return [row[0] for row in db.execute('SELECT id FROM people WHERE id IN (SELECT value FROM json_each(?)) ORDER BY created,id',
                                         (encoded(owners),))]


def labels_for_groups(scan,groups,origins,settings,owners):
    """Oldest label wins merges; its largest physical overlap wins a split."""
    from collections import Counter
    from .database import encoded
    from .body_identity import independent
    votes=defaultdict(list)
    for index,group in enumerate(groups):
        keys={physical_key(scan,row) for row in group}
        counts=Counter(person for key in keys for person in origins.get(key,()))
        accepted=len(independent(group,settings))>=settings['min_support']
        for person,total in counts.items():votes[person].append((total,accepted,index))
    assigned={}
    for owner in owners:
        candidates=[v for v in votes[owner] if v[2] not in assigned]
        if candidates:
            winner=max(candidates,key=lambda v:(v[0],v[1],-v[2]))
            assigned[winner[2]]=owner
    return assigned
