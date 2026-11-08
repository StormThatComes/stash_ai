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
    """Retain as many historical IDs as possible, then maximise independent overlap."""
    from .body_identity import independent
    if not origins or not owners or not groups:
        return {}
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    positions = {owner:i for i,owner in enumerate(owners)}
    group_order = sorted(range(len(groups)),key=lambda i:min(r['id'] for r in groups[i]))
    edges = []
    for row_index, group_index in enumerate(group_order):
        history = defaultdict(list)
        for row in groups[group_index]:
            for person in origins.get(physical_key(scan,row),()):
                if person in positions:
                    history[person].append(row)
        for person, rows in history.items():
            support = len(independent(rows,settings))
            if support:
                edges.append((row_index,positions[person],support))
    if not edges:
        return {}
    # Each group has its own unmatched column. Real edges dominate all
    # available overlap combined; age breaks equal-overlap ties deterministically.
    cardinality = 1 + sum(v for _,_,v in edges)
    matrix_rows, columns, values = [],[],[]
    for row,col,support in edges:
        matrix_rows.append(row)
        columns.append(col)
        values.append(1+cardinality+support+(len(owners)-col)/((len(owners)+1)*(len(groups)+1)))
    for row in range(len(groups)):
        matrix_rows.append(row)
        columns.append(len(owners)+row)
        values.append(1.)
    matrix = csr_matrix((values,(matrix_rows,columns)),shape=(len(groups),len(owners)+len(groups)))
    rows,cols = min_weight_full_bipartite_matching(matrix,maximize=True)
    return {group_order[r]:owners[c] for r,c in zip(rows,cols) if c<len(owners)}
