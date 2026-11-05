"""Physical body origins survive collection and scan-policy changes."""
from collections import defaultdict
from .config import digest


def physical_key(scan, row):
    return digest([scan[k] for k in ('scope', 'namespace', 'scene', 'input_sha')] +
                  [row[k] for k in ('cue', 'seconds', 'geometry')])


def origins_for_scene(db, scan):
    result = defaultdict(set)
    for row in db.execute('SELECT b.cue,b.seconds,b.geometry,i.person FROM body_origins i '
            'JOIN body_observations b ON b.id=i.observation JOIN scans s ON s.id=b.scan '
            'WHERE s.scope=? AND s.namespace=? AND s.scene=? AND s.input_sha=?',
            tuple(scan[k] for k in ('scope','namespace','scene','input_sha'))):
        result[physical_key(scan,row)].add(row['person'])
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
