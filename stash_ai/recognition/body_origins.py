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
