"""Conservative query envelopes for incremental rear-view matching."""
import base64
import hashlib
import json

import numpy as np

from .body_identity import vector
from .config import digest
from .database import encoded
from .dependencies import supported, version
from .scan_heads import predicate


def _envelope(db, scan, stamp):
    pivot = None
    radius = 0.
    for row in db.execute("""SELECT b.embedding FROM body_observations b JOIN body_members m
            ON b.id=m.observation WHERE b.scan=? AND m.policy=? AND b.rejected IS NULL
            AND b.embedding IS NOT NULL ORDER BY b.cue,b.id""", (scan, stamp)):
        feature = vector(row).astype('<f4')
        if pivot is None:
            pivot = feature
        else:
            radius = max(radius, float(np.linalg.norm(feature.astype(float) - pivot)))
    if pivot is None:
        return None
    return [base64.b64encode(pivot.tobytes()).decode('ascii'), radius]


def matching_scans(db, scope, namespace, selected, stamp, gallery, search, settings):
    """The triangle inequality can exclude a whole scene, never a possible witness."""
    current = set(selected)
    audited = supported(db)
    dependencies = hashlib.sha256(gallery.revision.encode())
    for owner, bank in sorted(search.bank.items()):
        dependencies.update(encoded([owner, search.mean_versions[owner]]).encode())
        for face in bank['face_ids']:
            dependencies.update(encoded(face).encode())
        for cue in bank['cues']:
            dependencies.update(encoded(cue).encode())
    generation = dependencies.hexdigest()
    distance = np.sqrt(2 * (1 - settings['match_threshold'])) + .003
    for scan, scene in db.execute("SELECT s.id,s.scene FROM scans s WHERE scope=? AND namespace=? "
                                  "AND status='complete' AND " + predicate(), (scope, namespace)):
        key = 'rear-pending:' + digest([scope, namespace, scene])
        saved = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        entry = json.loads(saved[0]) if saved else {}
        identity = [scan, stamp, version(db,'raw-body',scan) if audited else None]
        if scan in current or not audited or entry.get('identity') != identity:
            entry = dict(identity=identity, envelope=_envelope(db, scan, stamp))
        elif entry.get('checked') == generation:
            continue
        entry['checked'] = generation
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, encoded(entry)))
        if scan in current or entry['envelope'] is None:
            continue
        pivot, radius = entry['envelope']
        query = np.frombuffer(base64.b64decode(pivot), dtype='<f4')
        if gallery.tree.query(query, k=1, distance_upper_bound=distance+radius)[0] != np.inf:
            yield scan, scene
