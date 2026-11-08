"""Conservative per-scene query bounds and dependencies of deferred rear matches."""
import base64
import hashlib
import json

import numpy as np

from .body_identity import vector
from .config import digest
from .database import encoded
from .dependencies import supported, version
from .scan_heads import predicate


def _envelopes(db, scan, stamp):
    balls = []
    owners = set()
    for row in db.execute("""SELECT CASE WHEN COALESCE(json_extract(b.quality,'$.quality.orientation'),'unknown')<>'front'
            OR json_extract(m.proof,'$.query_perspective') IN ('back','side') THEN b.embedding END AS embedding,
            m.person FROM body_observations b
            JOIN body_members m ON b.id=m.observation WHERE b.scan=? AND m.policy=? AND b.rejected IS NULL
            AND b.embedding IS NOT NULL ORDER BY b.cue,b.id""", (scan,stamp)):
        if row['person']:
            owners.add(row['person'])
        # A missing historical orientation remains in the envelope.
        if row['embedding'] is None:
            continue
        feature = vector(row).astype(np.float64)
        distances = [np.linalg.norm(feature-centre) for centre,_ in balls]
        if len(balls)<8 and (not distances or min(distances)>.45):
            balls.append([feature,0.])
        else:
            index = int(np.argmin(distances))
            balls[index][1] = max(balls[index][1],float(distances[index]))
    return [[base64.b64encode(p.astype('<f4').tobytes()).decode('ascii'),r] for p,r in balls], sorted(owners)


def _bank_versions(gallery):
    if hasattr(gallery,'versions'):
        return gallery.versions
    result = []
    for owner,source,witnesses in gallery.banks:
        stamp = hashlib.sha256(encoded([owner,source]).encode())
        for row in witnesses:
            stamp.update(encoded([row['id'],row['roots'],row['quality'],row.get('perspective','back')]).encode())
            stamp.update(row['_vector'].tobytes())
        result.append(stamp.hexdigest())
    return result


class Pending:
    def __init__(self, db, gallery, search, settings):
        self.db, self.gallery, self.search = db, gallery, search
        self.audited = supported(db)
        from .local_dependencies import available
        self.local_clocks = self.audited and available(db) and db.execute('PRAGMA user_version').fetchone()[0] >= 13
        if self.local_clocks:
            from .dependency_schema import validate_triggers
            validate_triggers(db, required=True)
        self.banks = _bank_versions(gallery)
        self.radius = np.sqrt(2*(1-settings['match_threshold']+settings['ambiguity_margin']))+.003
        self.owners = {}
        self.floor = settings['match_threshold']-settings['ambiguity_margin']

    def owner_revision(self, owner):
        if owner not in self.owners:
            bank = self.search.bank.get(owner)
            if bank is None:
                self.owners[owner] = None
            else:
                value = hashlib.sha256(self.search.mean_versions[owner].encode())
                value.update(bytes([owner in getattr(self.search, 'unavailable', ())]))
                # Include face vetoes, co-presence, aliases and the exact modes.
                if self.audited:
                    value.update(str(version(self.db,'owner',owner)).encode())
                for face in bank['face_ids']:
                    value.update(encoded(face).encode())
                cues = bank['cues']
                for cue in (sorted(cues) if isinstance(cues,set) else cues):
                    value.update(encoded(cue).encode())
                self.owners[owner] = value.hexdigest()
        return self.owners[owner]

    def generation(self,entry,banks):
        owners = set(entry['local']) | set(entry.get('rivals',()))
        owners.update(self.gallery.owner_at(i) for i in banks)
        return digest([sorted(self.banks[i] for i in banks),
                       [(owner,self.owner_revision(owner)) for owner in sorted(owners)]])

    def candidates(self,scope,namespace,selected,stamp,*,active=()):
        from .rear_bounds import changed_modes
        catalogue = 'rear-rivals:'+digest(['profile-v2',scope,namespace,stamp])
        old = self.db.execute('SELECT value FROM metadata WHERE key=?',(catalogue,)).fetchone()
        previous = json.loads(old[0]) if old else None
        budget = self.search.config.data['sapiensid']['execution']['reference_bytes'] if self.search.mean_versions else 0
        changed = changed_modes(self.search,previous,budget)
        scope_generation = [version(self.db), version(self.db, 'heads')] if self.local_clocks else None
        from .profiling import count
        reverse = None
        nominated = None
        if self.local_clocks and self.banks:
            from .rear_reverse import ReverseIndex
            reverse = ReverseIndex(self.db,scope,namespace,stamp,self)
            nominated = reverse.nominate(self,changed)
        base = " FROM scans s "
        if self.local_clocks:
            base += ("LEFT JOIN evidence_versions b ON b.kind='raw-body' AND b.key=s.id "
                     "LEFT JOIN evidence_versions f ON f.kind='raw-observations' AND f.key=s.id "
                     "LEFT JOIN evidence_versions l ON l.kind='scene' AND l.key=json_array(s.scope,s.namespace,s.scene) ")
            fields = 's.id,s.scene,COALESCE(b.version,0),COALESCE(f.version,0),COALESCE(l.version,0)'
        else:
            fields = 's.id,s.scene'
        heads = self.db.execute('SELECT '+fields+base+"WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND "
                                +predicate(),(scope,namespace))
        for scan,scene,*clocks in heads:
            count('rear_scene_headers')
            key = 'rear-pending:' + digest([scope,namespace,scene])
            identity = ([scan,stamp,*clocks] if self.local_clocks else
                        [scan,stamp,version(self.db,'raw-body',scan) if self.audited else None,
                         version(self.db,'raw-observations',scan) if self.audited else None])
            if reverse is not None:
                reverse.state['scans'][scan] = identity
                if (nominated is not None and scan not in selected and scan not in nominated
                        and reverse.previous.get('scans',{}).get(scan)==identity
                        and self.db.execute('SELECT 1 FROM metadata WHERE key=?',(key,)).fetchone()):
                    count('rear_scenes_skipped_by_reverse_index')
                    continue
            count('rear_scene_envelopes_read')
            saved = self.db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
            entry = json.loads(saved[0]) if saved else {}
            refresh = (not self.audited or (scan in selected and not self.local_clocks)
                       or entry.get('identity') != identity or 'balls' not in entry)
            if refresh:
                balls,owners = _envelopes(self.db,scan,stamp)
                entry = dict(identity=identity,balls=balls,local=owners)
            banks = set()
            near_change = changed is None
            for pivot,radius in entry['balls']:
                query = np.frombuffer(base64.b64decode(pivot),dtype='<f4')
                if not near_change and changed:
                    near_change = any(float(query@mean) + radius*norm + .003 >= self.floor
                                      for mean,norm in changed)
                banks.update(self.gallery.nearby(query,self.radius+radius))
            revision = self.generation(entry,banks)
            certified = scan not in selected or (self.local_clocks and entry.get('checked_scope') == scope_generation)
            if certified and not refresh and not near_change and entry.get('checked')==revision:
                if reverse is not None:
                    reverse.store(scan,entry,banks,self)
                continue
            if scan in selected or scan in active or banks:
                yield scan,scene
                # The actual veto returned by the exact global search may be new.
                entry['rivals'] = [r[0] for r in self.db.execute("""SELECT DISTINCT json_extract(m.proof,'$.rear_retry.rival') AS rival
                    FROM body_observations b JOIN main.body_members m ON m.observation=b.id
                    WHERE b.scan=? AND rival IS NOT NULL ORDER BY rival""",(scan,))]
                revision = self.generation(entry,banks)
                entry['checked_scope'] = scope_generation
            if reverse is not None:
                reverse.store(scan,entry,banks,self)
            entry['checked'] = revision
            value = encoded(entry)
            if not saved or saved[0]!=value:
                self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,value))
        if reverse is not None:
            reverse.finish()
        if previous != self.search.mean_versions:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(catalogue,encoded(self.search.mean_versions)))
