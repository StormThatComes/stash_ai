"""Deterministic, non-transitive, anchor-confirmed identity view prototypes.

Build independent appearance subclusters from *anchor-matched* auxiliary faces.
Neither this module nor its prototypes may establish an identity or vote for an
identity merge. Retained geometry supplies only a signed five-landmark yaw proxy;
no physical head-angle interpretation, model inference or schema migration.
"""
from .resource_errors import ResourceLimit
from dataclasses import dataclass, field
import json
from .run_cache import metadata
import math

from .independent_evidence import Votes
from .clustering import Group, vector
from .progress import tracking
from .replay_addresses import address


def _signed_pose(geometry):
    """Coarse left/right/other hint; embed similarity determines subclusters."""
    try:
        row = metadata(geometry) if isinstance(geometry, str) else geometry
        if len(row) != 15:
            return None
        values = [float(x) for x in row]
        if not all(math.isfinite(x) for x in values):
            return None
        eye = ((values[4] + values[6]) / 2, (values[5] + values[7]) / 2)
        distance = math.hypot(values[4] - values[6], values[5] - values[7])
        if distance <= 1:
            return None
        horizontal = ((values[6]-values[4])/distance, (values[7]-values[5])/distance)
        signed = ((values[8]-eye[0])*horizontal[0] + (values[9]-eye[1])*horizontal[1]) / distance
        return signed
    except (ValueError, TypeError, OverflowError):
        return None


def pose_view(geometry, *, neutral=.12):
    signed=_signed_pose(geometry)
    return ('unknown' if signed is None else 'left' if signed < -neutral else
            'right' if signed > neutral else 'neutral')


def pose_matches(geometry, pose, *, neutral=.12, uncertainty=.02):
    hint=pose_view(geometry,neutral=neutral)
    # Stored legacy views used 'other' for both neutral and missing geometry.
    if pose == 'other':
        if hint == 'unknown':
            return True
        pose = 'neutral'
    if hint==pose:
        return True
    signed=_signed_pose(geometry)
    return bool(signed is not None and abs(abs(signed)-neutral)<=uncertainty and
                (pose=='neutral' or hint=='neutral' and pose==('left' if signed<0 else 'right')))


def query_pose_matches(geometry, pose, *, relaxed=False):
    """Neutral/side mismatch may be a query hint, never a training permission.

    The relaxed caller must require independent query support and the unchanged
    embedding, rival and co-presence gates. Unknown geometry and opposite side
    categories do not qualify for this fallback.
    """
    if pose_matches(geometry, pose):
        return True
    pose = 'neutral' if pose == 'other' else pose
    return bool(relaxed and _signed_pose(geometry) is not None and
                pose in ('left', 'right', 'neutral') and 'neutral' in (pose_view(geometry), pose))


def pose_fallback(geometry, pose):
    """Continuous extra score required for a query-only pose mismatch.

    Five-point geometry is a noisy proxy, not a hard identity boundary. Missing
    geometry and opposing sides need stronger support than adjacent categories.
    The return value is (score premium, independent support required).
    """
    value = _signed_pose(geometry)
    pose = 'neutral' if pose == 'other' else pose
    if value is None or pose == 'unknown':
        return .08, 3
    if pose_matches(geometry, pose):
        return 0., 2
    distance = (max(0., value + .12) if pose == 'left' else
                max(0., .12 - value) if pose == 'right' else max(0., abs(value) - .12))
    opposing = pose != 'neutral' and pose_view(geometry) not in ('neutral', pose)
    return min(.08, .1 * distance), 3 if opposing else 2


@dataclass
class _Cluster:
    pose: str
    first: str
    examples: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    cues: Votes = field(default_factory=Votes)
    centre_count: int = 0
    cue_times: dict = field(default_factory=dict)
    pixel_times: dict = field(default_factory=dict)
    cue_representatives: dict = field(default_factory=dict)
    qualities: list = field(default_factory=list)
    shots: list = field(default_factory=list)
    total: object = field(default=None,repr=False)
    pose_hints: set = field(default_factory=set)
    scene_minimum: int = 0
    scene_witnesses: dict = field(default_factory=dict)
    metadata_reservation: object = field(default=None, repr=False)
    metadata_bytes: int = 0
    database: object = field(default=None, repr=False)
    maximum: int = 8
    needs_final_selection: bool = False
    witnesses: dict = field(default_factory=dict, repr=False)

    def accepts_pose(self,geometry):
        return all(pose_matches(geometry,hint) for hint in (self.pose_hints or {self.pose}))

    def accepts(self, query, threshold):
        import numpy as np
        return all(float(np.dot(existing, query)) >= threshold for existing in self.examples)

    def add(self, row, query, maximum=8):
        from .independent_evidence import stamp, unit
        from .reference_selection import quality
        from .run_cache import current

        time, pixel = stamp(row), unit(row)
        q = metadata(row['quality']) if 'quality' in row.keys() else {}
        score = quality(row['quality']) if 'quality' in row.keys() else 0.
        shot = (row['scan'], 'shot', q['view_shot']) if q.get('view_shot') is not None else time
        record = (row['id'],score,query,shot,row['scene'] if 'scene' in row.keys() else None,
                  pose_view(row['geometry']) if 'geometry' in row.keys() else None)
        self.maximum = maximum
        key = time,pixel
        previous = self.witnesses.get(key)
        if previous and (-previous[1],previous[0]) <= (-score,row['id']):
            return
        if previous is None:
            shared = current()
            self.metadata_bytes += 1024 + (0 if self.database is not None else query.nbytes)
            if shared is not None and shared.working is not None:
                if self.metadata_reservation is None:
                    from .memory_size import Reservation
                    self.metadata_reservation = Reservation(shared.working, label='view-metadata')
                self.metadata_reservation.set(self.metadata_bytes)
        isolated = time not in self.cues.edges and pixel not in self.cues.units
        self.cues.add(row)
        self.witnesses[key] = (record[:2]+(None,)+record[3:]) if self.database is not None else record
        if isolated:
            self.needs_final_selection |= self.centre_count>=maximum
            self._append_witness(key,record,maximum)
        else:
            self.needs_final_selection = True
            # The new edge cannot improve the chosen proof when only its
            # timestamp is new and its fingerprint already owns a better row.
            chosen = self.cue_representatives.get(pixel)
            if (previous is None and time not in self.cue_times and len(self.cues.edges[time])==1
                    and chosen and (-chosen[1],chosen[0])<=(-score,row['id'])
                    and len(self.cues.edges[self.pixel_times[pixel]])==1):
                return
            self._resolve(maximum)

    def _resolve(self, maximum):
        # Rank competing edges deterministically. A maximum matching can replace
        # an earlier witness to admit another independent timestamp and crop.
        from .independent_evidence import Votes
        key_order = sorted(self.witnesses, key=lambda k:(-self.witnesses[k][1],self.witnesses[k][0]))
        time_order, pixels = {}, {}
        for time,pixel in key_order:
            time_order.setdefault(time,len(time_order))
            pixels.setdefault(pixel,len(pixels))
        voting = Votes()
        for time,pixel in key_order:
            t,p = time_order[time],pixels[pixel]
            voting.edges.setdefault(t,set()).add(p)
            voting.unmatched.add(t)
        voting.match(key=lambda i:i)
        self.examples.clear()
        self.shots.clear()
        self.qualities.clear()
        self.observations.clear()
        self.scene_witnesses.clear()
        self.pose_hints.clear()
        self.pixel_times.clear()
        self.cue_times.clear()
        self.cue_representatives.clear()
        self.total = None
        self.centre_count = 0
        selected = [key for key in key_order if voting.owners.get(pixels[key[1]])==time_order[key[0]]]
        for offset in range(0,len(selected),64):
            chunk = selected[offset:offset+64]
            features = {}
            if self.database is not None:
                ids = [self.witnesses[key][0] for key in chunk]
                features = {r['id']:vector(r) for r in self.database.execute(
                    'SELECT id,dimensions,embedding FROM observations WHERE id IN (SELECT value FROM json_each(?))',
                    (json.dumps(ids),))}
            for key in chunk:
                record = self.witnesses[key]
                if self.database is not None:
                    record = record[:2]+(features[record[0]],)+record[3:]
                self._append_witness(key,record,maximum)
        self.cues.set_matching({pixel:time for time,pixel in selected})
        self.first = min(self.observations)

    def _append_witness(self, key, record, maximum):
        time,pixel = key
        oid,score,query,shot,scene,pose = record
        self.pixel_times[pixel] = time
        self.cue_times[time] = self.cue_representatives[pixel] = (oid,score)
        self.centre_count += 1
        self.total = query.astype('float64').copy() if self.total is None else self.total + query
        if self.scene_minimum and scene is not None:
            witnesses = self.scene_witnesses.setdefault(scene,[])
            if len(witnesses)<self.scene_minimum:
                witnesses.append(oid)
        if pose is not None:
            self.pose_hints.add(pose)
            self.pose = 'neutral' if 'neutral' in self.pose_hints else min(self.pose_hints)
        self.examples.append(query)
        self.observations.append(oid)
        self.shots.append(shot)
        self.qualities.append(score)
        self._reduce(maximum)

    def _reduce(self, maximum):
        import numpy as np
        if maximum>8:
            maximum=min(maximum,max(8,math.ceil(2*math.sqrt(len(self.cues)))))
        if len(self.examples)>maximum:
            centre=self.total/self.centre_count
            first=max(range(len(self.examples)),key=lambda i:(float(np.dot(self.examples[i],centre))+.01*self.qualities[i],-i))
            chosen=[first]
            while len(chosen)<maximum:
                shots = {self.shots[j] for j in chosen if len(self.shots[j]) == 3}
                chosen.append(max((i for i in range(len(self.examples)) if i not in chosen),
                    key=lambda i:(len(self.shots[i]) == 3 and self.shots[i] not in shots,
                                  1.-max(float(np.dot(self.examples[i],self.examples[j])) for j in chosen)+.03*self.qualities[i],-i)))
            chosen.sort()
            self.examples=[self.examples[i] for i in chosen]
            self.observations=[self.observations[i] for i in chosen]
            self.qualities=[self.qualities[i] for i in chosen]
            self.shots=[self.shots[i] for i in chosen]

    def medoid(self):
        import numpy as np
        # Every independent confirmed cue contributes, including later evidence.
        scores=[float(np.dot(sample,self.total/self.centre_count))+.01*self.qualities[i]
                for i,sample in enumerate(self.examples)]
        index=max(range(len(scores)),key=lambda i:scores[i])
        return self.examples[index],self.observations[index]


@dataclass(frozen=True)
class ViewPrototype:
    group: Group
    pose: str
    medoid: str
    support: tuple
    support_count: int
    authority: str = 'verified_view'
    promotion: object = None
    local_scan: str | None = None

    @property
    def source_ids(self):
        return tuple(dict.fromkeys(self.support+tuple((self.promotion or {}).get('source_reference_ids',()))))

    @property
    def identifier(self):
        return self.pose + ':' + self.medoid


def prioritized(slots,maximum,cohesion=.8,*,primary=False,reserves=False):
    """Frequent first, then appearance diversity; reserve cohorts remain usable."""
    import numpy as np
    remaining=sorted(slots,key=lambda s:(-len(s.cues),s.pose,s.first))
    chosen=[]
    # Bounded exemplars serve as held-out queries. Never credit a view for
    # retrieving its own pixels or timestamp, even in overlapping cohorts.
    features = {}
    for slot in remaining:
        ids = set(slot.observations)
        stamps = {oid:stamp for stamp,(oid,_) in slot.cue_times.items() if oid in ids}
        pixels = {oid:unit for unit,(oid,_) in slot.cue_representatives.items() if oid in ids}
        examples = [(v, shot, pixels[oid]) for oid,v,shot in zip(slot.observations,slot.examples,slot.shots)
                    if oid in stamps and oid in pixels]
        features[id(slot)] = examples
    measured = any(len(shot)==3 for slot in remaining for shot in slot.shots)
    covered = set()
    def independent_hits(slot):
        examples = features[id(slot)]
        return {(stamp,pixel) for value,stamp,pixel in examples
                if any(other_pixel != pixel and stamp != other_stamp and
                       (not primary or not measured or stamp[0] != other_stamp[0]
                        or len(stamp)==len(other_stamp)==3)
                       and float(np.dot(value,other)) >= cohesion
                       for other,other_stamp,other_pixel in examples)}
    hits = {id(slot):independent_hits(slot) for slot in remaining}
    while remaining and len(chosen)<maximum:
        def priority(index):
            slot=remaining[index]
            gain = len(hits[id(slot)] - covered)
            different = slot.pose in ('left','right','neutral') and all(slot.pose!=item.pose for item in chosen)
            distance = 1-max((float(np.dot(slot.medoid()[0], item.medoid()[0])) for item in chosen),default=1.)
            return (bool(gain),different and bool(gain),gain,distance,len(slot.cues),slot.first)
        if chosen:
            winner = max(range(len(remaining)), key=priority)
        else:
            measured = any(len(shot)==3 for slot in remaining for shot in slot.shots)
            winner = (max(range(len(remaining)),key=lambda i:len(hits[id(remaining[i])]))
                      if measured else 0)
        if primary and measured and not hits[id(remaining[winner])] - covered:
            break
        selected = remaining.pop(winner)
        chosen.append(selected)
        # Mark also the exemplars of other candidates retrievable by this view.
        for slot in remaining:
            # An opposing profile cannot use this primary through the ordinary
            # pose gate. Its separately supported exemplars still add coverage.
            if primary and selected.pose != slot.pose:
                continue
            for value,stamp,pixel in features[id(slot)]:
                if any(stamp!=s and pixel!=p and float(np.dot(value,v)) >= cohesion
                       for v,s,p in features[id(selected)]):
                    covered.add((stamp,pixel))
    if primary:
        # Independently validated reserves can corroborate another appearance
        # even when a primary medoid already covers their shots.
        return chosen + ([s for s in remaining if hits[id(s)]] if reserves else [])
    return chosen+remaining


def coverage_threshold(grouping):
    """Half the angular match radius separates independently supported looks."""
    return max(grouping['view_cohesion_threshold'],
               math.sqrt((1 + grouping['view_match_threshold']) / 2))


def best_cohort(slots, row, sample, threshold):
    """Rank every admissible frozen cohort; never admit a transitive edge."""
    import numpy as np
    eligible=[]
    for i, slot in enumerate(slots):
        if not slot.accepts_pose(row['geometry']):
            continue
        scores=[float(np.dot(v,sample)) for v in slot.examples]
        if scores and min(scores)>=threshold:
            eligible.append((min(scores),sum(scores)/len(scores),slot.first,i))
    return max(eligible)[-1] if eligible else None


def witness_limit(grouping):
    return grouping.get('max_view_witnesses',32) if grouping.get('adaptive_view_witnesses',False) else 8


def add_cohort(slots,row,sample,threshold,*,complete,scene_minimum=0,maximum=8,db=None):
    index=best_cohort(slots,row,sample,threshold)
    slot=slots[index] if index is not None else None
    if slot is None:
        if not complete and len(slots)>=32:
            return 0
        slot=_Cluster(pose_view(row['geometry']),row['id'],scene_minimum=scene_minimum,database=db);slots.append(slot)
    before=slot.metadata_bytes
    slot.add(row,sample,maximum=maximum)
    return slot.metadata_bytes-before


def supported_cohorts(slots,cover,minimum,maximum,*,complete,cohesion=.8,primary=False):
    # Narrow cohorts need their own independent support. A singleton endpoint
    # cannot borrow the broad cohort's votes to become a new reference centre.
    eligible={}
    for slot in list(cover)+list(slots):
        if len(slot.cues)>=minimum:
            if slot.needs_final_selection:
                slot._resolve(slot.maximum)
                slot.needs_final_selection = False
            eligible.setdefault(slot.medoid()[1],slot)
    eligible=list(eligible.values())
    measured = any(any(len(shot)==3 for shot in slot.shots) for slot in eligible)
    ordered=prioritized(eligible,maximum,cohesion,primary=primary and measured,reserves=complete)
    return ordered if complete else ordered[:maximum]


def confirmed_views(db, policy, grouping, *, remaining_bytes, rows=None, complete=False, include_local=False):
    """Return bounded frozen per-view medoids from first-pass anchor confirmations.

    One SQL stream ordered by person/canonical sample key. Limited exemplars
    are held for one person at a time; only chosen immutable medoids persist.
    Weak/transitive chains cannot form a coherent prototype.
    """
    maximum = grouping['max_view_prototypes_per_person']
    minimum = grouping['view_min_support']
    cohesion = grouping['view_cohesion_threshold']
    if rows is None:
        rows = db.execute(
            f'SELECT c.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality '
            f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
            'JOIN scans s ON s.id=o.scan '
            "WHERE c.policy=? AND c.status='accepted' "
            "AND json_extract(c.payload,'$.reason')='auxiliary_identity' "
            'AND o.rejected IS NULL AND c.person IS NOT NULL '
            'ORDER BY c.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
            (policy,))
    output = []
    person, slots, cover = None, [], []
    used = scratch = 0

    def emit(owner):
        nonlocal used
        if owner is None:
            return
        eligible = supported_cohorts(slots,cover,minimum,maximum,complete=complete,cohesion=cohesion,primary=True)
        local = {}
        if include_local:
            primary_ids = {id(slot) for slot in eligible}
            for slot in supported_cohorts(slots,cover,minimum,maximum,complete=True,cohesion=cohesion):
                scans = {shot[0] for shot in slot.shots}
                if id(slot) not in primary_ids and len(scans)==1:
                    local[id(slot)] = scans.pop()
                    eligible.append(slot)
        for slot in eligible:
            sample, identifier = slot.medoid()
            if used + sample.nbytes > remaining_bytes:
                raise ResourceLimit('Confirmed view prototypes exceed clustering.max_reference_bytes; '
                                 'increase its value or lower max_view_prototypes_per_person.')
            used += sample.nbytes
            output.append(ViewPrototype(
                Group(owner, (sample,), True), slot.pose, identifier,
                tuple(slot.observations), len(slot.cues), local_scan=local.get(id(slot))))

    with tracking(rows, 'Build confirmed identity views', unit='observation') as tracked:
        for row in tracked:
            if person != row['person']:
                emit(person)
                person, slots, cover, scratch = row['person'], [], [], 0
            query = vector(row)
            scratch+=add_cohort(slots,row,query,cohesion,complete=complete,maximum=witness_limit(grouping),db=db)
            if complete:
                scratch+=add_cohort(cover,row,query,coverage_threshold(grouping),complete=True,maximum=witness_limit(grouping),db=db)
            if complete and used+scratch>remaining_bytes:
                raise ResourceLimit('Confirmed view evidence exceeds clustering.max_reference_bytes.')
    emit(person)
    return tuple(output)
