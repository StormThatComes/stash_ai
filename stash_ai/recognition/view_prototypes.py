"""Deterministic, non-transitive, anchor-confirmed identity view prototypes.

Build independent appearance subclusters from *anchor-matched* auxiliary faces.
Neither this module nor its prototypes may establish an identity or vote for an
identity merge. Retained geometry supplies only a signed five-landmark yaw proxy;
no physical head-angle interpretation, model inference or schema migration.
"""
from dataclasses import dataclass, field
import json
import math

from .clustering import Group, vector
from .progress import tracking
from .replay_addresses import address


def _signed_pose(geometry):
    """Coarse left/right/other hint; embed similarity determines subclusters."""
    try:
        row = json.loads(geometry) if isinstance(geometry, str) else geometry
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
    return ('other' if signed is None else 'left' if signed < -neutral else
            'right' if signed > neutral else 'other')


def pose_matches(geometry, pose, *, neutral=.12, uncertainty=.02):
    hint=pose_view(geometry,neutral=neutral)
    if hint==pose:
        return True
    signed=_signed_pose(geometry)
    return bool(signed is not None and abs(abs(signed)-neutral)<=uncertainty and
                (hint=='other' or pose=='other'))


def query_pose_matches(geometry, pose, *, relaxed=False):
    """Neutral/side mismatch may be a query hint, never a training permission.

    The relaxed caller must require independent query support and the unchanged
    embedding, rival and co-presence gates. Unknown geometry and opposite side
    categories do not qualify for this fallback.
    """
    return pose_matches(geometry, pose) or bool(relaxed and _signed_pose(geometry) is not None
        and pose in ('left', 'right', 'other') and 'other' in (pose_view(geometry), pose))


@dataclass
class _Cluster:
    pose: str
    first: str
    examples: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    cues: set = field(default_factory=set)
    qualities: list = field(default_factory=list)
    total: object = field(default=None,repr=False)
    pose_hints: set = field(default_factory=set)
    scene_minimum: int = 0
    scene_witnesses: dict = field(default_factory=dict)

    def accepts_pose(self,geometry):
        return all(pose_matches(geometry,hint) for hint in (self.pose_hints or {self.pose}))

    def accepts(self, query, threshold):
        import numpy as np
        return all(float(np.dot(existing, query)) >= threshold for existing in self.examples)

    def add(self, row, query, maximum=8):
        cue = (row['scan'], row['seconds'])
        if cue in self.cues:
            return
        self.cues.add(cue)
        if self.scene_minimum and 'scene' in row.keys():
            witnesses=self.scene_witnesses.setdefault(row['scene'],[])
            if len(witnesses)<self.scene_minimum:
                witnesses.append(row['id'])
        if 'geometry' in row.keys():
            self.pose_hints.add(pose_view(row['geometry']))
            if 'other' in self.pose_hints:
                self.pose='other'
        import numpy as np
        from .reference_selection import quality
        self.total = query.astype('float64').copy() if self.total is None else self.total + query
        self.examples.append(query)
        self.observations.append(row['id'])
        self.qualities.append(quality(row['quality']) if 'quality' in row.keys() else 0.)
        if maximum>8:
            maximum=min(maximum,max(8,math.ceil(2*math.sqrt(len(self.cues)))))
        if len(self.examples)>maximum:
            centre=self.total/len(self.cues)
            first=max(range(len(self.examples)),key=lambda i:(float(np.dot(self.examples[i],centre))+.01*self.qualities[i],-i))
            chosen=[first]
            while len(chosen)<maximum:
                chosen.append(max((i for i in range(len(self.examples)) if i not in chosen),
                    key=lambda i:(1.-max(float(np.dot(self.examples[i],self.examples[j])) for j in chosen)+.03*self.qualities[i],-i)))
            chosen.sort()
            self.examples=[self.examples[i] for i in chosen]
            self.observations=[self.observations[i] for i in chosen]
            self.qualities=[self.qualities[i] for i in chosen]

    def medoid(self):
        import numpy as np
        # Every independent confirmed cue contributes, including later evidence.
        scores=[float(np.dot(sample,self.total/len(self.cues)))+.01*self.qualities[i]
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

    @property
    def source_ids(self):
        return tuple(dict.fromkeys(self.support+tuple((self.promotion or {}).get('source_reference_ids',()))))

    @property
    def identifier(self):
        return self.pose + ':' + self.medoid


def prioritized(slots,maximum):
    """Frequent first, then appearance diversity; reserve cohorts remain usable."""
    import numpy as np
    remaining=sorted(slots,key=lambda s:(-len(s.cues),s.pose,s.first))
    chosen=[]
    while remaining and len(chosen)<maximum:
        winner=(remaining[0] if not chosen else max(remaining,key=lambda s:
            (1.-max(float(np.dot(s.medoid()[0],c.medoid()[0])) for c in chosen),len(s.cues),s.first)))
        chosen.append(winner);remaining.remove(winner)
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


def add_cohort(slots,row,sample,threshold,*,complete,scene_minimum=0,maximum=8):
    index=best_cohort(slots,row,sample,threshold)
    slot=slots[index] if index is not None else None
    if slot is None:
        if not complete and len(slots)>=32:
            return 0
        slot=_Cluster(pose_view(row['geometry']),row['id'],scene_minimum=scene_minimum);slots.append(slot)
    before=len(slot.examples)
    slot.add(row,sample,maximum=maximum)
    return (len(slot.examples)-before)*sample.nbytes


def supported_cohorts(slots,cover,minimum,maximum,*,complete):
    # Narrow cohorts need their own independent support. A singleton endpoint
    # cannot borrow the broad cohort's votes to become a new reference centre.
    eligible={}
    for slot in list(cover)+list(slots):
        if len(slot.cues)>=minimum:
            eligible.setdefault(slot.medoid()[1],slot)
    ordered=prioritized(list(eligible.values()),maximum)
    return ordered if complete else ordered[:maximum]


def confirmed_views(db, policy, grouping, *, remaining_bytes, rows=None, complete=False):
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
        for slot in supported_cohorts(slots,cover,minimum,maximum,complete=complete):
            sample, identifier = slot.medoid()
            if used + sample.nbytes > remaining_bytes:
                raise ValueError('Confirmed view prototypes exceed clustering.max_reference_bytes; '
                                 'increase its value or lower max_view_prototypes_per_person.')
            used += sample.nbytes
            output.append(ViewPrototype(
                Group(owner, (sample,), True), slot.pose, identifier,
                tuple(slot.observations), len(slot.cues)))

    with tracking(rows, 'Build confirmed identity views', unit='observation') as tracked:
        for row in tracked:
            if person != row['person']:
                emit(person)
                person, slots, cover, scratch = row['person'], [], [], 0
            query = vector(row)
            scratch+=add_cohort(slots,row,query,cohesion,complete=complete,maximum=witness_limit(grouping))
            if complete:
                scratch+=add_cohort(cover,row,query,coverage_threshold(grouping),complete=True,maximum=witness_limit(grouping))
            if complete and used+scratch>remaining_bytes:
                raise ValueError('Confirmed view evidence exceeds clustering.max_reference_bytes.')
    emit(person)
    return tuple(output)
