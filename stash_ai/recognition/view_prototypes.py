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


def pose_view(geometry, *, neutral=.12):
    """Coarse left/right/other hint; embed similarity determines subclusters."""
    try:
        row = json.loads(geometry) if isinstance(geometry, str) else geometry
        if len(row) != 15:
            return 'other'
        values = [float(x) for x in row]
        if not all(math.isfinite(x) for x in values):
            return 'other'
        eye = ((values[4] + values[6]) / 2, (values[5] + values[7]) / 2)
        distance = math.hypot(values[4] - values[6], values[5] - values[7])
        if distance <= 1:
            return 'other'
        horizontal = ((values[6]-values[4])/distance, (values[7]-values[5])/distance)
        signed = ((values[8]-eye[0])*horizontal[0] + (values[9]-eye[1])*horizontal[1]) / distance
        return 'left' if signed < -neutral else 'right' if signed > neutral else 'other'
    except (ValueError, TypeError, OverflowError):
        return 'other'


@dataclass
class _Cluster:
    pose: str
    first: str
    examples: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    cues: set = field(default_factory=set)

    def accepts(self, query, threshold):
        import numpy as np
        return all(float(np.dot(existing, query)) >= threshold for existing in self.examples)

    def add(self, row, query, maximum=8):
        cue = (row['scan'], row['seconds'])
        if cue in self.cues:
            return
        self.cues.add(cue)
        if len(self.examples) < maximum:
            self.examples.append(query)
            self.observations.append(row['id'])

    def medoid(self):
        import numpy as np
        # Stable tie preference is the first canonical observation.
        scores = [sum(float(np.dot(x, y)) for y in self.examples)
                  for x in self.examples]
        index = max(range(len(scores)), key=lambda i: scores[i])
        return self.examples[index], self.observations[index]


@dataclass(frozen=True)
class ViewPrototype:
    group: Group
    pose: str
    medoid: str
    support: tuple
    support_count: int

    @property
    def identifier(self):
        return self.pose + ':' + self.medoid


def confirmed_views(db, policy, grouping, *, remaining_bytes):
    """Return bounded frozen per-view medoids from first-pass anchor confirmations.

    One SQL stream ordered by person/canonical sample key. Limited exemplars
    are held for one person at a time; only chosen immutable medoids persist.
    Weak/transitive chains cannot form a coherent prototype.
    """
    maximum = grouping['max_view_prototypes_per_person']
    minimum = grouping['view_min_support']
    cohesion = grouping['view_cohesion_threshold']
    rows = db.execute(
        f'SELECT c.person,o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding '
        f'FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
        'JOIN scans s ON s.id=o.scan '
        "WHERE c.policy=? AND c.status='accepted' "
        "AND json_extract(c.payload,'$.reason')='auxiliary_identity' "
        'AND o.rejected IS NULL AND c.person IS NOT NULL '
        'ORDER BY c.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
        (policy,))
    output = []
    person, slots = None, []
    used = 0

    def emit(owner):
        nonlocal used
        if owner is None:
            return
        eligible = [s for s in slots if len(s.cues) >= minimum]
        eligible.sort(key=lambda s: (-len(s.cues), s.pose, s.first))
        for slot in eligible[:maximum]:
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
                person, slots = row['person'], []
            query = vector(row)
            pose = pose_view(row['geometry'])
            chosen = next((slot for slot in slots
                           if slot.pose == pose and slot.accepts(query, cohesion)), None)
            if chosen is None:
                # Bound quality evidence for a single person without silently
                # allowing new identities or ever-growing reference groups.
                if len(slots) >= 32:
                    continue
                chosen = _Cluster(pose, row['id'])
                slots.append(chosen)
            chosen.add(row, query)
    emit(person)
    return tuple(output)
