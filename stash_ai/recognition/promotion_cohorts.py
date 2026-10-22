"""Root-pair hypotheses and independently supported appearance representatives.

Each pair stays frozen. A query may support several hypotheses, but it cannot
erase an earlier pair by intersecting its proof with an unrelated confirmation.
Bounded exemplars nominate centres; every centre obtains its own narrow votes
and two-scene proof from the original, individually validated queries.
"""
from dataclasses import dataclass, field
from itertools import combinations

from .view_prototypes import _Cluster, pose_view, pose_matches


@dataclass
class _Cohort:
    cluster: _Cluster
    parents: tuple
    pivot: object
    witnesses: list = field(default_factory=list)


@dataclass(frozen=True)
class Appearance:
    sample: object
    medoid: str
    pose: str
    examples: tuple
    queries: tuple
    support: int
    scenes: tuple
    scene_count: int
    parents: tuple
    validated_ids: tuple = ()


class PromotionCohorts:
    def __init__(self, roots, radius, minimum, usage):
        self.roots, self.radius, self.minimum, self.usage = roots, radius, minimum, usage
        self.slots, self.vectors = {}, {}
        self.scratch = 0

    def _check(self, extra=0):
        if self.usage.bytes + self.scratch + extra > self.usage.limit:
            raise ValueError('Corroborated view evidence exceeds clustering.max_reference_bytes.')

    def add(self, row, query, qualified):
        import numpy as np
        raw = query.tobytes()
        if raw not in self.vectors:
            self.scratch += query.nbytes
            self._check()
            self.vectors[raw] = query
        query = self.vectors[raw]
        witness = {k: row[k] for k in ('id', 'scan', 'seconds', 'scene', 'geometry', 'quality')}
        for left, right in combinations(sorted(qualified), 2):
            if set(self.roots[left].support).intersection(self.roots[right].support):
                continue
            parents = (left, right)
            slots = self.slots.setdefault(parents, [])
            item = next((s for s in slots if float(np.dot(s.pivot, query)) >= self.radius and
                         s.cluster.accepts_pose(row['geometry']) and
                         s.cluster.accepts(query, self.radius)), None)
            if item is None:
                item = _Cohort(_Cluster(pose_view(row['geometry']), row['id']), parents, query)
                slots.append(item)
            if (row['scan'], row['seconds']) in item.cluster.cues:
                continue
            before = len(item.cluster.examples)
            item.cluster.add(witness, query)
            item.witnesses.append((witness, query))
            self.scratch += (len(item.cluster.examples) - before) * query.nbytes
            self._check()

    def supported(self, *, minimum_scenes=2, track_ids=()):
        import numpy as np
        emitted = set()
        for slots in self.slots.values():
            for item in slots:
                if len(item.cluster.cues) < self.minimum:
                    continue
                # Keep the dominant medoid first, then the bounded appearance
                # coverage witnesses. Frequency cannot suppress a supported edge.
                preferred = item.cluster.medoid()[1]
                centres = sorted(zip(item.cluster.observations, item.cluster.examples),
                                 key=lambda v: (v[0] != preferred, v[0]))
                for identifier, sample in centres:
                    key = (sample.tobytes(), item.cluster.pose)
                    if key in emitted:
                        continue
                    support = _Cluster(item.cluster.pose, identifier)
                    scenes, validated = {}, []
                    for row, query in item.witnesses:
                        if (float(np.dot(sample, query)) < self.radius or
                                not pose_matches(row['geometry'], item.cluster.pose)):
                            continue
                        support.add(row, query)
                        scenes.setdefault(row['scene'], row['id'])
                        if row['id'] in track_ids:
                            validated.append(row['id'])
                    if len(support.cues) < self.minimum or len(scenes) < minimum_scenes:
                        continue
                    examples = (sample,) + tuple(v for v in support.examples
                                                 if v.tobytes() != sample.tobytes())
                    self._check(sum(v.nbytes for v in examples))
                    chosen_scenes = tuple(sorted(scenes)[:2])
                    queries = tuple(dict.fromkeys((identifier,) + tuple(support.observations) +
                                    tuple(scenes[s] for s in chosen_scenes)))
                    emitted.add(key)
                    yield Appearance(sample, identifier, item.cluster.pose, examples, queries,
                                     len(support.cues), chosen_scenes, len(scenes), item.parents,tuple(validated))
