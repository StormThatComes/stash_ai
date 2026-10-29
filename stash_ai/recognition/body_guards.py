"""Face contradictions and simultaneous claims veto body-only evidence."""
from collections import defaultdict
import numpy as np

from .clustering import vector


class FaceGuard:
    def __init__(self, db, config):
        self.db = db
        self.floor = config.data['sapiensid']['matching']['face_floor']
        self.owners = {}

    def allows(self, row, bank, person=None):
        face = row.get('_face_vector')
        if face is None:
            return True
        others = [r['_face_vector'] for r in bank if r.get('_face_vector') is not None]
        if person is not None:
            if person not in self.owners:
                self.owners[person] = [vector(r) for r in self.db.execute(
                    'SELECT o.dimensions,o.embedding FROM observations o JOIN members m '
                    'ON m.observation=o.id WHERE m.person=? AND o.rejected IS NULL', (person,))]
            others = self.owners[person]
        return all(len(face) == len(other) and float(np.dot(face, other)) >= self.floor
                   for other in others)


def resolve_claims(rows, decisions, config):
    """Frozen measured face owners participate; only inferred claims change."""
    claims = defaultdict(list)
    for row in rows:
        decision = decisions[row['id']]
        if decision['person'] is not None:
            claims[decision['person'], row['scan'], row['cue']].append(row)
    threshold = max(config.data['recognition']['match_threshold'],
                    config.data['recognition']['cooccurrence_merge_threshold'])
    for group in claims.values():
        conflict = False
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                a, b = first.get('_face_vector'), second.get('_face_vector')
                # A reflection needs measured face evidence. Similar clothes
                # alone cannot prove that two simultaneous bodies are one person.
                if a is None or b is None or len(a) != len(b) or float(np.dot(a, b)) < threshold:
                    conflict = True
        if conflict:
            for row in group:
                decision = decisions[row['id']]
                if decision['reason'] != 'measured_face_association':
                    decision.update(person=None, status='withheld', reason='body_copresence_conflict')
