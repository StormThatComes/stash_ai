"""Face contradictions and simultaneous claims veto body-only evidence."""
from collections import defaultdict

from .clustering import vector


class FaceGuard:
    def __init__(self, db, config, *, scores=None):
        from .body_identity import Scores
        self.scores=scores if scores is not None else Scores(config)
        self.db = db
        self.floor = config.data['sapiensid']['matching']['face_floor']

    def allows(self, row, bank, person=None):
        face = row.get('_face_vector')
        if face is None:
            return True
        others = [r['_face_vector'] for r in bank if r.get('_face_vector') is not None]
        if person is not None:
            # Stream complete face vetoes; a large owner must not allocate an
            # unbounded second reference bank outside the configured budget.
            others = (vector(r) for r in self.db.execute(
                'SELECT o.dimensions,o.embedding FROM observations o JOIN members m '
                'ON m.observation=o.id WHERE m.person=? AND o.rejected IS NULL', (person,)))
        from .progress import tracking
        from stash_ai.runtime.cancellation import current
        token=current()
        with tracking(others,'Check body face vetoes',unit='face') as witnesses:
            for other in witnesses:
                if token is not None:token.check()
                if len(face)!=len(other) or self.scores.face_similarity(face,other)<self.floor:return False
        return True


def resolve_claims(rows, decisions, config, *, scores=None):
    """Frozen measured face owners participate; only inferred claims change."""
    from .body_identity import Scores
    if scores is None:scores=Scores(config)
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
                if a is None or b is None or len(a) != len(b) or scores.face_similarity(a,b) < threshold:
                    conflict = True
        if conflict:
            for row in group:
                decision = decisions[row['id']]
                if decision['reason'] != 'measured_face_association':
                    decision.update(person=None, status='withheld', reason='body_copresence_conflict')
