"""Frozen, independently rooted appearance evidence for scene-local turns."""
from collections import defaultdict

from .body_identity import independent


class TurnWitnesses:
    def __init__(self, lookup, settings, scores):
        self.lookup = lookup
        self.settings = settings
        self.scores = scores
        self.by_view = defaultdict(list)

    def add(self, owner, angle, member, roots, shot):
        bank = self.by_view[owner,angle]
        if any(item[1]==roots for item in bank):
            return
        # Keep bounded examples from different shots before retaining repeated episodes.
        if len(bank)>=16:
            if any(item[2]==shot for item in bank):
                return
            duplicate = next((i for i,item in enumerate(bank) if sum(w[2]==item[2] for w in bank)>1),None)
            if duplicate is None:
                return
            del bank[duplicate]
        bank.append((member,roots,shot))

    def verify(self, owner, angle, row, roots, *, outside_shot=None, distinct_shots=False, confirmed_core=False):
        bank = self.by_view.get((owner,angle),())
        eligible = []
        for member,ends,shot in bank:
            if (set(roots).intersection(ends) or row['cue']==member['cue'] or
                    outside_shot is not None and shot==outside_shot):
                continue
            score = self.scores.face_similarity(row['_vector'], member['_vector'])
            if score>=self.settings['match_threshold']:
                eligible.append((member,ends,shot,score))
        eligible.sort(key=lambda item:(-item[3],item[0]['id'],tuple(item[1])))
        if confirmed_core and len(roots)==2:
            for member,ends,_,score in eligible:
                if score>=.9999 and len(independent([self.lookup[k] for k in (*roots,*ends)],self.settings))==4:
                    return dict(view_score=score,view_floor=.9999,view_witnesses=[member['id']],view_roots=list(ends))
        # Visit pairs in descending order of their weaker score, with stable ties.
        pairs = ((a,b) for i,b in enumerate(eligible) for a in eligible[:i])
        for a,b in pairs:
            if set(a[1]).intersection(b[1]) or distinct_shots and a[2]==b[2]:
                continue
            if distinct_shots and len(independent([a[0],b[0]],self.settings))<2:
                continue
            if len(independent([self.lookup[k] for k in (*a[1],*b[1])],self.settings))<4:
                continue
            similarity = self.scores.face_similarity(a[0]['_vector'],b[0]['_vector'])
            # A novel appearance cannot borrow the identity of a tightly
            # repeated rear view. Calibrate only from the two external episodes.
            floor = max(self.settings['match_threshold'],1-2*(1-similarity)-.0001)
            low = min(a[3],b[3])
            if similarity>=self.settings['cohesion_threshold'] and low>=floor:
                return dict(view_score=low,view_floor=floor,view_witnesses=[a[0]['id'],b[0]['id']],
                            view_roots=[*a[1],*b[1]])
        return None
