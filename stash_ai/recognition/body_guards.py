"""Face contradictions and simultaneous claims veto body-only evidence."""
from collections import defaultdict



class FaceGuard:
    def __init__(self, db, config, *, scores=None, frozen=None):
        self.frozen=frozen
        from .body_identity import Scores
        self.scores=scores if scores is not None else Scores(config)
        self.db = db
        self.floor = config.data['sapiensid']['matching']['face_floor']
        self.grouping = config.data.get('clustering', {})
        from collections import OrderedDict
        self.banks = OrderedDict()
        self.bank_bytes = 0
        self.bank_limit = min(8*1024**2, config.data['sapiensid']['execution']['reference_bytes']//16)

    def witnesses(self, person):
        from .body_scene_jobs import Witness, face_witness_query
        if self.frozen is not None:
            for raw in self.frozen[person]:
                yield raw if isinstance(raw,Witness) else Witness(raw['dimensions'],raw['embedding'])
            return
        key = person, self.db.total_changes, self.db.execute('PRAGMA data_version').fetchone()[0]
        if key in self.banks:
            self.banks.move_to_end(key)
            yield from self.banks[key][0]
            return
        retained, used = [], 512
        for raw in self.db.execute(face_witness_query(), (person,)):
            value = Witness(*raw)
            used += 2*len(value.embedding)+len(value.quality)+512
            if used <= self.bank_limit:
                retained.append(value)
            else:
                retained.clear()
            yield value
        # Only a fully consumed bank is reusable; early vetoes store no prefix.
        if used <= self.bank_limit:
            while self.banks and self.bank_bytes+used > self.bank_limit:
                _, (_, amount) = self.banks.popitem(last=False)
                self.bank_bytes -= amount
            self.banks[key] = retained, used
            self.bank_bytes += used

    def allows(self, row, bank, person=None):
        face = row.get('_face_vector')
        if face is None:
            return True
        import json
        from .pose_policy import anchor_visibility
        raw = row.get('face_quality')
        q = json.loads(raw) if isinstance(raw,str) else (raw or {})
        gates = (('pose_deviation','max_anchor_pose_deviation',True),
                 ('pixels','min_anchor_face_pixels',False),('sharpness','min_anchor_sharpness',False))
        weak = any(self.grouping.get(key) is not None and type(q.get(metric)) in (float,int)
                   and (q[metric]>self.grouping[key] if upper else q[metric]<self.grouping[key])
                   for metric,key,upper in gates)
        from .visibility import cached
        weak |= bool(cached(q) and q['visible_face_fraction']<anchor_visibility(self.grouping))
        others = (r for r in bank if r.get('_face_vector') is not None)
        if person is not None:
            # Stream complete face vetoes; a large owner must not allocate an
            # unbounded second reference bank outside the configured budget.
            others = self.witnesses(person)
        from .progress import tracking
        from stash_ai.runtime.cancellation import current
        token=current()
        first = None
        distinct_time = distinct_pixels = False
        from .independent_evidence import unit
        with tracking(others,'Check body face vetoes',unit='face') as witnesses:
            for witness in witnesses:
                if token is not None:token.check()
                other = witness.validated if person is not None else witness['_face_vector']
                if len(face)!=len(other):
                    return False
                if self.scores.face_similarity(face,other)>=self.floor:
                    continue
                if not weak and (person is None or witness.strict):
                    return False
                if person is None:
                    record = dict(scan=witness['scan'],seconds=witness['seconds'],quality=witness.get('face_quality') or '{}')
                else:
                    record = dict(scan=witness.scan,seconds=witness.seconds,quality=witness.quality)
                stamp = record['scan'],record['seconds']
                pixel = unit(record)
                if first is None:
                    first = stamp,pixel
                distinct_time |= stamp!=first[0]
                distinct_pixels |= pixel!=first[1]
                if distinct_time and distinct_pixels:
                    return False
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
                    decision.pop('rooted_view', None)
                    decision.pop('rooted_track', None)
