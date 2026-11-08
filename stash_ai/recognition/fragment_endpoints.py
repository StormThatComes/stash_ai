"""Nominate different body perspectives by their independent measured face endpoints."""
from collections import defaultdict


def enabled(settings):
    value = settings.get('endpoint_matching')
    return settings.get('tracklet_matching', False) if value is None else value


def nominations(looks, lookup, roots, shots, scores, settings):
    if not enabled(settings) or not looks:
        return {}
    from .independent_evidence import unit
    by_owner = defaultdict(lambda: defaultdict(list))
    for (owner,kind),entries in looks.items():
        for start,end,member in entries:
            by_owner[owner][kind,start,end,shots[member['cue']]].append(member['id'])
    eligible = {}
    face_threshold = settings.get('face_threshold',1.)
    face_margin = settings.get('face_margin',.1)
    for owner,tracks in by_owner.items():
        rivals = [lookup[k]['_face_vector'] for k,p in roots.items()
                  if p!=owner and lookup[k].get('_face_vector') is not None]
        fragments = sorted(tracks.items())
        for i,((kind,a,b,shot),members) in enumerate(fragments):
            for (other,c,d,other_shot),peers in fragments[i+1:]:
                if kind==other or shot==other_shot:
                    continue
                ids = a,b,c,d
                if len(set(ids))!=4:
                    continue
                faces = [lookup[k] for k in ids]
                if any(f.get('_face_vector') is None for f in faces):
                    continue
                pixels = {unit(dict(scan=f['scan'],seconds=f['seconds'],quality=f.get('face_quality') or '{}')) for f in faces}
                if len(pixels)!=4:
                    continue
                vectors = [f['_face_vector'] for f in faces]
                minimum = min(scores.face_similarity(x,y) for x in vectors[:2] for y in vectors[2:])
                if minimum<face_threshold:
                    continue
                rival = max((scores.face_similarity(x,y) for x in vectors for y in rivals),default=-1.)
                if minimum-rival<face_margin:
                    continue
                for oid in members:
                    if oid not in eligible or minimum>eligible[oid][2]:
                        eligible[oid] = owner,[(a,b,oid),(c,d,peers[0])],minimum
                for oid in peers:
                    if oid not in eligible or minimum>eligible[oid][2]:
                        eligible[oid] = owner,[(a,b,members[0]),(c,d,oid)],minimum
    return eligible


class ExternalEndpoints:
    """Owner-keyed nomination of separately face-rooted views in another scan."""
    def __init__(self,db,gallery,config):
        from functools import lru_cache
        from .pose_policy import anchor_sql
        from .reference_roles import strict_sql
        from .scan_heads import predicate
        from .independent_evidence import unit
        from .clustering import vector
        self.gallery = gallery
        self.banks = defaultdict(list)
        for index in range(len(gallery.banks)):
            self.banks[gallery.owner_at(index)].append(index)
        self.minimum = config.data['recognition']['match_threshold']
        self.body_floor = config.data['sapiensid']['identities']['match_threshold']-config.data['sapiensid']['identities']['ambiguity_margin']
        self.revisions = {}
        sql = ('SELECT o.* FROM body_observations b JOIN observations o ON o.id=b.face AND o.scan=b.scan '
               'JOIN members m ON m.observation=o.id JOIN scans s ON s.id=b.scan '
               'WHERE b.id=? AND m.person=? AND o.rejected IS NULL AND b.rejected IS NULL AND '
               + strict_sql()+' AND '+anchor_sql('o',config.data['clustering'])+' AND '+predicate())
        @lru_cache(maxsize=128)
        def face(root,owner):
            row = db.execute(sql,(root,owner)).fetchone()
            return (vector(row),unit(row)) if row else None
        self.face = face

    def revision(self,owner,versions):
        from .config import digest
        if owner not in self.revisions:
            self.revisions[owner] = digest(sorted(versions[i] for i in self.banks.get(owner,())))
        return self.revisions[owner]

    def match(self,scan,kind,decision,occupied,ranked,*,reject_rival=None):
        import numpy as np
        proof = decision.get('rooted_track') or {}
        owner = proof.get('owner')
        if not owner or owner in occupied or len(proof.get('roots',()))!=2:
            return None
        local = [self.face(root,owner) for root in proof['roots']]
        if any(f is None for f in local) or local[0][1]==local[1][1]:
            return None
        for candidate in ranked:
            rival = candidate[2]
            if rival!=owner and max(candidate[0],candidate[5])>=self.body_floor:
                if reject_rival is None or not reject_rival(rival,local):
                    return None
        # Disk galleries fetch one bank at a time; all eligible banks participate.
        for index in self.banks.get(owner,()):
            _,_,bank = self.gallery.banks[index]
            for w in bank:
                if w['scan']==scan or w.get('perspective','back')==kind:
                    continue
                external = [self.face(root,owner) for root in w['roots']]
                if (len(external)!=2 or any(f is None for f in external) or
                        len({f[1] for f in [*external,*local]})!=4):
                    continue
                score = min(float(a[0] @ b[0]) for a in local for b in external)
                if np.isfinite(score) and score>=self.minimum:
                    return owner,score,w
        return None


def split_owner_nominations(looks, lookup, roots, shots, scores, settings):
    """Propose a split face ID only from four independent, physically compatible endpoints."""
    if not enabled(settings):
        return
    from .independent_evidence import unit
    tracks = {}
    occupied = defaultdict(set)
    for oid,owner in roots.items():
        occupied[owner].add(lookup[oid]['cue'])
    for (owner,kind), entries in looks.items():
        for a,b,member in entries:
            tracks.setdefault((owner,kind,shots[member['cue']],a,b),member['id'])
    by_owner = defaultdict(list)
    for (owner,kind,shot,a,b),member in tracks.items():
        ends = [lookup[a],lookup[b]]
        if any(r.get('_face_vector') is None for r in ends):
            continue
        pixels = {unit(dict(scan=r['scan'],seconds=r['seconds'],quality=r.get('face_quality') or '{}')) for r in ends}
        if len(pixels)==2:
            by_owner[owner].append((kind,shot,[a,b],pixels,member))
    from itertools import combinations
    for a,b in combinations(sorted(by_owner),2):
        if occupied[a] & occupied[b]:
            continue
        best = None
        for kind,shot,left,pixels,member in by_owner[a]:
            for other,view,right,units,_ in by_owner[b]:
                if kind==other or shot==view or pixels & units:
                    continue
                score = min(scores.face_similarity(lookup[i]['_face_vector'],lookup[j]['_face_vector'])
                            for i in left for j in right)
                if score>=settings['face_threshold'] and (best is None or score>best[0]):
                    best = score,member,left,right
        if best is not None:
            score,member,left,right = best
            yield member,dict(people=[a,b],roots=left+right,minimum=score)
