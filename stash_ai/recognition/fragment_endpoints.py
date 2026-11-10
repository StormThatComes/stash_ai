"""Nominate different body perspectives by their independent measured face endpoints."""
from collections import defaultdict


def enabled(settings):
    value = settings.get('endpoint_matching')
    return settings.get('tracklet_matching', False) if value is None else value


def distinct_view(first, second, left, right, scores, settings):
    if first != second:
        return True
    # A categorical "side" can represent both profiles. Require measured
    # appearance separation before admitting another view of the same kind.
    if left is None or right is None:
        return False
    return scores.face_similarity(left,right) < settings.get('match_threshold',.85)


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
                if shot==other_shot:
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
                # Certify the actual member. Equal-score witnesses have stable addresses.
                for src,dest,ends,other_ends in ((members,peers,(a,b),(c,d)),(peers,members,(c,d),(a,b))):
                    ordered = sorted(dest)
                    for oid in sorted(src):
                        peer = next((k for k in ordered if distinct_view(kind,other,lookup.get(oid,{}).get('_vector'),
                                     lookup.get(k,{}).get('_vector'),scores,settings)), None)
                        if peer is None:
                            continue
                        proof = [(*ends,oid),(*other_ends,peer)]
                        prior = eligible.get(oid)
                        if prior is None or (-minimum,proof)<(-prior[2],prior[1]):
                            eligible[oid] = owner,proof,minimum
    return eligible


def open_nominations(looks, rows, decisions, roots, shots, scores, settings, guard):
    """An open path can attach to a two-ended look, but never trains that look."""
    if not looks or not enabled(settings):
        return {}
    from .body_identity import copresent, quality
    from .independent_evidence import unit
    with_faces = {oid: rows[oid] for oid in roots if rows[oid].get('_face_vector') is not None}
    units = {oid:unit(dict(scan=r['scan'],seconds=r['seconds'],quality=r.get('face_quality') or '{}'))
             for oid,r in with_faces.items()}
    body_cores = defaultdict(list)
    for oid,d in decisions.items():
        if (oid in rows and d.get('status')=='accepted' and d.get('role')=='anchor'
                and d.get('reason')=='body_scene_core' and d.get('person') and not d.get('view_exclusion')):
            body_cores[d['person']].append(rows[oid])
    output = {}
    for oid,d in decisions.items():
        hint = d.get('open_face_root') or {}
        source, owner = hint.get('observation'), hint.get('owner')
        if (source not in with_faces or roots[source]!=owner or hint.get('returning') or oid not in rows or
                d.get('status')=='accepted' or d.get('track_blocker',{}).get('reason') in
                ('different_face_root','competing_transition','unresolved_partner')):
            continue
        row = rows[oid]
        q = quality(row)
        if (q.get('query_only') or q.get('expected_people',0)>q.get('detected_people',1) or
                row.get('face_person') or shots.get(row['cue'])!=shots.get(rows[source]['cue'])):
            continue
        own = [v for key,v in with_faces.items() if roots[key]==owner]
        if copresent([row],own) or not guard.allows(row,own,owner):
            continue
        foreign = [v for key,v in with_faces.items() if roots[key]!=owner]
        rivals = [scores.face_similarity(row['_vector'],v['_vector']) for v in foreign]
        for other,core in body_cores.items():
            if other!=owner and not copresent([row],core):
                rivals.extend(scores.face_similarity(row['_vector'],r['_vector']) for r in core)
        face_rival = max((scores.face_similarity(with_faces[source]['_face_vector'],v['_face_vector'])
                          for v in foreign),default=-1.)
        for (p,kind), witnesses in looks.items():
            if p!=owner:
                continue
            for a,b,w in witnesses:
                if a not in units or b not in units or len({units[source],units[a],units[b]})!=3:
                    continue
                if shots[row['cue']]==shots[w['cue']] or decisions[w['id']].get('view_exclusion'):
                    continue
                body = scores.face_similarity(row['_vector'],w['_vector'])
                if body<settings['match_threshold'] or body-max(rivals,default=-1.)<settings['ambiguity_margin']:
                    continue
                minimum = min(scores.face_similarity(with_faces[source]['_face_vector'],with_faces[k]['_face_vector'])
                              for k in (a,b))
                if minimum>=settings['face_threshold'] and minimum-face_rival>=settings['face_margin']:
                    option = owner,[(a,b,w['id'])],minimum
                    if oid not in output or (-minimum,a,b,w['id']) < (-output[oid][2],*output[oid][1][0]):
                        output[oid] = option
    return output


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
        self.body_match = config.data['sapiensid']['identities']['match_threshold']
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

    def match(self,scan,kind,decision,occupied,ranked,*,reject_rival=None,query=None,allow_open=False):
        import numpy as np
        proof = decision.get('rooted_track') or {}
        roots = proof.get('roots',())
        owner = proof.get('owner')
        single = False
        if not roots and allow_open and query is not None:
            hint = decision.get('open_face_root') or {}
            if not hint.get('returning') and hint.get('observation'):
                owner, roots, single = hint.get('owner'), [hint['observation']], True
        if not owner or owner in occupied or len(roots)!=(1 if single else 2):
            return None
        local = [self.face(root,owner) for root in roots]
        if any(f is None for f in local) or len({f[1] for f in local})!=len(local):
            return None
        for candidate in ranked:
            rival = candidate[2]
            if rival!=owner and max(candidate[0],candidate[5])>=self.body_floor:
                if reject_rival is None or not reject_rival(rival,local):
                    return None
        # Disk galleries fetch one bank at a time; all eligible banks participate.
        best = None
        for index in self.banks.get(owner,()):
            _,_,bank = self.gallery.banks[index]
            for w in bank:
                if w['scan']==scan:
                    continue
                same_kind = w.get('perspective','back')==kind
                if same_kind or single:
                    if query is None:
                        continue
                    body_score = float(query @ w['_vector'])
                    if not np.isfinite(body_score):
                        continue
                    if single and body_score<self.body_match or same_kind and not single and body_score>=self.body_match:
                        continue
                external = [self.face(root,owner) for root in w['roots']]
                if (len(external)!=2 or any(f is None for f in external) or
                        len({f[1] for f in [*external,*local]})!=len(local)+2):
                    continue
                score = min(float(a[0] @ b[0]) for a in local for b in external)
                if np.isfinite(score) and score>=self.minimum:
                    key = (-score,w['scan'],tuple(w['roots']),w.get('id',''))
                    if best is None or key<best[0]:
                        best = key,w
        if best is not None:
            return owner,-best[0][0],best[1]
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
            for other,view,right,units,peer in by_owner[b]:
                if shot==view or pixels & units or not distinct_view(kind,other,lookup.get(member,{}).get('_vector'),
                        lookup.get(peer,{}).get('_vector'),scores,settings):
                    continue
                score = min(scores.face_similarity(lookup[i]['_face_vector'],lookup[j]['_face_vector'])
                            for i in left for j in right)
                if score>=settings['face_threshold'] and (best is None or score>best[0]):
                    best = score,member,left,right
        if best is not None:
            score,member,left,right = best
            yield member,dict(people=[a,b],roots=left+right,minimum=score)
