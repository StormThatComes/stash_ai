"""Validate face-rooted fragments nominated by the frozen face catalogue."""
from functools import lru_cache
import hashlib
import heapq
import json
from itertools import combinations

from .body_identity import policy
from .fragment_endpoints import enabled
from .scan_heads import predicate


def remember(best, scenes, proof, sources):
    """Retain a canonical best example and distinct participating source scans."""
    key = tuple(proof['people'])
    order = (-proof['minimum'], tuple(sources), tuple(proof['roots']))
    if key not in best or order < best[key][0]:
        best[key] = order, dict(proof, source_scans=list(sources))
    scenes.setdefault(key,set()).update(sources)


def verified(db, config, scope, namespace, bank, result):
    settings = config.data['sapiensid']['identities']
    if not enabled(settings) or len(bank)<2:
        return
    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    from .global_views import consolidation_plan
    from .strict_partition import summary_views
    from .clustering import _groups, vector
    from .reference_usage import ReferenceUsage
    from .body_global_search import Search, exact_pair
    from .resource_errors import ResourceLimit
    from .independent_evidence import unit
    from .body_identity import Scores, vector as body_vector

    stamp = policy(config)
    occupied = {p for d in result['decisions'] if d['outcome']=='merge' for p in d['people']}
    eligible = bank.keys()-occupied
    if len(eligible)<2:
        return
    best, scenes = {}, {}
    sql = """SELECT o.* FROM body_observations b JOIN observations o ON o.id=b.face AND o.scan=b.scan
        JOIN members m ON m.observation=o.id WHERE b.id=? AND b.scan=? AND m.person=? AND b.rejected IS NULL
        AND o.rejected IS NULL AND """ + strict_sql()+' AND '+anchor_sql('o',config.data['clustering'])

    @lru_cache(maxsize=128)
    def face(scan, oid, owner):
        row = db.execute(sql,(oid,scan,owner)).fetchone()
        return (vector(row),unit(row)) if row else None

    scores = Scores(config)
    threshold = config.data['recognition']['match_threshold']
    def validate(pair, roots, sources):
        entries = [face(scan,oid,p) for p,scan,ids in zip(pair,sources,(roots[:2],roots[2:])) for oid in ids]
        if len(entries)!=4 or any(f is None for f in entries) or len({f[1] for f in entries})!=4:
            return
        minimum = min(scores.face_similarity(a[0],b[0]) for a in entries[:2] for b in entries[2:])
        if minimum>=threshold:
            remember(best,scenes,dict(people=list(pair), roots=list(roots), minimum=minimum),sources)
            if len(best)>config.data['sapiensid']['execution']['max_candidate_pairs']:
                raise ResourceLimit('Too many distinct cross-perspective endpoint nominations.')

    rows = db.execute("""SELECT b.scan,j.value FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan JOIN json_each(m.proof,'$.endpoint_pairs') j
        WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND m.policy=? AND json_extract(m.proof,'$.endpoint_pairs') IS NOT NULL AND """+predicate(),(scope,namespace,stamp))
    for scan,raw in rows:
        proof = json.loads(raw)
        pair = tuple(proof['people'])
        if len(pair)==2 and set(pair)<=eligible:
            validate(pair,proof['roots'],(scan,scan))

    # First use indexed face representatives to nominate identities. Compare
    # original endpoint vectors only inside pairs that passed that face plan.
    focus = {p for pair in best for p in pair}
    tracks_sql = """FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND m.policy=? AND
        m.person=json_extract(m.proof,'$.rooted_track.owner') AND b.rejected IS NULL AND
        json_extract(m.proof,'$.view_exclusion') IS NULL AND """+predicate()
    focus.update(owner for owner, in db.execute('SELECT DISTINCT m.person '+tracks_sql,(scope,namespace,stamp))
                 if owner in eligible)
    if len(focus)<2:
        return
    cap = config.data['sapiensid']['global_matching']['max_references']
    @lru_cache(maxsize=64)
    def representatives(owner):
        rows = db.execute("""SELECT b.scan,b.id,json_extract(m.proof,'$.rooted_track.roots') AS roots,
            json_extract(m.proof,'$.rooted_track.perspective') AS kind,
            CAST(json_extract(b.quality,'$.quality.sharpness') AS REAL) AS sharpness """+tracks_sql+
            ' AND m.person=? ORDER BY b.scan,kind,sharpness DESC,b.cue,b.id',(scope,namespace,stamp,owner))
        from itertools import groupby
        def strata():
            for (scan,kind),group in groupby(rows,key=lambda r:(r[0],r[3])):
                priority = hashlib.sha256(f'{scan}:{kind}'.encode()).digest()
                from .endpoint_references import representatives_for_view
                for order,oid,roots in representatives_for_view(db,group,owner,scan,kind,stamp,settings,face,scores):
                    yield (order,priority),scan,kind,oid,tuple(roots)
        # Keep a complementary look even when many scans fill the ordinary
        # reference budget. Both heaps store identifiers, never embeddings.
        main, alternate = [], []
        for entry in strata():
            pool = main if entry[0][0]==0 else alternate
            pool.append(entry)
            if len(pool)>2*cap:
                pool[:] = heapq.nsmallest(cap,pool)
        main, alternate = sorted(main),sorted(alternate)
        reserved = min(len(alternate), cap//3 or int(cap>1))
        primaries = main[:cap-reserved]
        return primaries + alternate[:cap-len(primaries)]

    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    views = summary_views(_groups(db,scope,namespace,config.data['clustering'],usage=usage),usage)
    face_plan = consolidation_plan(db,views,config,focus=focus)
    def verified_pair(pair, search):
        a,b = pair
        left, right = representatives(a), representatives(b)
        for _,scan,kind,oid,roots in left:
            for _,other,view,peer,ends in right:
                if scan==other:
                    continue
                if kind==view:
                    # Same-angle duplicates cannot certify a distinct appearance.
                    first,second = (body_vector(db.execute('SELECT embedding FROM body_observations WHERE id=?',
                                                          (i,)).fetchone()) for i in (oid,peer))
                    if scores.face_similarity(first,second)>=settings['match_threshold']:
                        continue
                validate(pair,roots+ends,(scan,other))
        if pair not in best:
            return None
        evidence = exact_pair(*pair, bank, config, search.evidence)
        if set(evidence['blockers']) - {'body_similarity', 'body_cohesion'} or evidence['face_minimum'] is None:
            return None
        return dict(best[pair][1], supporting_scans=len(scenes[pair]))

    with Search(db, config, bank) as search:
        for decision in face_plan:
            people = sorted(decision['people'])
            if decision['outcome'] != 'merge' or not set(people) <= eligible:
                continue
            if len(people) * (len(people)-1) // 2 > config.data['sapiensid']['execution']['max_candidate_pairs']:
                raise ResourceLimit('Too many pairs within a face endpoint nomination.')
            edges = {}
            neighbours = {owner: set() for owner in people}
            for pair in combinations(people, 2):
                proof = verified_pair(pair, search)
                if proof is not None:
                    edges[pair] = proof
                    a, b = pair
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            # Only publish isolated complete components. A-B and B-C alone
            # cannot prove A-C or authorize two overlapping alias writes.
            for a in people:
                group = sorted({a} | neighbours[a])
                if len(group) < 2 or group[0] != a:
                    continue
                members = set(group)
                if any(neighbours[p] != members - {p} for p in group):
                    continue
                proofs = [edges[pair] for pair in combinations(group, 2)]
                result['decisions'].append(dict(people=group, outcome='merge', blockers=[],
                    endpoint_pair=proofs[0], **({'endpoint_pairs': proofs} if len(proofs)>1 else {}),
                    cores={p:list(dict.fromkeys(r['id'] for core in bank[p]['cores'] for r in core)) for p in group},
                    face_witnesses={p:list(bank[p]['face_ids']) for p in group}))
