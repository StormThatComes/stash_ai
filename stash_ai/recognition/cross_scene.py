"""Frozen, conservative cross-scene confirmation of provisional anchor groups.

This pass follows strict cold replay. It never merges established identities,
uses local thresholds, or turns its own assignments into further reference
evidence. A provisional person's samples remain uncommitted until the minimum
independent support is met across at least two different selected scenes.
"""
from collections import defaultdict
from dataclasses import dataclass
import time

from .clustering import _groups, co_presence_allows, matching_policy, vector
from .database import encoded
from .pose_policy import anchor_sql
from .progress import tracking
from .reference_usage import ReferenceUsage
from .replay_addresses import address


@dataclass
class Provisional:
    person: str
    rows: list
    scenes: frozenset

    @property
    def samples(self):
        return [query for _, query in self.rows]

    @property
    def independent(self):
        return {(row['scan'], row['seconds']) for row, _ in self.rows}


def _cross_score(left, right):
    """Every pair contributes; one weak edge prevents complete-link fusion."""
    import numpy as np
    scores = [float(np.dot(a, b)) for a in left for b in right]
    return (sum(scores) / len(scores), min(scores))


def _rank_reference(samples, references, match, cohesion):
    # Reuse the coordinator's native BLAS screening of immutable reference
    # blocks, while retaining the exact scalar comparisons for every finalist.
    # Every source sample must satisfy full-reference cohesion.
    from .scoring import comparisons
    by_person = {}
    for query in samples:
        for reference, mean, minimum in comparisons(
                references, query, float('-inf'), cohesion):
            if minimum >= cohesion:
                by_person.setdefault(reference.identifier, []).append(
                    (mean, minimum, reference))
    ranked=[]
    for _, evidence in by_person.items():
        if len(evidence)!=len(samples):
            continue
        mean=sum(item[0] for item in evidence)/len(evidence)
        if mean >= match-0.15:
            ranked.append((mean,evidence[0][2].identifier,evidence[0][2]))
    ranked.sort(key=lambda v:(-v[0],v[1]))
    return ranked


def _has_other_scene(db, person, scenes):
    return db.execute(
        'SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
        'WHERE m.person=? AND o.scan NOT IN (SELECT value FROM json_each(?)) LIMIT 1',
        (person, encoded(sorted(scenes)))).fetchone() is not None


def _same_cue_rival(db, row, person, policy, config):
    peers=db.execute(
        f'SELECT o.geometry,o.embedding FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
        "WHERE c.policy=? AND c.scan=? AND c.status='accepted' "
        'AND c.person=? AND o.cue=? AND o.id<>?',
        (policy, row['scan'], person, row['cue'], row['id']))
    from .physical_faces import duplicate
    return any(not duplicate(row,peer,config.data['clustering'].get('deduplicate_detections',False)) for peer in peers)


def _commit(db, policy, sources, person, reason, proof, *, created):
    # Ensure FK constraints before adding the first member.
    db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
               (person, proof['scope'], proof['namespace'], created, None,
                'stash-ai-provisional:' + person))
    for source in sources:
        for row, _ in source.rows:
            db.execute(
                "UPDATE candidates SET person=?,status='accepted',"
                "payload=json_set(payload,'$.reason',?,'$.cross_scene',json(?)) "
                "WHERE policy=? AND id=? AND status='withheld'",
                (person, reason, encoded(proof), policy, row['candidate']))
            db.execute('INSERT OR IGNORE INTO members VALUES (?,?)', (row['id'], person))

def apply_cross_scene_support(database, selected, config, *, progress=None):
    settings, clustering = config.data['recognition'], config.data['clustering']
    if not settings['cross_scene_provisional'] or len(selected) < 2:
        return
    db, policy = database.connection, matching_policy(config)
    match, cohesion, margin = (settings['match_threshold'],
                              clustering['cohesion_threshold'],
                              settings['ambiguity_margin'])
    scopes = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    usage = ReferenceUsage(clustering['max_reference_bytes'], progress)
    # Strict accepted references are a frozen snapshot for this pass.
    references = _groups(db, scopes['scope'], scopes['namespace'], clustering,
                         usage=usage, stage='cross_scene_provisional')
    anchor = anchor_sql('o', clustering)
    rows = db.execute(
        f'SELECT c.id AS candidate,c.person,o.* FROM candidates c '
        f'JOIN observations o ON o.id={address(db, "c")} '
        "WHERE c.policy=? AND c.status='withheld' AND c.person IS NOT NULL "
        "AND json_extract(c.payload,'$.reason')='candidate_group' "
        f'AND o.rejected IS NULL AND {anchor} '
        'ORDER BY c.person,o.scan,o.cue,o.seconds,o.id', (policy,))
    pending = defaultdict(list)
    with tracking(rows, 'Read provisional cross-scene anchors',
                  unit='observation') as tracked:
        for row in tracked:
            pending[row['person']].append((row, vector(row)))
    provisional = [Provisional(person, evidence,
                    frozenset(row['scan'] for row, _ in evidence))
                   for person, evidence in sorted(pending.items())]
    if not provisional:
        return
    if len(provisional) > clustering['max_people_per_lookup']:
        raise ValueError('Cross-scene provisional candidates exceed max_people_per_lookup.')
    # First, attach coherent provisional cues to a *previously confirmed*
    # identity. The destination must already have a reference in another scene.
    attached = set()
    with tracking(provisional, 'Attach cross-scene provisional anchors',
                  unit='group') as tracked:
        for group in tracked:
            ranked = _rank_reference(group.samples, references, match, cohesion)
            if not ranked or ranked[0][0] < match:
                continue
            if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin:
                continue
            target = ranked[0][2]
            if not _has_other_scene(db, target.identifier, group.scenes):
                continue
            if any(_same_cue_rival(db, row, target.identifier, policy, config)
                   or not co_presence_allows(db, row, target, references, query, config)
                   for row, query in group.rows):
                continue
            proof = {'format':1,'scope':scopes['scope'],'namespace':scopes['namespace'],
                     'source_scenes':sorted(group.scenes),'target':target.identifier,
                     'similarity':round(ranked[0][0],6),
                     'support':len(group.independent),'frozen':True}
            # Existing people already carry verified independent support.
            for row, _ in group.rows:
                db.execute(
                    "UPDATE candidates SET person=?,status='accepted',"
                    "payload=json_set(payload,'$.reason','cross_scene_identity',"
                    "'$.cross_scene',json(?)) WHERE id=? AND policy=? AND status='withheld'",
                    (target.identifier, encoded(proof), row['candidate'], policy))
                db.execute('INSERT OR IGNORE INTO members VALUES (?,?)',
                           (row['id'], target.identifier))
            attached.add(group.person)

    # Build a frozen compatibility graph. All nodes in a component must form
    # a *complete clique*, not merely be connected by a near-neighbour chain.
    # A marginal third candidate makes the whole component unsafe, rather than
    # allowing input-order-dependent greedy pair formation.
    pool = [g for g in provisional if g.person not in attached]
    n = len(pool)
    # Union-find stores O(n) component membership; clique checks need only
    # each node's number of strong edges, never an O(n^2) edge table.
    parents=list(range(n))
    sizes=[1]*n
    strong_counts=[0]*n
    from .frozen_subgroups import Edges, nonoverlapping
    edges=Edges(clustering['max_reference_bytes']//4)
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    def join(i,j):
        a,b=find(i),find(j)
        if a==b:
            return
        if sizes[a]<sizes[b]:
            a,b=b,a
        parents[b]=a
        sizes[a]+=sizes[b]
    barred = set()
    with tracking(pool, 'Check provisional identity competition', unit='group') as tracked:
        for i, group in enumerate(tracked):
            ranked = _rank_reference(group.samples, references, match, cohesion)
            if ranked and ranked[0][0] >= match-margin:
                barred.add(i)
    # Native BLAS prefilter on the first sample is *exactly safe* for the
    # complete-link cohesion bound: if it falls below cohesion, the pair
    # cannot be joined, regardless of the other members.
    import numpy as np
    dims = {v.shape[0] for g in pool for v in g.samples}
    if len(dims)>1:
        raise ValueError('Cross-scene provisional group has mixed embedding dimensions.')
    first_vectors = (np.stack([g.samples[0] for g in pool]).astype('float32')
                     if pool else np.empty((0,0),dtype='float32'))
    # Never allow f32 BLAS rounding to screen a pair that exact np.dot
    # would accept. The upper error bound follows the native scorer's
    # gamma_d rule; all retained pairs receive exact scalar rescoring.
    dimension=first_vectors.shape[1] if n else 0
    unit=np.finfo(np.float32).eps/2
    gamma=dimension*unit/(1-dimension*unit) if dimension else 0.
    multiplier=4*gamma+16*np.finfo(np.float64).eps
    norms=np.linalg.norm(first_vectors.astype(np.float64),axis=1)
    with tracking(range(0, n, 128), 'Compare provisional identity blocks', unit='block', position=0) as tracked:
        for offset in tracked:
            block = first_vectors[offset:offset+128] @ first_vectors.T
            with tracking(block, 'Compare provisional identity groups', unit='group') as groups:
                for local, scores in enumerate(groups):
                    i = offset+local
                    if i in barred:
                        continue
                    errors=multiplier*norms[i]*norms + dimension*np.finfo(np.float32).tiny
                    for j in np.flatnonzero(scores[i+1:] + errors[i+1:] >= cohesion) + i+1:
                        if j in barred or not pool[i].scenes.isdisjoint(pool[j].scenes):
                            continue
                        mean, low = _cross_score(pool[i].samples, pool[j].samples)
                        if low < cohesion or mean < match-margin:
                            continue
                        edges.add(i,int(j),mean)
                        join(i,int(j))
                        if mean >= match:
                            strong_counts[i]+=1
                            strong_counts[j]+=1
    components=defaultdict(list)
    for i in range(n):
        if i not in barred:
            components[find(i)].append(i)
    clusters=[]
    with tracking(sorted(components.items()), 'Confirm cross-scene provisional groups',
                  unit='component') as tracked:
        for _,component in tracked:
            if len(component)<2:
                continue
            # Every member must have strict evidence to every other member:
            # no transitive "A matches B, B matches C" membership.
            if any(strong_counts[i] != len(component)-1 for i in component):
                proposed=[]
                for members in edges.proposals(component,match,margin):
                    floors=edges.floors(members)
                    if any(floors[i]-edges.score(i,j)<margin for i in members for j in component if j not in members):
                        continue
                    proposed.append(members)
                for members in nonoverlapping(proposed):
                    clusters.append([pool[i] for i in members])
                continue
            if len(set().union(*(pool[i].scenes for i in component))) != sum(
                    len(pool[i].scenes) for i in component):
                continue
            clusters.append([pool[i] for i in component])
    # Never commit partial provisional clusters, nor two cues from one scene.
    with tracking(clusters, 'Commit provisional identity groups', unit='group', position=1) as tracked:
        for group in tracked:
            if len(set().union(*(g.scenes for g in group))) < 2:
                continue
            independent = {(row['scan'], row['seconds']) for g in group for row,_ in g.rows}
            if len(independent) < settings['min_identity_support']:
                continue
            # Check no competition with accepted people and no co-presence
            # conflicts; all membership changes are decided from frozen evidence.
            queries = [v for g in group for v in g.samples]
            ranked = _rank_reference(queries, references, match, cohesion)
            if ranked and ranked[0][0] >= match-margin:
                continue
            leader = min(group, key=lambda g:(min(g.scenes),g.person)).person
            proof = {'format':1,'scope':scopes['scope'],'namespace':scopes['namespace'],
                     'sources':sorted(g.person for g in group),
                     'scenes':sorted(set().union(*(g.scenes for g in group))),
                     'support':len(independent),'frozen':True}
            # Do not borrow the relaxed same-scene threshold. New provisional
            # people have no pre-existing members and cannot act as merge votes.
            for g in group:
                for row,_ in g.rows:
                    if _same_cue_rival(db,row,leader,policy,config):
                        break
                else:
                    continue
                break
            else:
                _commit(db, policy, group, leader, 'cross_scene_provisional', proof,
                        created=getattr(database, 'identity_clock', time.time)())
