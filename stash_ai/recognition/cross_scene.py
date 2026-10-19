"""Frozen, conservative cross-scene confirmation of provisional anchor groups.

This pass follows strict cold replay. It never merges established identities,
uses local thresholds, or turns its own assignments into further reference
evidence. A provisional person's samples remain uncommitted until the minimum
independent support is met across at least two different selected scenes.
"""
from collections import defaultdict
from dataclasses import dataclass
import time

from .clustering import _groups, _similarities, co_presence_allows, matching_policy, vector
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
    ranked = []
    for reference in references:
        means, lows = zip(*(_similarities(reference, query) for query in samples))
        mean, minimum = sum(means) / len(means), min(lows)
        if minimum >= cohesion and mean >= match - 0.15:
            ranked.append((mean, reference.identifier, reference))
    ranked.sort(key=lambda v: (-v[0], v[1]))
    return ranked


def _has_other_scene(db, person, scenes):
    return db.execute(
        'SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
        'WHERE m.person=? AND o.scan NOT IN (SELECT value FROM json_each(?)) LIMIT 1',
        (person, encoded(sorted(scenes)))).fetchone() is not None


def _same_cue_rival(db, row, person, policy):
    return db.execute(
        f'SELECT 1 FROM candidates c JOIN observations o ON o.id={address(db, "c")} '
        "WHERE c.policy=? AND c.scan=? AND c.status='accepted' "
        'AND c.person<>? AND o.cue=? LIMIT 1',
        (policy, row['scan'], person, row['cue'])).fetchone() is not None


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
            if any(_same_cue_rival(db, row, target.identifier, policy)
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
    for i, group in enumerate(pool):
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
    for offset in range(0, n, 128):
        block = first_vectors[offset:offset+128] @ first_vectors.T
        for local, scores in enumerate(block):
            i = offset+local
            if i in barred:
                continue
            for j in np.flatnonzero(scores[i+1:] >= cohesion) + i+1:
                if j in barred or not pool[i].scenes.isdisjoint(pool[j].scenes):
                    continue
                mean, low = _cross_score(pool[i].samples, pool[j].samples)
                if low < cohesion or mean < match-margin:
                    continue
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
                continue
            if len(set().union(*(pool[i].scenes for i in component))) != sum(
                    len(pool[i].scenes) for i in component):
                continue
            clusters.append([pool[i] for i in component])
    # Never commit partial provisional clusters, nor two cues from one scene.
    for group in clusters:
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
                if _same_cue_rival(db,row,leader,policy):
                    break
            else:
                continue
            break
        else:
            _commit(db, policy, group, leader, 'cross_scene_provisional', proof,
                    created=getattr(database, 'identity_clock', time.time)())
