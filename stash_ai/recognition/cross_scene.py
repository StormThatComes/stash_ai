"""Frozen, conservative cross-scene confirmation of provisional anchor groups.

This pass follows strict cold replay. It never merges established identities,
uses local thresholds, or turns its own assignments into further reference
evidence. A provisional person's samples remain uncommitted until the minimum
independent support is met across at least two different selected scenes.
"""
from collections import defaultdict
from dataclasses import dataclass
import uuid

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


def _commit(db, policy, sources, person, reason, proof):
    for source in sources:
        for row, _ in source.rows:
            db.execute(
                "UPDATE candidates SET person=?,status='accepted',"
                "payload=json_set(payload,'$.reason',?,'$.cross_scene',json(?)) "
                f'WHERE policy=? AND {address(db, "candidates") }=?'
                if False else
                "UPDATE candidates SET person=?,status='accepted',"
                "payload=json_set(payload,'$.reason',?,'$.cross_scene',json(?)) "
                "WHERE policy=? AND id=? AND status='withheld'",
                (person, reason, encoded(proof), policy, row['candidate']))
            db.execute('INSERT OR IGNORE INTO members VALUES (?,?)', (row['id'], person))
    # Explicitly establish the new person only *after* independent support.
    db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
               (person, proof['scope'], proof['namespace'], 0., None,
                'stash-ai-provisional:' + person))


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

    # Other provisional sources can corroborate each other, but never through
    # transitive single-link: every cluster merger rechecks *every* cross pair.
    pool = [g for g in provisional if g.person not in attached]
    # Explicitly separate multi-scene evidence from individual same-scene cues.
    clusters = [[g] for g in pool]
    frozen = [(g, _rank_reference(g.samples, references, match, cohesion)) for g in pool]
    barred = {g.person for g, ranked in frozen if ranked and ranked[0][0] >= match-margin}
    def score(left, right):
        a = [v for g in left for v in g.samples]
        b = [v for g in right for v in g.samples]
        return _cross_score(a, b)
    def scenes_of(items):
        return set().union(*(g.scenes for g in items))
    with tracking(range(len(clusters)), 'Confirm cross-scene provisional groups',
                  unit='group') as tracked:
        for _ in tracked:
            # Pair only mutually unambiguous closest groups. Re-evaluate
            # prototypes after every accepted pair; an existing identity rival
            # vetoes any would-be provisional match.
            pairs = {}
            for i, left in enumerate(clusters):
                if not left or any(g.person in barred for g in left):
                    continue
                options = []
                for j, right in enumerate(clusters):
                    if i == j or not right or any(g.person in barred for g in right):
                        continue
                    if not scenes_of(left).isdisjoint(scenes_of(right)):
                        continue
                    mean, low = score(left, right)
                    if low >= cohesion and mean >= match-margin:
                        options.append((mean, j))
                options.sort(key=lambda item:(-item[0],item[1]))
                if options and options[0][0] >= match and (
                        len(options)==1 or options[0][0]-options[1][0] >= margin):
                    pairs[i] = options[0]
            matched = False
            for i, (mean,j) in sorted(pairs.items()):
                if j in pairs and pairs[j][1] == i and i < j:
                    clusters[i].extend(clusters[j])
                    clusters[j] = []
                    matched = True
            if not matched:
                break
    # Never commit partial provisional clusters, nor two cues from one scene.
    for group in clusters:
        if not group or len(scenes_of(group)) < 2:
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
                 'scenes':sorted(scenes_of(group)),
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
            _commit(db, policy, group, leader, 'cross_scene_provisional', proof)
