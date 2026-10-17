"""Read-only, evidence-based diagnostics for fragmented face identities."""
from __future__ import annotations

from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

from .clustering import matching_policy, vector
from .database import APPLICATION_ID, SCHEMA_VERSION, _reject_executable_schema
from .identity_merges import _merge_support_policy
from .models import namespace


def _open_readonly(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Face database does not exist: {path}')
    if path.stat().st_nlink != 1:
        raise ValueError('Recognition SQLite must not be a hard-linked file.')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        if (db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
                or db.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION):
            raise ValueError('Unsupported facial recognition database/version.')
        _reject_executable_schema(db)
        db.execute('BEGIN')  # One consistent snapshot, including during another recognition run.
        return db
    except BaseException:
        db.close()
        raise


def _selected_scan(db, scene, model_namespace, policy):
    scans = db.execute(
        "SELECT id,scope,scene,namespace,policy_sha FROM scans "
        "WHERE scene=? AND namespace=? AND status='complete' ORDER BY rowid DESC",
        (scene, model_namespace)).fetchall()
    if not scans:
        raise ValueError('No completed recognition scan for this scene and recognizer.')
    scopes = {row['scope'] for row in scans}
    if len(scopes) != 1:
        raise ValueError('Scene ID is present in multiple Stash scopes; a scoped report is required.')
    scan = scans[0]
    if not db.execute('SELECT 1 FROM candidates WHERE scan=? AND policy=? LIMIT 1',
                      (scan['id'], policy)).fetchone():
        raise ValueError('Latest scene scan has no grouping evidence for the current face configuration; '
                         'run recognition with this scene/configuration before reporting.')
    return scan


def _scene_identities(db, scan):
    return [dict(row) for row in db.execute(
        'SELECT m.person AS identity_id,p.performer AS performer_id,COUNT(*) AS observations '
        'FROM members m JOIN observations o ON o.id=m.observation '
        'JOIN people p ON p.id=m.person '
        'WHERE o.scan=? AND o.rejected IS NULL '
        'GROUP BY m.person,p.performer ORDER BY m.person', (scan,))]


def _representatives(db, scan, person, maximum):
    rows = db.execute(
        'SELECT o.id,o.seconds,o.dimensions,o.embedding,s.scene '
        'FROM observations o JOIN members m ON m.observation=o.id '
        'JOIN scans s ON s.id=o.scan '
        'WHERE m.person=? AND s.scope=? AND s.namespace=? '
        "AND s.status='complete' AND o.rejected IS NULL "
        'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id LIMIT ?',
        (person, scan['scope'], scan['namespace'], maximum))
    return [(dict(id=row['id'], scene=row['scene'], seconds=row['seconds']), vector(row))
            for row in rows]


def _score(left, right):
    """Equivalent to the flat-group cross mean/minimum checks; not a replay."""
    import numpy as np
    first = np.stack([sample for _, sample in left])
    second = np.stack([sample for _, sample in right])
    if first.shape[1] != second.shape[1]:
        raise ValueError('Incompatible recognition embedding dimensions.')
    values = first @ second.T
    low = np.unravel_index(int(values.argmin()), values.shape)
    # Sum converted float32 dot scores in float64, as the matching code does.
    mean = math.fsum(float(value) for value in values.flat) / values.size
    return mean, float(values[low]), (left[low[0]][0], right[low[1]][0])


def _co_present(db, left, right):
    return db.execute(
        'SELECT 1 FROM members a JOIN observations x ON x.id=a.observation '
        'JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
        'JOIN members b ON b.observation=y.id '
        'WHERE a.person=? AND b.person=? AND x.id<>y.id '
        'AND x.rejected IS NULL AND y.rejected IS NULL LIMIT 1',
        (left, right)).fetchone() is not None


def _recorded_support(db, scan, left, right, references, config):
    """Recheck recorded nominations; missing nominations are not proof of a failed merge."""
    policy = _merge_support_policy(config)
    settings, clustering = config.data['recognition'], config.data['clustering']
    rows = db.execute(
        'SELECT o.id,o.seconds,o.dimensions,o.embedding,s.scene FROM merge_support v '
        'JOIN observations o ON o.id=v.observation JOIN scans s ON s.id=o.scan '
        'WHERE v.policy=? AND ((v.source=? AND v.target=?) OR (v.source=? AND v.target=?)) '
        'AND s.scope=? AND s.namespace=? AND s.policy_sha=? AND s.status=\'complete\' '
        'AND o.rejected IS NULL AND NOT EXISTS ('
        ' SELECT 1 FROM scans newer WHERE newer.scope=s.scope AND newer.namespace=s.namespace '
        ' AND newer.scene=s.scene AND newer.status=\'complete\' AND newer.rowid>s.rowid) '
        'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.seconds,o.id',
        (policy, left, right, right, left, scan['scope'], scan['namespace'], scan['policy_sha']))
    from .clustering import _similarities
    cues = set()
    for row in rows:
        cue = (row['scene'], row['seconds'])
        if cue in cues:
            continue
        face = vector(row)
        if all((mean >= settings['match_threshold'] and low >= clustering['cohesion_threshold'])
               for mean, low in (_similarities(group, face) for group in (
                   type('Reference', (), {'samples': [vec for _, vec in references[person]]})()
                   for person in (left, right)))):
            cues.add(cue)
    return len(cues)


def build(config, scene_id, *, limit=20, min_mean=None):
    """Read only committed state; returned suggestions do not promise a cold-replay merge."""
    if not 1 <= limit <= 500:
        raise ValueError('Report limit must be 1..500.')
    settings, clustering = config.data['recognition'], config.data['clustering']
    if min_mean is None:
        min_mean = settings['match_threshold'] - .20
    if not -1 <= min_mean <= 1 or not math.isfinite(min_mean):
        raise ValueError('Report min-mean must be between -1 and 1.')
    policy = matching_policy(config)
    with closing(_open_readonly(config.database)) as db:
        scan = _selected_scan(db, str(scene_id), namespace(config), policy)
        identities = _scene_identities(db, scan['id'])
        references = {person['identity_id']: _representatives(
            db, scan, person['identity_id'], clustering['max_samples_per_person']) for person in identities}
        if any(not records for records in references.values()):
            raise ValueError('A scene identity has no valid retained reference embeddings.')
        decisions = [dict(row) for row in db.execute(
            "SELECT json_extract(payload,'$.reason') AS reason,status,COUNT(*) AS count "
            "FROM candidates WHERE scan=? AND policy=? "
            "GROUP BY json_extract(payload,'$.reason'),status ORDER BY count DESC,reason",
            (scan['id'], policy))]
        total, rejected = db.execute(
            'SELECT COUNT(*),COUNT(*) FILTER (WHERE rejected IS NOT NULL) '
            'FROM observations WHERE scan=?', (scan['id'],)).fetchone()
        people = {item['identity_id']: item for item in identities}
        ids = list(people)
        pairs = []
        for n, left in enumerate(ids):
            for right in ids[n + 1:]:
                mean, minimum, worst = _score(references[left], references[right])
                if mean < min_mean:
                    continue
                blockers = []
                if mean < settings['match_threshold']:
                    blockers.append('cross_mean_below_match_threshold')
                if minimum < clustering['cohesion_threshold']:
                    blockers.append('cross_sample_below_cohesion_threshold')
                pairs.append({
                    'left': people[left], 'right': people[right],
                    'cross_mean': round(mean, 6), 'cross_minimum': round(minimum, 6),
                    'worst_representatives': {'left': worst[0], 'right': worst[1]},
                    'blockers': blockers,
                    'indicative_thresholds': {
                        'match_threshold_at_most': round(mean, 6),
                        'cohesion_threshold_at_most': round(minimum, 6)},
                })
        pairs.sort(key=lambda pair: (-pair['cross_mean'], -pair['cross_minimum'],
                                     pair['left']['identity_id'], pair['right']['identity_id']))
        selected_pairs = pairs[:limit]
        for pair in selected_pairs:
            a, b = pair['left']['identity_id'], pair['right']['identity_id']
            co_present = _co_present(db, a, b)
            if co_present and pair['cross_minimum'] < settings['cooccurrence_merge_threshold']:
                pair['blockers'].append('co_present_requires_stronger_cross_similarity')
            pair['co_present_observed'] = co_present
            votes = _recorded_support(db, scan, a, b, references, config)
            pair['recorded_merge_support'] = {
                'validated_nominations': votes, 'required': settings['min_merge_support'],
                'complete_replay_proof': False,
                'note': 'Only stored nominations surviving current checks; missing votes do not '
                        'establish why a cold replay would reject this pair.'}
            pair['can_merge'] = None  # Not inferable from these diagnostics alone.
        return {
            'format': 1, 'kind': 'recognition_fragmentation',
            'scene_id': str(scene_id), 'scope': scan['scope'], 'scan_id': scan['id'],
            'namespace': scan['namespace'], 'matching_policy': policy,
            'thresholds': {
                'match_threshold': settings['match_threshold'],
                'cohesion_threshold': clustering['cohesion_threshold'],
                'ambiguity_margin': settings['ambiguity_margin'],
                'cooccurrence_merge_threshold': settings['cooccurrence_merge_threshold'],
                'min_merge_support': settings['min_merge_support'],
                'same_scene_match_bonus': clustering['same_scene_match_bonus']},
            'summary': {'observations': total, 'rejected': rejected,
                        'identities': len(identities), 'pairs_checked': len(ids)*(len(ids)-1)//2,
                        'pairs_above_min_mean': len(pairs), 'pairs_reported': len(selected_pairs)},
            'decisions': decisions, 'identities': identities,
            'potential_fragments': selected_pairs,
            'limitations': [
                'Pair checks use current bounded references, not a complete cold replay.',
                'A successful pairwise comparison does not guarantee an identity merge; '
                'nominations, co-presence, ambiguity and constituent history still matter.',
                'Only identities assigned to the selected scene are compared; withheld '
                'observations appear in the decisions summary, not as separate identities.',
                'Recorded merge support is diagnostic, not a proof of future merging.',
            ],
        }


def execute(args, config):
    result = build(config, args.scene_id, limit=args.limit, min_mean=args.min_mean)
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        target = Path(args.output).expanduser().resolve()
        if target == config.database.resolve():
            raise ValueError('Report output must not overwrite the face database.')
        with target.open('x', encoding='utf-8') as file:
            file.write(output)
        print(f'Face fragmentation report written: {target}')
    return 0
