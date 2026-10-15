"""Cold replay of a selected evidence set, followed by stable identity binding.

Historical identities never seed the replay. Original sample ownership is kept
separately so a later split can reactivate profiles retired by consolidation.
"""
from contextlib import closing
import json
import itertools
import sqlite3
import time
from types import SimpleNamespace
import uuid

from .clustering import _groups, _result, _similarities, cluster_scene, matching_policy, reference_context, vector
from .config import digest
from .database import SCHEMA, encoded
from .reference_usage import ReferenceUsage
from .progress import tracking, work
from .ranking import best_two


def sample_key(row):
    return digest([row['scope'], row['namespace'], row['scene'], row['input_sha'],
                   row['cue'], row['seconds'], row['source'], row['rect'], row['geometry']])


def _observations(db, scan):
    return db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'
                      'o.rejected,o.dimensions,o.embedding,s.scope,s.namespace,s.scene,s.input_sha FROM observations o '
                      'JOIN scans s ON s.id=o.scan WHERE o.scan=? ORDER BY o.cue,o.id', (scan,))


def _batches(rows, size=256):
    iterator = iter(rows)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def _inserted(db, sql, rows):
    """Expose completed work only after each bounded SQL batch succeeds."""
    for batch in _batches(rows):
        db.executemany(sql, batch)
        yield from batch


def _copy_observations(db, replay, scan):
    for batch in _batches(_observations(db, scan)):
        # Source/rect are consumed by sample_key on the authoritative row.
        # Grouping needs geometry, time and embeddings, never quality or JPEGs.
        replay.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           ((row['id'], row['scan'], row['cue'], row['seconds'], '', '[]',
                             row['geometry'], '{}', row['rejected'], row['dimensions'],
                             row['embedding'], None) for row in batch))
        replay.executemany('INSERT INTO sample_keys VALUES (?,?)',
                           ((row['id'], sample_key(row)) for row in batch))
        yield from batch


def _restore_origins(db, replay, scan):
    keys = replay.execute('SELECT k.sample_key FROM sample_keys k JOIN observations o '
                          'ON o.id=k.observation WHERE o.scan=?', (scan,))
    for batch in _batches(keys):
        old = db.execute('SELECT * FROM identity_origins WHERE sample_key IN '
                         '(SELECT value FROM json_each(?))', (encoded([row[0] for row in batch]),))
        replay.executemany('INSERT OR REPLACE INTO origins VALUES (?,?,?)', old)
        yield from batch


def _apply_members(db, replay, mapping):
    rows = replay.execute('SELECT m.observation,m.person,k.sample_key FROM members m '
                          'JOIN sample_keys k ON k.observation=m.observation')
    for batch in _batches(rows):
        db.executemany('INSERT INTO members VALUES (?,?)',
                       ((row['observation'], mapping[row['person']]) for row in batch))
        db.executemany('INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)',
                       ((row['sample_key'], mapping[row['person']], row['observation']) for row in batch))
        yield from batch


def _mapped_candidates(replay, mapping, policy):
    for row in replay.execute('SELECT * FROM candidates WHERE policy=?', (policy,)):
        payload = json.loads(row['payload'])
        payload['identity_basis'] = 'cold_selected_replay'
        yield (row['id'], row['scan'], policy, mapping.get(row['person']), row['status'], encoded(payload))


def _copy_selection(database, replay, selected, scope, namespace):
    db = database.connection
    replay.executescript(SCHEMA + '''
      CREATE TABLE sample_keys(observation TEXT PRIMARY KEY, sample_key TEXT NOT NULL);
      CREATE TABLE origins(sample_key TEXT PRIMARY KEY, person TEXT, observation TEXT);
      CREATE TABLE old_people(id TEXT PRIMARY KEY, created REAL, performer TEXT);
    ''')
    with tracking(
        db.execute(
            'SELECT id,created,performer FROM people WHERE scope=? AND namespace=?',
            (scope, namespace),
        ),
        'Copy historical identity labels',
        unit='person',
        position=0,
        total=lambda: db.execute(
            'SELECT COUNT(*) FROM people WHERE scope=? AND namespace=?',
            (scope, namespace),
        ).fetchone()[0],
    ) as tracked:
        for row in tracked:
            replay.execute('INSERT INTO old_people VALUES (?,?,?)', tuple(row))
    with tracking(
        selected,
        'Copy selected evidence',
        unit='scene',
        position=0,
    ) as tracked:
        for item in tracked:
            record = db.execute('SELECT * FROM scans WHERE id=?', (item['scan'],)).fetchone()
            if (record is None or record['status'] != 'complete' or record['scope'] != scope
                    or record['namespace'] != namespace or record['scene'] != item['scene']):
                raise ValueError('Identity rebuild requires complete compatible selected scans.')
            replay.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', tuple(record))
            with tracking(
                _copy_observations(db, replay, item['scan']),
                'Copy face observations',
                unit='observation',
                total=lambda: db.execute(
                    'SELECT COUNT(*) FROM observations WHERE scan=?',
                    (item['scan'],),
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass  # The bounded copy iterator has completed both inserts.
            # Seed origins from old memberships only when no older origin exists.
            # Superseded scans are useful for identity labels, never grouping evidence.
            historical = db.execute(
                    'SELECT '
                    'o.cue,o.seconds,o.source,o.rect,o.geometry,o.id,s.scope,s.namespace,s.scene,s.input_sha,m.person'
                    ' FROM members m JOIN observations o ON o.id=m.observation JOIN scans s ON '
                    's.id=o.scan WHERE s.scope=? AND s.namespace=? AND s.scene=? ORDER BY '
                    's.rowid,o.cue,o.id',
                    (scope, namespace, item['scene']),
                )
            with tracking(
                _inserted(replay, 'INSERT OR IGNORE INTO origins VALUES (?,?,?)',
                          ((sample_key(row), row['person'], row['id']) for row in historical)),
                'Read historical sample ownership',
                unit='member',
                total=lambda: db.execute(
                    'SELECT COUNT(*) FROM members m JOIN observations o ON o.id=m.observation JOIN '
                    'scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? AND s.scene=?',
                    (scope, namespace, item['scene']),
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
            with tracking(
                _restore_origins(db, replay, item['scan']),
                'Restore original sample ownership',
                unit='observation',
                total=lambda: replay.execute(
                    'SELECT COUNT(*) FROM observations WHERE scan=?',
                    (item['scan'],),
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
    with work('Commit replay evidence', 1, unit='transaction', position=0) as committed:
        replay.commit()
        committed.advance()


def _bind(database, replay, config, scope, namespace):
    """One historical identity per new cluster; IDs cannot change the partition."""
    with work('Compute identity votes', 1, unit='query', position=0) as votes:
        replay.executescript('''CREATE TABLE votes AS
          SELECT m.person AS cluster,o.person AS person,COUNT(*) AS total
          FROM members m JOIN sample_keys k ON k.observation=m.observation
          JOIN origins o ON o.sample_key=k.sample_key GROUP BY m.person,o.person;''')
        votes.advance()
    mapping, used, performers = {}, set(), set()
    # Oldest wins a consolidation. On a split its largest overlapping piece wins.
    with tracking(
        replay.execute(
            'SELECT v.*,p.performer FROM votes v JOIN old_people p ON p.id=v.person ORDER BY '
            'p.created,p.id,v.total DESC,v.cluster',
        ),
        'Bind historical identity votes',
        unit='vote',
        position=0,
        total=lambda: replay.execute(
            'SELECT COUNT(*) FROM votes v JOIN old_people p ON p.id=v.person',
        ).fetchone()[0],
    ) as tracked:
        for row in tracked:
            alias = database.connection.execute('SELECT 1 FROM identity_merges WHERE source=?', (row['person'],)).fetchone()
            if (not alias and row['cluster'] not in mapping and row['person'] not in used
                    and (not row['performer'] or row['performer'] not in performers)):
                mapping[row['cluster']] = row['person']
                used.add(row['person'])
                if row['performer']:
                    performers.add(row['performer'])
    unmatched = [r['id'] for r in replay.execute('SELECT id FROM people WHERE NOT EXISTS '
                 '(SELECT 1 FROM identity_merges WHERE source=people.id) ORDER BY created,id') if r['id'] not in mapping]
    old_groups = []
    if unmatched:
        try:
            old_groups = _groups(database.connection, scope, namespace, config.data['clustering'])
        except ValueError as error:
            # Optional label reuse cannot prevent a smaller cold partition from
            # fixing a historical reference set that exceeds the new budgets.
            if 'budget' not in str(error):
                raise
    with tracking(
        unmatched,
        'Bind rebuilt identities',
        unit='identity',
        position=0,
    ) as tracked:
        for cluster in tracked:
            ranked = []
            samples = ([vector(row) for row in replay.execute(
                'SELECT o.dimensions,o.embedding FROM observations o JOIN members m '
                'ON m.observation=o.id WHERE m.person=? ORDER BY o.id LIMIT ?',
                (cluster, config.data['clustering']['max_samples_per_person']))] if old_groups else [])
            with tracking(
                old_groups,
                'Compare historical identities',
                unit='identity',
            ) as tracked:
                for old in tracked:
                    means, minimum = [], 1.
                    for query in samples:
                        mean, low = _similarities(old, query)
                        means.append(mean)
                        minimum = min(minimum, low)
                    if means and minimum >= config.data['clustering']['cohesion_threshold']:
                        best_two(ranked, (sum(means) / len(means), old.identifier))
            if (ranked and ranked[0][0] >= config.data['recognition']['match_threshold']
                    and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= config.data['recognition']['ambiguity_margin'])):
                candidate = ranked[0][1]
                performer = replay.execute('SELECT performer FROM old_people WHERE id=?', (candidate,)).fetchone()[0]
                if candidate not in used and (not performer or performer not in performers):
                    mapping[cluster] = candidate
                    used.add(candidate)
                    if performer:
                        performers.add(performer)
                    continue
            seed = replay.execute('SELECT k.sample_key FROM sample_keys k JOIN members m ON m.observation=k.observation '
                                  'WHERE m.person=? ORDER BY k.sample_key LIMIT 1', (cluster,)).fetchone()[0]
            attempt = 0
            while True:
                candidate = str(uuid.uuid5(uuid.UUID(database.installation), encoded([scope, namespace, 'rebuild', seed, attempt])))
                if candidate not in used and not database.connection.execute(
                        'SELECT 1 FROM identity_merges WHERE source=?', (candidate,)).fetchone():
                    break
                attempt += 1
            mapping[cluster] = candidate
            used.add(candidate)
    old_groups.clear()
    return mapping


def _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress):
    db, policy = database.connection, matching_policy(config)
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    with work('Apply rebuilt identities', 1, unit='transaction', position=0) as committed:
        with db:
            with tracking(
                _inserted(db, 'INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)',
                          replay.execute('SELECT * FROM origins')),
                'Save original sample ownership',
                unit='member',
                total=lambda: replay.execute(
                    'SELECT COUNT(*) FROM origins',
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
            with tracking(
                mapping.values(),
                'Save rebuilt identities',
                unit='identity',
            ) as tracked:
                for person in tracked:
                    db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
                        (person, scope, namespace, time.time(), None, f'stash-ai-recognition:{database.installation}:{person}'))
            with tracking(
                selected,
                'Replace selected assignments',
                unit='scene',
            ) as tracked:
                for item in tracked:
                    db.execute('DELETE FROM members WHERE observation IN (SELECT o.id FROM observations o JOIN scans s '
                               'ON s.id=o.scan WHERE s.scope=? AND s.namespace=? AND s.scene=?)', (scope, namespace, item['scene']))
                    db.execute('DELETE FROM candidates WHERE scan=? AND policy=?', (item['scan'], policy))
            with tracking(
                _apply_members(db, replay, mapping),
                'Save rebuilt memberships',
                unit='member',
                total=lambda: replay.execute(
                    'SELECT COUNT(*) FROM members',
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
            with tracking(
                _inserted(db, 'INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                          _mapped_candidates(replay, mapping, policy)),
                'Save rebuilt candidates',
                unit='candidate',
                total=lambda: replay.execute(
                    'SELECT COUNT(*) FROM candidates WHERE policy=?',
                    (policy,),
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
            references = _groups(db, scope, namespace, config.data['clustering'], usage=usage, stage='retained')
            context = reference_context(references)
            references.clear()
            with tracking(
                selected,
                'Save reference fingerprints',
                unit='scene',
            ) as tracked:
                for item in tracked:
                    db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?) WHERE scan=? AND policy=?",
                               (context, item['scan'], policy))
            state = {'fingerprint': fingerprint, 'context': context}
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                       ('identity-rebuild:' + digest([scope, namespace]), encoded(state)))
        committed.advance()


def rebuild_selected(database, selected, config, *, progress=None):
    """Replay exactly these scans; all historical membership is label-only input.

    The temporary SQLite uses disk-backed scratch, streaming vectors without
    JPEGs. Only the final assignment replacement commits to the durable DB.
    """
    if not selected:
        return {}
    selected = sorted(selected, key=lambda item: (int(item['scene']), item['scan']))
    db = database.connection
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    if first is None:
        raise ValueError('Identity rebuild requires a selected scan.')
    scope, namespace = first['scope'], first['namespace']
    fingerprint = digest({'format': 1, 'scans': selected, 'policy': matching_policy(config)})
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    state = db.execute('SELECT value FROM metadata WHERE key=?', ('identity-rebuild:' + digest([scope, namespace]),)).fetchone()
    if state and json.loads(state[0]).get('fingerprint') == fingerprint:
        references = _groups(db, scope, namespace, config.data['clustering'], usage=usage)
        context = reference_context(references)
        references.clear()
        if json.loads(state[0]).get('context') == context:
            with tracking(
                selected,
                'Validate cached identity rebuild',
                unit='scene',
                position=0,
            ) as tracked:
                for item in tracked:
                    with tracking(
                        db.execute(
                            'SELECT dimensions,embedding FROM observations WHERE scan=? AND rejected '
                            'IS NULL',
                            (item['scan'],),
                        ),
                        'Validate cached face observations',
                        unit='observation',
                        total=lambda: db.execute(
                            'SELECT COUNT(*) FROM observations WHERE scan=? AND rejected IS NULL',
                            (item['scan'],),
                        ).fetchone()[0],
                    ) as tracked:
                        for row in tracked:
                            vector(row)
            usage.begin('cached')
            return _results(db, selected, config)
    with closing(sqlite3.connect('')) as replay:
        replay.row_factory = sqlite3.Row
        replay.execute('PRAGMA cache_size=-32768')
        _copy_selection(database, replay, selected, scope, namespace)
        # Stable replay identity removes installation-dependent UUID tie order.
        temporary = SimpleNamespace(connection=replay,
            installation=str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([scope, namespace, 'cold-replay']))),
            identity_clock=itertools.count(1).__next__)
        with tracking(
            selected,
            'Rebuild selected identities',
            unit='scene',
            position=0,
        ) as tracked:
            for item in tracked:
                cluster_scene(temporary, item['scan'], config, progress=progress, _cold_replay=True)
        mapping = _bind(database, replay, config, scope, namespace)
        _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress)
    return _results(db, selected, config)


def _results(db, selected, config):
    result = {}
    policy = matching_policy(config)
    def summaries():
        for batch in _batches(selected):
            grouped = {item['scan']: [] for item in batch}
            for row in db.execute('SELECT scan,person,status,COUNT(*) AS observations FROM candidates '
                'WHERE scan IN (SELECT value FROM json_each(?)) AND policy=? '
                'GROUP BY scan,person,status ORDER BY scan,person,status',
                (encoded(list(grouped)), policy)):
                grouped[row['scan']].append({name: row[name] for name in ('person', 'status', 'observations')})
            for item in batch:
                yield item['scan'], grouped[item['scan']]
    with tracking(
        summaries(),
        'Collect rebuilt identities',
        unit='scene',
        position=0,
        total=len(selected),
    ) as tracked:
        for scan, rows in tracked:
            result[scan] = rows
    return result
