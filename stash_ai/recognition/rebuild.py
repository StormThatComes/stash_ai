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


def sample_key(row):
    return digest([row['scope'], row['namespace'], row['scene'], row['input_sha'],
                   row['cue'], row['seconds'], row['source'], row['rect'], row['geometry']])


def _observations(db, scan):
    return db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,'
                      'o.rejected,o.dimensions,o.embedding,s.scope,s.namespace,s.scene,s.input_sha FROM observations o '
                      'JOIN scans s ON s.id=o.scan WHERE o.scan=? ORDER BY o.cue,o.id', (scan,))


def _copy_selection(database, replay, selected, scope, namespace):
    db = database.connection
    replay.executescript(SCHEMA + '''
      CREATE TABLE sample_keys(observation TEXT PRIMARY KEY, sample_key TEXT NOT NULL);
      CREATE TABLE origins(sample_key TEXT PRIMARY KEY, person TEXT, observation TEXT);
      CREATE TABLE old_people(id TEXT PRIMARY KEY, created REAL, performer TEXT);
    ''')
    for row in db.execute('SELECT id,created,performer FROM people WHERE scope=? AND namespace=?', (scope, namespace)):
        replay.execute('INSERT INTO old_people VALUES (?,?,?)', tuple(row))
    for item in selected:
        record = db.execute('SELECT * FROM scans WHERE id=?', (item['scan'],)).fetchone()
        if (record is None or record['status'] != 'complete' or record['scope'] != scope
                or record['namespace'] != namespace or record['scene'] != item['scene']):
            raise ValueError('Identity rebuild requires complete compatible selected scans.')
        replay.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', tuple(record))
        for row in _observations(db, item['scan']):
            # Stream one embedding at a time; JPEGs never enter the scratch DB.
            replay.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                tuple(row[name] for name in ('id','scan','cue','seconds','source','rect','geometry','quality',
                                             'rejected','dimensions','embedding')) + (None,))
            replay.execute('INSERT INTO sample_keys VALUES (?,?)', (row['id'], sample_key(row)))
        # Seed origins from old memberships only when no older origin exists.
        # Superseded scans are useful for identity labels, never grouping evidence.
        for row in db.execute('SELECT o.cue,o.seconds,o.source,o.rect,o.geometry,o.id,s.scope,s.namespace,s.scene,'
                              's.input_sha,m.person FROM members m JOIN observations o ON o.id=m.observation '
                              'JOIN scans s ON s.id=o.scan WHERE s.scope=? AND s.namespace=? AND s.scene=? '
                              'ORDER BY s.rowid,o.cue,o.id', (scope, namespace, item['scene'])):
            key = sample_key(row)
            replay.execute('INSERT OR IGNORE INTO origins VALUES (?,?,?)', (key, row['person'], row['id']))
        for row in replay.execute('SELECT sample_key FROM sample_keys WHERE observation IN '
                                  '(SELECT id FROM observations WHERE scan=?)', (item['scan'],)):
            old = db.execute('SELECT * FROM identity_origins WHERE sample_key=?', (row[0],)).fetchone()
            if old is not None:
                replay.execute('INSERT OR REPLACE INTO origins VALUES (?,?,?)', tuple(old))
    replay.commit()


def _bind(database, replay, config, scope, namespace):
    """One historical identity per new cluster; IDs cannot change the partition."""
    replay.executescript('''CREATE TABLE votes AS
      SELECT m.person AS cluster,o.person AS person,COUNT(*) AS total
      FROM members m JOIN sample_keys k ON k.observation=m.observation
      JOIN origins o ON o.sample_key=k.sample_key GROUP BY m.person,o.person;''')
    mapping, used, performers = {}, set(), set()
    # Oldest wins a consolidation. On a split its largest overlapping piece wins.
    for row in replay.execute('SELECT v.*,p.performer FROM votes v JOIN old_people p ON p.id=v.person '
                              'ORDER BY p.created,p.id,v.total DESC,v.cluster'):
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
    for cluster in unmatched:
        ranked = []
        for old in old_groups:
            if old.identifier in used:
                continue
            means, minimum = [], 1.
            for row in replay.execute('SELECT o.dimensions,o.embedding FROM observations o JOIN members m '
                                      'ON m.observation=o.id WHERE m.person=? ORDER BY o.id LIMIT ?',
                                      (cluster, config.data['clustering']['max_samples_per_person'])):
                mean, low = _similarities(old, vector(row))
                means.append(mean)
                minimum = min(minimum, low)
            if means and minimum >= config.data['clustering']['cohesion_threshold']:
                ranked.append((sum(means) / len(means), old.identifier))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if (ranked and ranked[0][0] >= config.data['recognition']['match_threshold']
                and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= config.data['recognition']['ambiguity_margin'])):
            candidate = ranked[0][1]
            performer = replay.execute('SELECT performer FROM old_people WHERE id=?', (candidate,)).fetchone()[0]
            if not performer or performer not in performers:
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
    with db:
        for row in replay.execute('SELECT * FROM origins'):
            db.execute('INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)', tuple(row))
        for person in mapping.values():
            db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
                (person, scope, namespace, time.time(), None, f'stash-ai-recognition:{database.installation}:{person}'))
        for item in selected:
            db.execute('DELETE FROM members WHERE observation IN (SELECT o.id FROM observations o JOIN scans s '
                       'ON s.id=o.scan WHERE s.scope=? AND s.namespace=? AND s.scene=?)', (scope, namespace, item['scene']))
            db.execute('DELETE FROM candidates WHERE scan=? AND policy=?', (item['scan'], policy))
        for row in replay.execute('SELECT * FROM members'):
            person = mapping[row['person']]
            db.execute('INSERT INTO members VALUES (?,?)', (row['observation'], person))
            key = replay.execute('SELECT sample_key FROM sample_keys WHERE observation=?', (row['observation'],)).fetchone()[0]
            db.execute('INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)', (key, person, row['observation']))
        for row in replay.execute('SELECT * FROM candidates WHERE policy=?', (policy,)):
            payload = json.loads(row['payload'])
            # A provisional group may be withheld and have no accepted person.
            person = mapping.get(row['person'])
            payload['identity_basis'] = 'cold_selected_replay'
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                (row['id'], row['scan'], policy, person, row['status'], encoded(payload)))
        references = _groups(db, scope, namespace, config.data['clustering'], usage=usage, stage='retained')
        context = reference_context(references)
        references.clear()
        for item in selected:
            db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?) WHERE scan=? AND policy=?",
                       (context, item['scan'], policy))
        state = {'fingerprint': fingerprint, 'context': context}
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   ('identity-rebuild:' + digest([scope, namespace]), encoded(state)))


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
            for item in selected:
                for row in db.execute('SELECT dimensions,embedding FROM observations WHERE scan=? AND rejected IS NULL', (item['scan'],)):
                    vector(row)
            usage.begin('cached')
            return {item['scan']: _result(db, item['scan'], matching_policy(config)) for item in selected}
    with closing(sqlite3.connect('')) as replay:
        replay.row_factory = sqlite3.Row
        replay.execute('PRAGMA cache_size=-32768')
        _copy_selection(database, replay, selected, scope, namespace)
        # Stable replay identity removes installation-dependent UUID tie order.
        temporary = SimpleNamespace(connection=replay,
            installation=str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([scope, namespace, 'cold-replay']))),
            identity_clock=itertools.count(1).__next__)
        for item in selected:
            cluster_scene(temporary, item['scan'], config, progress=progress)
        mapping = _bind(database, replay, config, scope, namespace)
        _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress)
    return {item['scan']: _result(db, item['scan'], matching_policy(config)) for item in selected}
