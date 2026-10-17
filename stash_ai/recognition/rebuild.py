"""Cold replay of a selected evidence set, followed by stable identity binding.

Historical identities never seed the replay. Original sample ownership is kept
separately so a later split can reactivate profiles retired by consolidation.
"""
from stash_ai.runtime.cleanup import closing
import json
import itertools
import sqlite3
import time
from types import SimpleNamespace
import uuid

from .clustering import _groups, _result, _similarities, cluster_scene, matching_policy, reference_context, vector, complete_candidates
from .config import digest
from .database import SCHEMA, encoded
from .reference_usage import ReferenceUsage
from .progress import tracking, work
from .ranking import best_two
from .profiling import measured, profile, connection
from .reference_index import references as reference_scope
from .scoring import score_scope
from .evidence_cache import evidence_scope
from .scene_context import apply_scene_context
from .sample_keys import sample_key, physical_keys
from .replay_addresses import addresses
from .replay_memory import memory_replay, Lookahead, report_pages
from . import rebuild_cache
from .replay_checkpoint import Checkpoints, ReplayClock
from .compute import compute_scope
from .parallel_scores import FrozenScores


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


def _restore_origins(db, replay):
    keys = replay.execute('SELECT sample_key FROM sample_keys')
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


def _mapped_candidates(replay, mapping, policy, context):
    for row in replay.execute('SELECT * FROM candidates WHERE policy=?', (policy,)):
        payload = json.loads(row['payload'])
        payload['identity_basis'] = 'cold_selected_replay'
        payload['reference_context'] = context
        yield (row['id'], row['scan'], policy, mapping.get(row['person']), row['status'], encoded(payload))


@measured('copy')
def _copy_selection(database, replay, selected, scope, namespace):
    db = database.connection
    replay.executescript(SCHEMA + '''
      CREATE TABLE sample_keys(observation TEXT PRIMARY KEY, sample_key TEXT NOT NULL);
      CREATE TABLE origins(sample_key TEXT PRIMARY KEY, person TEXT, observation TEXT);
      CREATE TABLE old_people(id TEXT PRIMARY KEY, created REAL, performer TEXT, retired INTEGER);
    ''')
    with tracking(
        db.execute(
            'SELECT id,created,performer,EXISTS(SELECT 1 FROM identity_merges WHERE source=people.id) '
            'FROM people WHERE scope=? AND namespace=?',
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
            replay.execute('INSERT INTO old_people VALUES (?,?,?,?)', tuple(row))
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
    # Physical keys include the scene, so batching scene traversal preserves
    # first-origin precedence within each historical scan/cue/observation order.
    for batch in _batches(selected):
        scenes = encoded([item['scene'] for item in batch])
        historical = db.execute('SELECT o.cue,o.seconds,o.source,o.rect,o.geometry,o.id,'
            's.scope,s.namespace,s.scene,s.input_sha,m.person FROM members m '
            'JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
            'WHERE s.scope=? AND s.namespace=? AND s.scene IN (SELECT value FROM json_each(?)) '
            'ORDER BY s.rowid,o.cue,o.id', (scope,namespace,scenes))
        with tracking(_inserted(replay, 'INSERT OR IGNORE INTO origins VALUES (?,?,?)',
                ((sample_key(row),row['person'],row['id']) for row in historical)),
                'Read historical sample ownership', unit='member') as tracked:
            for row in tracked:
                pass
    with tracking(_restore_origins(db,replay), 'Restore original sample ownership',
                  unit='observation') as tracked:
        for row in tracked:
            pass
    with work('Commit replay evidence', 1, unit='transaction', position=0) as committed:
        replay.commit()
        committed.advance()


@measured('binding')
def _bind(database, replay, config, scope, namespace, *, usage=None):
    """One historical identity per new cluster; IDs cannot change the partition."""
    database._rebuild_optional_budget_fallback = False
    with work('Compute identity votes', 1, unit='query', position=0) as votes:
        replay.execute('CREATE TABLE votes(cluster TEXT,person TEXT,total INTEGER,performer TEXT,retired INTEGER)')
        inserted = replay.execute('INSERT INTO votes SELECT m.person,o.person,COUNT(*),p.performer,p.retired '
            'FROM members m JOIN sample_keys k ON k.observation=m.observation '
            'JOIN origins o ON o.sample_key=k.sample_key JOIN old_people p ON p.id=o.person '
            'GROUP BY m.person,o.person ORDER BY p.created,p.id,COUNT(*) DESC,m.person')
        vote_count = inserted.rowcount
        votes.advance()
    mapping, used, performers = {}, set(), set()
    # Oldest wins a consolidation. On a split its largest overlapping piece wins.
    with tracking(
        replay.execute('SELECT * FROM votes ORDER BY rowid'),
        'Bind historical identity votes', unit='vote', position=0, total=vote_count,
    ) as tracked:
        for row in tracked:
            if (not row['retired'] and row['cluster'] not in mapping and row['person'] not in used
                    and (not row['performer'] or row['performer'] not in performers)):
                mapping[row['cluster']] = row['person']
                used.add(row['person'])
                if row['performer']:
                    performers.add(row['performer'])
    unmatched = [r['id'] for r in replay.execute('SELECT id FROM people WHERE NOT EXISTS '
                 '(SELECT 1 FROM identity_merges WHERE source=people.id) AND EXISTS '
                 '(SELECT 1 FROM members WHERE person=people.id) ORDER BY created,id') if r['id'] not in mapping]
    old_groups = []
    if unmatched:
        try:
            old_groups = _groups(database.connection, scope, namespace, config.data['clustering'],
                                 usage=usage, stage='historical_binding')
        except ValueError as error:
            # Optional label reuse cannot prevent a smaller cold partition from
            # fixing a historical reference set that exceeds the new budgets.
            if 'budget' not in str(error):
                raise
            database._rebuild_optional_budget_fallback = True
    historical_performers = {}
    historical_scores = FrozenScores(old_groups)
    for batch in _batches(old_groups):
        historical_performers.update(replay.execute('SELECT id,performer FROM old_people WHERE id IN '
            '(SELECT value FROM json_each(?))', (encoded([g.identifier for g in batch]),)))
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
                'ON m.observation=o.id JOIN scans s ON s.id=o.scan WHERE m.person=? '
                'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id LIMIT ?',
                (cluster, config.data['clustering']['max_samples_per_person']))] if old_groups else [])
            parallel = historical_scores.score(tuple(samples), float('-inf'),
                                                config.data['clustering']['cohesion_threshold'])
            contributions = None
            if parallel is not None:
                contributions = {}
                for values in parallel:
                    for position, mean, low in values:
                        contributions.setdefault(position, []).append((mean, low))
            with tracking(
                old_groups,
                'Compare historical identities',
                unit='identity',
            ) as tracked:
                for position, old in enumerate(tracked):
                    means, minimum = [], 1.
                    if contributions is None:
                        for query in samples:
                            mean, low = _similarities(old, query)
                            means.append(mean)
                            minimum = min(minimum, low)
                    else:
                        values = contributions.get(position, [])
                        if len(values) == len(samples):
                            means = [mean for mean, _ in values]
                            minimum = min([1.] + [low for _, low in values])
                    if means and minimum >= config.data['clustering']['cohesion_threshold']:
                        best_two(ranked, (sum(means) / len(means), old.identifier))
            if (ranked and ranked[0][0] >= config.data['recognition']['match_threshold']
                    and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= config.data['recognition']['ambiguity_margin'])):
                candidate = ranked[0][1]
                performer = historical_performers[candidate]
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


@measured('apply')
def _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress, *, usage=None):
    db, policy = database.connection, matching_policy(config)
    usage = usage or ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
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
            references = _groups(db, scope, namespace, config.data['clustering'], usage=usage, stage='retained')
            context = reference_context(references)
            references.clear()
            with tracking(
                _inserted(db, 'INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                          _mapped_candidates(replay, mapping, policy, context)),
                'Save rebuilt candidates',
                unit='candidate',
                total=lambda: replay.execute(
                    'SELECT COUNT(*) FROM candidates WHERE policy=?',
                    (policy,),
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
            state = dict(format=2, fingerprint=fingerprint, context=context,
                inputs=rebuild_cache.input_context(db,selected,scope,namespace),
                bindings=rebuild_cache.binding_context(db,scope,namespace),
                limits=rebuild_cache.resource_limits(config),
                resource_reuse=not getattr(database,'_rebuild_optional_budget_fallback',False))
            rebuild_cache.save(db,scope,namespace,state)
        committed.advance()


def rebuild_selected(database, selected, config, *, progress=None):
    with profile(database, config.data['execution'].get('profile_rebuild', False)), compute_scope(config), score_scope(), physical_keys(database, config):
        return _rebuild_selected(database, selected, config, progress=progress)


@measured('rebuild')
def _rebuild_selected(database, selected, config, *, progress=None):
    """Replay exactly these scans; all historical membership is label-only input.

    The private SQLite stays in RAM, streaming vectors without
    JPEGs. Only the final assignment replacement commits to the durable DB.
    """
    if not selected:
        return {}
    scenes = [int(item['scene']) for item in selected]
    if len(set(scenes)) != len(scenes):
        raise ValueError('Identity rebuild requires exactly one selected scan per scene.')
    selected = sorted(selected, key=lambda item: (int(item['scene']), item['scan']))
    db = database.connection
    first = db.execute('SELECT scope,namespace FROM scans WHERE id=?', (selected[0]['scan'],)).fetchone()
    if first is None:
        raise ValueError('Identity rebuild requires a selected scan.')
    scope, namespace = first['scope'], first['namespace']
    fingerprint = digest({'format': 1, 'scans': selected, 'policy': matching_policy(config)})
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    state = rebuild_cache.lookup(db,scope,namespace,fingerprint)
    if state and rebuild_cache.admissible(state,config):
        references = _groups(db,scope,namespace,config.data['clustering'],usage=usage)
        context = reference_context(references)
        references.clear()
        if (state.get('context')==context
                and state.get('inputs')==rebuild_cache.input_context(db,selected,scope,namespace)
                and state.get('bindings')==rebuild_cache.binding_context(db,scope,namespace)):
            with work('Validate cached identity rebuild',len(selected),unit='scene',position=0) as checked:
                complete=rebuild_cache.complete(db,selected,matching_policy(config),context)
                if complete:
                    checked.advance(len(selected))
            if complete:
                usage.begin('cached')
                return _results(db,selected,config)
    with memory_replay(config) as (replay, replay_uri):
        replay.row_factory = sqlite3.Row
        replay = connection(replay)
        _copy_selection(database, replay, selected, scope, namespace)
        checkpoints = Checkpoints(database, selected, config, scope, namespace)
        cursor, next_clock = checkpoints.restore(replay)
        clock = ReplayClock(next_clock)
        # Stable replay identity removes installation-dependent UUID tie order.
        temporary = SimpleNamespace(connection=replay,
            installation=str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([scope, namespace, 'cold-replay']))),
            identity_clock=clock)
        with addresses(replay), reference_scope(replay, scope, namespace, config), evidence_scope(replay, config):
            with Lookahead(replay_uri, config) as lookahead:
                # Prime the first remaining scene as well: with explicit
                # prefetch enabled no selected scene may silently go serial.
                if cursor < len(selected):
                    lookahead.submit(selected[cursor]['scan'])
                with tracking(
                    selected[cursor:],
                    'Rebuild selected identities',
                    total=len(selected), initial=cursor,
                    unit='scene',
                    position=0,
                ) as tracked:
                    for index, item in enumerate(tracked, start=cursor):
                        lookahead.take(item['scan'])
                        lookahead.submit(selected[index+1]['scan'] if index+1<len(selected) else None)
                        cluster_scene(temporary, item['scan'], config, progress=progress, _cold_replay=True)
                        checkpoints.save(replay, index+1, clock.value)
            # Freeze strict references after every selected scene has matched and
            # merged. Contextual attachments cannot become another scene's anchors
            # or bridge identities during the replay.
            checkpoints.save(replay, len(selected), clock.value, final=True)
            apply_scene_context(temporary, selected, config, progress=progress)
        report_pages(replay)
        mapping = _bind(database, replay, config, scope, namespace, usage=usage)
        _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress, usage=usage)
    return _results(db, selected, config)


@measured('results')
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
