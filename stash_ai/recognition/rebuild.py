"""Cold replay of a selected evidence set, followed by stable identity binding.

Historical identities never seed the replay. Original sample ownership is kept
separately so a later split can reactivate profiles retired by consolidation.
"""
from .evidence_pixels import alias_scope
from .profiling import pipeline_stage
from stash_ai.runtime.cleanup import closing
import json
import itertools
import sqlite3
import time
from types import SimpleNamespace
import uuid

from .clustering import Group, _groups, _result, _similarities, cluster_scene, matching_policy, reference_context, vector, complete_candidates
from .config import digest
from .database import SCHEMA, encoded
from .reference_usage import ReferenceUsage
from .progress import tracking, work, stage
from .ranking import best_two
from .profiling import measured, profile, connection
from .reference_index import references as reference_scope
from .scoring import score_scope
from .scanning import scan_identity
from .reference_compatibility import fallback_sql
from .evidence_cache import evidence_scope
from .scene_context import apply_scene_context
from .auxiliary import apply_auxiliary
from .pose_policy import compact_quality
from .sample_keys import sample_key, physical_keys
from .replay_addresses import addresses
from .replay_memory import memory_replay, Lookahead, report_pages
from . import rebuild_cache
from .replay_checkpoint import Checkpoints, ReplayClock
from .compute import compute_scope, current as compute_current
from .parallel_scores import FrozenScores


def _observations(db, scan, *, share_vectors=False):
    return db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'
                      'o.quality,o.rejected,o.dimensions,'+('NULL AS embedding' if share_vectors else 'o.embedding')+
                      ',s.scope,s.namespace,s.scene,s.input_sha FROM observations o '
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


def _copy_observations(db, replay, scan, *, retain_pose=False, share_vectors=False):
    for batch in _batches(_observations(db, scan, share_vectors=share_vectors)):
        from .pixel_storage import copy_rows
        copy_rows(db,replay,batch)
        # Source/rect are consumed by sample_key on the authoritative row.
        # Only the pose measurement is needed by opt-in pose-aware matching; never copy JPEGs.
        replay.executemany('INSERT INTO main.observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           ((row['id'], row['scan'], row['cue'], row['seconds'], '', '[]',
                             row['geometry'], compact_quality(row['quality']),
                              row['rejected'], row['dimensions'],
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


def _apply_members(db, replay, mapping, changed=None):
    rows = replay.execute('SELECT m.observation,m.person,k.sample_key FROM members m '
                          'JOIN sample_keys k ON k.observation=m.observation')
    for batch in _batches(rows):
        proposed = encoded([[r['observation'], mapping[r['person']]] for r in batch])
        source = ("SELECT json_extract(value,'$[0]') AS observation, "
                  "json_extract(value,'$[1]') AS person FROM json_each(?)")
        if changed is not None:
            changed.update(row[0] for row in db.execute(
                'WITH proposed AS ('+source+') SELECT DISTINCT s.scene FROM proposed p '
                'JOIN observations o ON o.id=p.observation JOIN scans s ON s.id=o.scan '
                'LEFT JOIN members m ON m.observation=p.observation WHERE m.person IS NOT p.person',
                (proposed,)))
        db.execute('INSERT INTO members SELECT observation,person FROM ('+source+') WHERE 1 '
                   'ON CONFLICT(observation) DO UPDATE SET person=excluded.person '
                   'WHERE members.person IS NOT excluded.person', (proposed,))
        db.executemany('INSERT OR IGNORE INTO identity_origins VALUES (?,?,?)',
                       ((row['sample_key'], mapping[row['person']], row['observation']) for row in batch))
        yield from batch


def _mapped_candidates(replay, mapping, policy, context, representative_ids=()):
    for row in replay.execute('SELECT * FROM candidates WHERE policy=?', (policy,)):
        payload = json.loads(row['payload'])
        from .reference_roles import STRICT_REASONS
        payload['reference_selected'] = payload.get('observation') in representative_ids
        payload['reference_role'] = ('strict' if row['status'] == 'accepted' and payload.get('reason') in STRICT_REASONS else 'attachment')
        payload['identity_basis'] = ('selected_replay_with_frozen_external_references'
                                     if payload.get('historical_reference') else 'cold_selected_replay')
        if payload.get('provisional_pool'):
            payload['identity_basis']='frozen_unresolved_pool'
            payload['provisional_pool']['identity']=mapping.get(row['person'])
        payload['reference_context'] = context
        yield (row['id'], row['scan'], policy, mapping.get(row['person']), row['status'], encoded(payload))


REPLAY_SCHEMA = SCHEMA + '''
      CREATE TABLE sample_keys(observation TEXT PRIMARY KEY, sample_key TEXT NOT NULL);
      CREATE TABLE origins(sample_key TEXT PRIMARY KEY, person TEXT, observation TEXT);
      CREATE TABLE old_people(id TEXT PRIMARY KEY, created REAL, performer TEXT, retired INTEGER);
    '''


@measured('copy')
@stage('Copy selected face evidence')
def _copy_selection(database, replay, selected, scope, namespace, *, retain_pose=False,bindings=True,inputs=None):
    db = database.connection
    replay.executescript(REPLAY_SCHEMA)
    from .replay_vector_inputs import prepare
    shared_vectors = prepare(db, replay)
    from .snapshot_import import bulk
    from .scene_input_shards import open_shards
    with bulk(replay),open_shards(inputs,shared_vectors) as shards:
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
                if shards is not None and shards.restore(replay,item['scan']):continue
                with tracking(
                    _copy_observations(db, replay, item['scan'], retain_pose=retain_pose, share_vectors=shared_vectors),
                    'Copy face observations',
                    unit='observation',
                    total=lambda: db.execute(
                        'SELECT COUNT(*) FROM observations WHERE scan=?',
                        (item['scan'],),
                    ).fetchone()[0],
                ) as tracked:
                    for row in tracked:
                        pass  # The bounded copy iterator has completed both inserts.
                if shards is not None:shards.save(replay,item['scan'])
    if bindings:_copy_bindings(database,replay,selected,scope,namespace)
    with work('Commit replay evidence', 1, unit='transaction', position=0) as committed:
        replay.commit()
        committed.advance()


def _copy_bindings(database,replay,selected,scope,namespace):
    """Refresh mutable ownership independently of the immutable raw image."""
    db=database.connection
    replay.execute('DELETE FROM old_people')
    replay.execute('DELETE FROM origins')
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
    replay.commit()


@measured('binding')
@stage('Bind stable performer identities')
def _bind(database, replay, config, scope, namespace, *, usage=None, external_bindings=None, selected_scenes=()):
    """Bind stable labels, including independently verified external consolidations."""
    database._rebuild_optional_budget_fallback = False
    with work('Compute identity votes', 1, unit='query', position=0) as votes:
        replay.execute('CREATE TABLE votes(cluster TEXT,person TEXT,total INTEGER,performer TEXT,retired INTEGER)')
        inserted = replay.execute('INSERT INTO votes SELECT m.person,o.person,COUNT(*),p.performer,p.retired '
            'FROM members m JOIN sample_keys k ON k.observation=m.observation '
            'JOIN origins o ON o.sample_key=k.sample_key JOIN old_people p ON p.id=o.person '
            'GROUP BY m.person,o.person ORDER BY p.created,p.id,COUNT(*) DESC,m.person')
        vote_count = inserted.rowcount
        votes.advance()
    mapping = dict(external_bindings or {})
    used = set(mapping.values())
    performers = {row[0] for row in replay.execute('SELECT performer FROM old_people WHERE id IN '
        '(SELECT value FROM json_each(?)) AND performer IS NOT NULL',(encoded(sorted(used)),))}
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
                                 usage=usage, stage='historical_binding',scan_policy=scan_identity(config,namespace),label_scenes=selected_scenes,
                                 compatibility=fallback_sql(database.connection,config,namespace))
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
            from .reference_selection import select
            from .reference_roles import strict_sql
            from .pose_policy import anchor_sql
            samples = ([sample for _, sample in select(replay.execute(
                'SELECT o.id,o.dimensions,o.embedding,o.quality,o.geometry,o.seconds,s.scene FROM observations o JOIN members m ON m.observation=o.id '
                'JOIN scans s ON s.id=o.scan WHERE m.person=? AND ' +
                anchor_sql('o', config.data['clustering']) + ' AND ' + strict_sql() +
                ' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
                (cluster,)), config.data['clustering']['max_samples_per_person'])] if old_groups else [])
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
                    and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= config.data[