"""Cold replay of a selected evidence set, followed by stable identity binding.

Historical identities never seed the replay. Original sample ownership is kept
separately so a later split can reactivate profiles retired by consolidation.
"""
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


def _observations(db, scan):
    return db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'
                      'o.quality,o.rejected,o.dimensions,o.embedding,s.scope,s.namespace,s.scene,s.input_sha FROM observations o '
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


def _copy_observations(db, replay, scan, *, retain_pose=False):
    for batch in _batches(_observations(db, scan)):
        # Source/rect are consumed by sample_key on the authoritative row.
        # Only the pose measurement is needed by opt-in pose-aware matching; never copy JPEGs.
        replay.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
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


def _apply_members(db, replay, mapping):
    rows = replay.execute('SELECT m.observation,m.person,k.sample_key FROM members m '
                          'JOIN sample_keys k ON k.observation=m.observation')
    for batch in _batches(rows):
        db.executemany('INSERT INTO members VALUES (?,?)',
                       ((row['observation'], mapping[row['person']]) for row in batch))
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


@measured('copy')
@stage('Copy selected face evidence')
def _copy_selection(database, replay, selected, scope, namespace, *, retain_pose=False):
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
                _copy_observations(db, replay, item['scan'], retain_pose=retain_pose),
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
@stage('Commit rebuilt identities')
def _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress, *, usage=None,history=None):
    db, policy = database.connection, matching_policy(config)
    usage = usage or ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    with work('Apply rebuilt identities', 1, unit='transaction', position=0) as committed:
        with db:
            from .reference_registry import apply as apply_registry
            registry_changed=apply_registry(database,history,config) if history is not None else False
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
                          _mapped_candidates(replay, mapping, policy, '')),
                'Save rebuilt candidates',
                unit='candidate',
                total=lambda: replay.execute(
                    'SELECT COUNT(*) FROM candidates WHERE policy=?',
                    (policy,),
                ).fetchone()[0],
            ) as tracked:
                for row in tracked:
                    pass
            from .unresolved_pool import apply as apply_pool
            pooled=apply_pool(database,replay,config,mapping)
            from .publication_queue import enqueue
            for item in selected:
                enqueue(db,scope,namespace,item['scene'],config,'selected_identity_rebuild')
            references = _groups(db, scope, namespace, config.data['clustering'], usage=usage, stage='retained',scan_policy=scan_identity(config,namespace),compatibility=fallback_sql(db,config,namespace),
                label_scenes=[item['scene'] for item in selected])
            context = reference_context(references)
            representative_ids = {identifier for group in references for identifier in group.sample_ids}
            references.clear()
            # Current decision roles must exist before selecting references;
            # historical policy rows cannot decide the new bank's authority.
            db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?,"
                "'$.reference_selected',json(CASE WHEN json_extract(payload,'$.observation') IN "
                "(SELECT value FROM json_each(?)) THEN 'true' ELSE 'false' END)) WHERE policy=? "
                'AND scan IN (SELECT value FROM json_each(?))',
                (context,encoded(sorted(representative_ids)),policy,encoded([item['scan'] for item in selected])))
            if pooled:
                db.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_context',?,"
                    "'$.reference_selected',json(CASE WHEN json_extract(payload,'$.observation') IN "
                    "(SELECT value FROM json_each(?)) THEN 'true' ELSE 'false' END)) WHERE id IN "
                    '(SELECT value FROM json_each(?))',(context,encoded(sorted(representative_ids)),encoded(pooled)))
            state = dict(format=2, fingerprint=fingerprint, context=context,
                corroboration_pending=getattr(database,'_rebuild_corroboration_pending',False),
                corroborated_plan=sorted(getattr(database,'_rebuild_corroborated_plan',())),
                inputs=rebuild_cache.input_context(db,selected,scope,namespace,config),
                bindings=rebuild_cache.binding_context(db,scope,namespace),
                limits=rebuild_cache.resource_limits(config),
                resource_reuse=not getattr(database,'_rebuild_optional_budget_fallback',False))
            # Refresh the post-commit dependency stamp even when the external
            # evidence itself did not change, so the next run remains a warm read.
            from .historical_references import context as external_context
            state['fingerprint']=digest(dict(format=2,scans=selected,policy=policy,
                external_references=external_context(db,selected,config,scope,namespace,cache=True)))
            rebuild_cache.save(db,scope,namespace,state)
        committed.advance()


from .run_cache import scope as run_cache_scope


@pipeline_stage('rebuild')
def rebuild_selected(database, selected, config, *, progress=None):
    with profile(database, config.data['execution'].get('profile_rebuild', False)), run_cache_scope(config), compute_scope(config), score_scope(), physical_keys(database, config):
        from .reference_registry import InvalidatedSourceViews
        excluded=set()
        for attempt in range(2):
            try:
                return _rebuild_selected(database,selected,config,progress=progress,excluded_views=excluded)
            except InvalidatedSourceViews as error:
                if attempt or not error.medoids-excluded:raise
                excluded.update(error.medoids)


@measured('rebuild')
def _rebuild_selected(database, selected, config, *, progress=None,excluded_views=()):
    """Replay selected scans, then reconcile against frozen external references.

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
    from . import historical_references
    from .dependencies import read_only as read_only_dependencies
    with stage('Validate external identity dependencies'), read_only_dependencies():
        external_context = historical_references.context(db,selected,config,scope,namespace,cache=True)
    fingerprint = digest({'format': 2, 'scans': selected, 'policy': matching_policy(config),
                          'external_references': external_context})
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'], progress)
    state = rebuild_cache.lookup(db,scope,namespace,fingerprint)
    if state and rebuild_cache.admissible(state,config):
        from .dependencies import certified
        if certified(db,scope,namespace,state):
            from .profiling import count
            count('certified_warm_rebuilds')
            return _results(db,selected,config)
        references = _groups(db,scope,namespace,config.data['clustering'],usage=usage,scan_policy=scan_identity(config,namespace),compatibility=fallback_sql(db,config,namespace),
                           label_scenes=[item['scene'] for item in selected])
        context = reference_context(references)
        references.clear()
        with read_only_dependencies():
            matches = (state.get('context')==context
                and state.get('inputs')==rebuild_cache.input_context(db,selected,scope,namespace,config)
                and state.get('bindings')==rebuild_cache.binding_context(db,scope,namespace))
        if matches:
            with work('Validate cached identity rebuild',len(selected),unit='scene',position=0) as checked:
                with stage('Validate cached identity decisions'):
                    complete=rebuild_cache.complete(db,selected,matching_policy(config),context)
                if complete:
                    checked.advance(len(selected))
            if complete:
                from .dependencies import snapshot,certify
                with snapshot(db):certify(db,scope,namespace,state)
                usage.begin('cached')
                return _results(db,selected,config)
    from .corroborated_views import previous_queries,previous_strict,previous_hints,defer_new_groups
    previous=previous_queries(db,selected,config,scope,namespace)
    prior_strict=set(previous_strict(db,selected,config,scope,namespace))
    hints=previous_hints(db,config,scope,namespace)
    database._rebuild_corroboration_pending=False
    database._rebuild_corroborated_plan={row['id'] for row in hints}
    database._rebuild_revalidated_medoids=set()
    with memory_replay(config) as (replay, replay_uri):
        replay.row_factory = sqlite3.Row
        replay = connection(replay)
        _copy_selection(database, replay, selected, scope, namespace,
                        retain_pose=config.data['clustering']['max_anchor_pose_deviation'] is not None)
        from .evidence_pixels import canonicalize
        canonicalize(replay)
        replay.commit()
        if previous:
            replay.execute('CREATE TEMP TABLE prior_attachment_queries(id TEXT PRIMARY KEY)')
            replay.executemany('INSERT INTO prior_attachment_queries VALUES (?)',((i,) for i in previous))
            replay.commit()
        checkpoints = Checkpoints(database, selected, config, scope, namespace)
        late_context = digest([external_context, sorted(excluded_views), sorted(prior_strict),
            sorted(previous), [(r['id'], r['geometry'], bytes(r['embedding']).hex()) for r in hints]])
        late = Checkpoints(database, selected, config, scope, namespace,
                           phase='selected', context=late_context)
        with read_only_dependencies():
            ready_context = digest([late_context, rebuild_cache.binding_context(db,scope,namespace)])
        ready = Checkpoints(database, selected, config, scope, namespace,
                            phase='ready', context=ready_context)
        with stage('Restore rebuild checkpoint'):
            ready.restore(replay)
            if ready.restored_payload is not None:
                from .ready_checkpoint import restore
                mapping, history = restore(database, replay, ready.restored_payload)
                _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress, usage=usage,history=history)
                return _results(db,selected,config)
            cursor, next_clock = late.restore(replay)
            if late.restored_payload is None:
                cursor, next_clock = checkpoints.restore(replay)
        if cursor < len(selected):
            replay.execute('DROP TABLE IF EXISTS temp.local_core_order')
        clock = ReplayClock(next_clock)
        from .local_cohorts import prepare as prepare_local_cohorts
        with stage('Prepare local face cohorts'):
            if not replay.execute("SELECT 1 FROM sqlite_temp_master WHERE name='local_core_order'").fetchone():
                prepare_local_cohorts(replay,selected,config)
        # Stable replay identity removes installation-dependent UUID tie order.
        temporary = SimpleNamespace(connection=replay,
            installation=str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([scope, namespace, 'cold-replay']))),
            identity_clock=clock)
        from .late_checkpoint import encode as encode_late, decode as decode_late
        if late.restored_payload is not None:
            early, deferred_hints = decode_late(replay, late.restored_payload)
        else:
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
                            with stage('Match selected scene', scene=item['scene']):
                                lookahead.take(item['scan'])
                                lookahead.submit(selected[index+1]['scan'] if index+1<len(selected) else None)
                                cluster_scene(temporary, item['scan'], config, progress=progress, _cold_replay=True)
                                compute_current().scenes_applied += 1
                                compute_current().publish()
                                checkpoints.save(replay, index+1, clock.value)
                # Freeze strict references after every selected scene has matched and
                # merged. Contextual attachments cannot become another scene's anchors
                # or bridge identities during the replay.
                checkpoints.save(replay, len(selected), clock.value, final=True)
                from .strict_partition import resolve as resolve_strict_partition
                with stage('Consolidate strict identity cores'):
                    resolve_strict_partition(temporary, selected, config, progress=progress)
                from .cross_scene import apply_cross_scene_support
                with stage('Confirm cross-scene provisional identities'):
                    apply_cross_scene_support(temporary, selected, config, progress=progress)
                with stage('Classify prior appearance hints'):
                    deferred_hints=defer_new_groups(replay,config,hints,prior_strict)
                from .shared_references import retry_strict, early_views
                with stage('Retry strict identity assignments'):
                    from .recovery_convergence import converge
                    converge(replay, lambda: retry_strict(temporary, selected, config))
                with stage('Build early verified appearance views'):
                    early = early_views(temporary, selected, config,excluded_views=excluded_views)
                with stage('Resolve same-scene identity context'):
                    apply_scene_context(temporary, selected, config, progress=progress)
                from .anchor_views import apply_anchor_views
                with stage('Recover anchor appearance views'):
                    apply_anchor_views(temporary, selected, config, progress=progress)
                from .global_views import apply_global_views
                with stage('Consolidate global appearance views'):
                    global_state=apply_global_views(temporary, selected, config, progress=progress)
                from .cohort_resolution import selected as resolve_selected_cohorts
                from dataclasses import replace
                from .identity_merges import canonical
                early=tuple(replace(v,group=Group(canonical(replay,v.group.identifier),v.group.samples,True)) for v in early)
                with stage('Resolve selected identity cohorts'):
                    resolve_selected_cohorts(temporary,config,early,global_state)
                with stage('Retry changed strict evidence'):
                    from .recovery_convergence import converge
                    converge(replay, lambda: retry_strict(temporary, selected, config))
                global_state=None  # Release frozen exemplars before the historical bank.
            replay.commit()
            late.phase_payload = encode_late(early, deferred_hints)
            if checkpoints.enabled:
                late.save(replay, len(selected), clock.value, final=True)
        with stage('Freeze historical identity references'):
            history = historical_references.capture(db,selected,config,scope,namespace,external_context,resolve_aliases=False)
        with stage('Reconcile historical identities'):
            external_bindings = historical_references.reconcile(database,replay,selected,config,scope,namespace,external_context,
                progress=progress,history=history,early=early,excluded_views=excluded_views)
        from .unresolved_pool import resolve as resolve_pool
        with stage('Resolve retained unassigned faces'):
            resolve_pool(database,replay,selected,config,history,external_bindings,early)
        mapping = _bind(database, replay, config, scope, namespace, usage=usage,external_bindings=external_bindings,
                        selected_scenes=[item['scene'] for item in selected])
        from .shared_references import auxiliary_bank
        with stage('Build auxiliary reference bank'):
            shared_groups, shared_views, confirmations = auxiliary_bank(replay,config,scope,namespace,history,mapping,early)
        with stage('Assign auxiliary faces'):
            apply_auxiliary(temporary, selected, config, progress=progress,reference_groups=shared_groups,
                            reference_views=shared_views,reference_confirmations=confirmations)
        from .shared_references import finalize_early
        finalize_early(replay,config,mapping)
        from .pooled_views import finalize as finalize_pool
        finalize_pool(replay,config,mapping)
        from .selected_corroboration import resolve as restore_corroboration
        with stage('Revalidate corroborated identity views'):
            database._rebuild_corroboration_pending=restore_corroboration(
                database,temporary,selected,config,history,previous,hints=hints)
        from .pose_recovery import recover as recover_uncertain_pose, recover_appearances, recover_joint, freeze, recover_final
        with stage('Recover supported strict appearances'):
            recover_appearances(database,temporary,config,history)
        with stage('Recover uncertain face pose'):
            recover_uncertain_pose(database,temporary,config,history)
        with stage('Recover joint appearance and pose queries'):
            recover_joint(database,temporary,config,history)
        unvalidated=set(deferred_hints)-database._rebuild_revalidated_medoids
        if unvalidated and replay.execute("SELECT 1 FROM candidates c WHERE c.status='withheld' "
                "AND json_extract(c.payload,'$.reason')='prior_appearance_pending' AND EXISTS "
                "(SELECT 1 FROM json_each(json_extract(c.payload,'$.prior_appearance_hints')) h "
                'WHERE h.value IN (SELECT value FROM json_each(?))) LIMIT 1',(encoded(sorted(unvalidated)),)).fetchone():
            database._rebuild_corroboration_pending=True
        for row in replay.execute('SELECT DISTINCT person FROM members'):
            mapping.setdefault(row[0],row[0])
        from .reference_registry import invalidated_sources,InvalidatedSourceViews
        with stage('Validate final reference ownership'):
            invalid=invalidated_sources(history,replay,config,mapping)
        if invalid:raise InvalidatedSourceViews(invalid)
        from .subcomponents import finalize as finalize_subcomponents
        with stage('Finalize supported identity subcomponents'):
            needs_repair=config.data['clustering']['anchor_view_matching'] and replay.execute(
                "SELECT 1 FROM candidates WHERE json_extract(payload,'$.global_view_consolidation.blocker')="
                "'incomplete_strong_component' LIMIT 1").fetchone()
            frozen=freeze(temporary,config,history) if needs_repair else None
            repaired=finalize_subcomponents(temporary,config,mapping,history)
        if repaired:
            with stage('Retry queries against final canonical identities'):
                recover_final(database,temporary,config,history,frozen)
            for row in replay.execute('SELECT DISTINCT person FROM members'):
                mapping.setdefault(row[0],row[0])
            invalid=invalidated_sources(history,replay,config,mapping)
            if invalid:raise InvalidatedSourceViews(invalid)
        replay.commit()
        from .ready_checkpoint import encode as encode_ready
        ready.phase_payload = encode_ready(database, mapping, history)
        if checkpoints.enabled:
            ready.save(replay, len(selected), clock.value, final=True)
        report_pages(replay)
        _apply(database, replay, selected, config, scope, namespace, mapping, fingerprint, progress, usage=usage,history=history)
    return _results(db, selected, config)


@measured('results')
@stage('Collect identity results')
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
